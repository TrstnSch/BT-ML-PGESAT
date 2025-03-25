import torch
import torch.nn as nn

from mlp import MLP

class NeuroSAT(nn.Module):
    def __init__(self, opts, device):
        super(NeuroSAT, self).__init__()
        self.device = device
        self.opts = opts

        self.init_tensor = torch.ones(1, device=self.device)
        self.init_tensor.requires_grad = False

        self.L_init = nn.Linear(in_features=1, out_features=opts['emb_dim']).to(self.device)
        self.C_init = nn.Linear(in_features=1, out_features=opts['emb_dim']).to(self.device)

        self.L_msg = MLP(input_size=opts['emb_dim'], hidden_size=opts['emb_dim'], output_size=opts['emb_dim']).to(self.device)
        self.C_msg = MLP(input_size=opts['emb_dim'], hidden_size=opts['emb_dim'], output_size=opts['emb_dim']).to(self.device)

        self.L_update = nn.LSTM(input_size=opts['emb_dim']*2, hidden_size=opts['emb_dim'],).to(self.device)
        self.C_update = nn.LSTM(input_size=opts['emb_dim'], hidden_size=opts['emb_dim'],).to(self.device)

        self.lnorm_l = nn.LayerNorm(opts['emb_dim']).to(self.device)
        self.lnorm_c = nn.LayerNorm(opts['emb_dim']).to(self.device)

        self.L_vote = MLP(input_size=opts['emb_dim'], hidden_size=opts['emb_dim'], output_size=1).to(self.device)

        self.denom = torch.sqrt(torch.tensor([opts['emb_dim']], device=self.device))

    def forward(self, problem, edge_weights=None):
        n_variables = problem.n_variables
        n_literals = problem.n_literals
        n_clauses = len(problem.clauses)
        n_problems = len(problem.is_sat)
        n_vars_batch = n_variables // n_problems

        edges = torch.tensor(problem.batch_edges, device=self.device).t().long()

        init_tensor = self.init_tensor
        L_init = self.L_init(init_tensor).view(1,1,-1)
        L_init = L_init.repeat(1, n_literals, 1).to(self.device)
        C_init = self.C_init(init_tensor).view(1,1,-1)
        C_init = C_init.repeat(1, n_clauses, 1).to(self.device)

        L_state = (L_init, torch.zeros(1, n_literals, self.opts['emb_dim'], device=self.device))
        C_state = (C_init, torch.zeros(1, n_clauses, self.opts['emb_dim'], device=self.device))

        connections = torch.sparse_coo_tensor(
            indices=edges,
            values=torch.ones(problem.n_cells, device=self.device) if edge_weights==None else edge_weights,             # TODO: edge_weights need to be in a specific shape!! Sum of all cells = n_cells
            size=torch.Size([n_literals, n_clauses])
        ).to_dense()

        iteration_votes = []
        all_literal_embeddings = []  # Speicher für Literal-Einbettungen nach jeder Iteration
        all_clause_embeddings = []  # Speicher für Klausel-Einbettungen nach jeder Iteration

        for _ in range(self.opts['iterations']):
            L_hidden = self.lnorm_l(L_state[0].squeeze(0))
            L_pre_msg = self.L_msg(L_hidden)
            LC_msg = torch.matmul(connections.t(), L_pre_msg)

            _, C_state = self.C_update(LC_msg.unsqueeze(0), C_state)

            C_hidden = self.lnorm_c(C_state[0].squeeze(0))
            C_pre_msg = self.C_msg(C_hidden)
            CL_msg = torch.matmul(connections, C_pre_msg)

            _, L_state = self.L_update(torch.cat([CL_msg, self.flip(L_state[0].squeeze(0), n_variables)],
                                                 dim=1).unsqueeze(0), L_state)

            # Speichere die aktuellen Literal- und Klauselzustände
            all_literal_embeddings.append(L_state[0].squeeze(0).clone())  # Shape: (n_literals, emb_dim)

            all_clause_embeddings.append(C_state[0].squeeze(0).clone())  # Shape: (n_clauses, emb_dim)

            # Berechnung der Votes nach der Iteration
            logits = L_state[0].squeeze(0)
            vote = self.L_vote(logits)
            iteration_votes.append(vote.clone())  # Votes speichern
            iteration_votes_sorted = []
            for vote in iteration_votes:
                # `vote` hat die Form [n_literals, 1]
                # Trennen in positive und negative Literale
                vote_join = torch.cat([vote[:n_variables, :], vote[n_variables:, :]], dim=1)  # [n_variables, 2]

                # Gruppieren nach Problemen
                vote_per_problem = vote_join.view(n_problems, n_vars_batch, 2)  # [n_problems, n_vars_per_batch, 2]

                # Optional: Weiterverarbeitung, z.B. Mittelwert oder andere Aggregationen
                iteration_votes_sorted.append(vote_per_problem)


        logits = L_state[0].squeeze(0)
        clauses = C_state[0].squeeze(0)

        vote = self.L_vote(logits)
        vote_join = torch.cat([vote[:n_variables,:], vote[n_variables:,:]], dim=1)              # Concatenates positive and negative literals
        self.vote = vote_join
        vote_join = vote_join.view(n_variables, -1, 2).view(n_problems, -1)                     # First view transforms concatenated literals to [[pos, neg], []...] over complete problem batch. Second view divides into problems!
        vote_mean = torch.mean(vote_join, dim=1)

        return vote_mean, iteration_votes_sorted, all_literal_embeddings, all_clause_embeddings
        #return vote_mean, iteration_votes_sorted
        #return vote_mean

    def flip(self, msg, n_variables):
        return torch.cat([msg[n_variables:2*n_variables, :], msg[:n_variables, :]], dim=0)




