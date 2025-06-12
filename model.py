import torch
import numpy as np
import torch.nn.functional as F

class Flatten(torch.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        bs = x.shape[0]
        return x.view(bs, -1)

class Encoder(torch.nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()

        self.zdim = args.zdim

        self.extracter = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, 2, padding=0, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 32, 2, padding=0, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 2, padding=0, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, 2, padding=0, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 256, 2, padding=0, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 512, 2, padding=0, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, 2, padding=0, stride=2),
            torch.nn.MaxPool2d(2, stride=2)
        )

        self.norm = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(512),
        )

        self.aggr1 = torch.nn.Sequential(
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
        )

        self.aggr2_out = torch.nn.Sequential(
            torch.nn.BatchNorm1d(512),
            torch.nn.Tanh()
        )

        self.weight_relative = torch.nn.Parameter(torch.empty(20, 512))
        torch.nn.init.xavier_normal_(self.weight_relative)

        self.mu_predictor = torch.nn.Linear(512, self.zdim)
        self.logvar_predictor = torch.nn.Linear(512, self.zdim)

    def encode(self, graphs, input_masks, relative_distance, abs_pe):
        bs, graph_num, h, w, _ = graphs.shape  # [N, M, d, d, 1]
        graphs = graphs.reshape(bs * graph_num, h, w, 1).permute(0, 3, 1, 2)
        cn1_out = self.extracter(graphs)
        cn1_out = cn1_out.reshape(bs, graph_num, 512)

        whole_cn1_out, part_cn1_out = torch.split(cn1_out, [1, graph_num - 1], dim=1)
        coff = ((part_cn1_out.unsqueeze(dim=1).expand(-1, graph_num - 1, -1, -1) * part_cn1_out.unsqueeze(dim=2).expand(-1, -1, graph_num - 1, -1)).sum(dim=3)
                / (part_cn1_out.unsqueeze(dim=1).expand(-1, graph_num - 1, -1, -1).norm(p=2, dim=3) * part_cn1_out.unsqueeze(dim=2).expand(-1, -1, graph_num - 1, -1).norm(p=2, dim=3) + 1e-10))
        winner_1 = F.one_hot(torch.argmax(coff, dim=2), graph_num - 1)
        winner_2 = F.one_hot(torch.argmax(coff - coff * winner_1, dim=2), graph_num - 1)
        winner_3 = F.one_hot(torch.argmax(coff - coff * (winner_1 + winner_2), dim=2), graph_num - 1)
        edge_weight = coff * (winner_1 + winner_2 * 0.5 + winner_3 * 0.2)

        slide_mask, _ = torch.split(input_masks, [1, graph_num - 2], dim=2)

        masked_coff = torch.cat([torch.cat([torch.ones([bs, 1, 1]), torch.zeros([bs, 1, graph_num - 1])], dim=2).to(slide_mask.device),
                                 torch.cat([0.5 * slide_mask, edge_weight * input_masks], dim=2)], dim=1)

        coff_adjs = masked_coff / (masked_coff.sum(dim=2, keepdims=True).expand(-1, -1, graph_num) + 1e-10)

        bn1_out = self.norm(cn1_out.permute(0, 2, 1))

        # relative PE
        relative_embeddings = torch.matmul(F.one_hot(relative_distance, 20).float(), self.weight_relative)  # bs * 20 * 20 * 512
        relative_pe = torch.nn.Tanh()(torch.matmul(part_cn1_out.unsqueeze(dim=2), relative_embeddings.permute(0, 1, 3, 2)).squeeze(dim=2))  # bs * 20 * 20

        cn2_out = (torch.matmul(coff_adjs, bn1_out.permute(0, 2, 1))  # bs * 21 * 512
                   + (coff_adjs * torch.cat([torch.zeros([bs, 1, graph_num]).to(bn1_out.device), torch.cat([torch.zeros([bs, graph_num - 1, 1]).to(bn1_out.device), relative_pe], dim=2)], dim=1)).sum(dim=2, keepdim=True)
                   + torch.cat([-1.1 * torch.ones([bs, 1, 512]).to(bn1_out.device), abs_pe * slide_mask.expand(-1, -1, 512)], dim=1)).reshape(bs * graph_num, 512)
        aggr1_out = self.aggr1(cn2_out)

        gcn_out = (aggr1_out + cn2_out).reshape(bs, graph_num, 512).sum(dim=1)

        bn2_out = self.aggr2_out(gcn_out)

        mu = self.mu_predictor(bn2_out)
        logvar = self.logvar_predictor(bn2_out)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(logvar)
        return mu + std * eps

    def forward(self, input_graphs, input_masks, relative_distance, abs_pe):
        mu, logvar = self.encode(input_graphs, input_masks, relative_distance, abs_pe)
        z = self.reparameterize(mu, logvar)

        return z, mu, logvar

class Decoder(torch.nn.Module):
    def __init__(self, args, zdim=128):
        super(Decoder, self).__init__()
        self.hidden_size = args.decoder_dim
        self.num_layers = 1
        self.zdim = args.zdim
        self.num_gaussians = args.num_gaussians

        # Maps the latent vector to an initial cell/hidden vector
        self.hidden_cell_predictor = torch.nn.Sequential(
            torch.nn.Linear(zdim, 2 * self.hidden_size),
            torch.nn.Tanh()
        )

        self.lstm = torch.nn.LSTM(
            5 + self.zdim,
            self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True)

        self.parameters_predictor = torch.nn.Linear(self.hidden_size, self.num_gaussians + 5 * self.num_gaussians + 3)

    def get_mixture_params(self, output):
        pen_logits = output[:, 0:3]
        pi, mu1, mu2, sigma1, sigma2, corr = torch.split(output[:, 3:], self.num_gaussians, dim=1)

        pi = torch.nn.functional.softmax(pi, dim=-1)
        pen = torch.nn.functional.softmax(pen_logits, dim=-1)

        sigma1 = sigma1.exp()
        sigma2 = sigma2.exp()
        corr = torch.nn.Tanh()(corr)

        return pi, mu1, mu2, sigma1, sigma2, corr, pen, pen_logits

    def forward(self, inputs, z, hidden_and_cell=None):
        bs, steps = inputs.shape[:2]

        # Every step in the sequence takes the latent vector as input so we replicate it here
        expanded_z = z.unsqueeze(1).repeat(1, inputs.shape[1], 1)
        inputs = torch.cat([inputs, expanded_z], 2)

        if hidden_and_cell is None:
            # Initialize from latent vector
            hidden_and_cell = self.hidden_cell_predictor(z)
            hidden = hidden_and_cell[:, :self.hidden_size]
            hidden = hidden.unsqueeze(0).contiguous()
            cell = hidden_and_cell[:, self.hidden_size:]
            cell = cell.unsqueeze(0).contiguous()
            hidden_and_cell = (hidden, cell)

        outputs, hidden_and_cell = self.lstm(inputs, hidden_and_cell)

        # if self.training:
        # At train time we want parameters for each time step
        outputs = outputs.contiguous().reshape(bs*steps, self.hidden_size)
        params = self.parameters_predictor(outputs)
        pi, mu1, mu2, sigma1, sigma2, corr, pen, pen_logits = self.get_mixture_params(params)

        return pi, mu1, mu2, sigma1, sigma2, corr, pen, pen_logits, hidden_and_cell
