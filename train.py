import argparse
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision

import six
import math
import model
from seq2img import make_graph_
import random

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, categories, graph_number=20, graph_picture_size=256, mask_prob=0.1, mode='train'):
        super(Dataset, self).__init__()

        self.mode = mode
        self.limit = 1000
        self.graph_number = graph_number
        self.graph_picture_size = graph_picture_size
        self.mask_prob = mask_prob

        if self.mode not in ["train", "test", "valid"]:
            return ValueError("Only allowed data mode are 'train' and 'test', 'valid'.")

        count = 0
        for ctg in categories:
            # load sequence data
            seq_path = os.path.join(data_dir, ctg + '.npz')
            if six.PY3:
                seq_data = np.load(seq_path, encoding='latin1', allow_pickle=True)
            else:
                seq_data = np.load(seq_path, allow_pickle=True)

            if count == 0:
                train_seqs = seq_data['train']
                valid_seqs = seq_data['valid']
                test_seqs = seq_data['test']
            else:
                train_seqs = np.concatenate((train_seqs, seq_data['train']))
                valid_seqs = np.concatenate((valid_seqs, seq_data['valid']))
                test_seqs = np.concatenate((test_seqs, seq_data['test']))
            count += 1

        self.max_seq_len = self.get_max_len(np.concatenate((train_seqs, valid_seqs, test_seqs)))

        if self.mode == 'train':
            self.strokes = self.preprocess_data(train_seqs)
        elif self.mode == 'valid':
            self.strokes = self.preprocess_data(valid_seqs)
        else:
            self.strokes = self.preprocess_data(test_seqs)

    def preprocess_data(self, seqs):
        # pre-process
        strokes = []
        scale_factor = self.calculate_normalizing_scale_factor(seqs)

        count_data = 0  # the number of drawing with length less than N_max
        for i in range(len(seqs)):
            seq = np.copy(seqs[i])
            if len(seq) <= self.max_seq_len:    # keep data with length less than N_max
                count_data += 1
                # removes large gaps from the data
                seq = np.minimum(seq, self.limit)     # prevent large values
                seq = np.maximum(seq, -self.limit)    # prevent small values
                seq = np.array(seq, dtype=float)  # change data type
                seq[:, 0:2] /= scale_factor       # scale the first two dims of data
                strokes.append(seq)
        return strokes

    def get_max_len(self, strokes):
        max_len = 0
        for stroke in strokes:
            ml = len(stroke)
            if ml > max_len:
                max_len = ml
        return max_len

    def calculate_normalizing_scale_factor(self, sketches):
        data = []
        for sketch in sketches:
            for stroke in sketch:
                data.append(stroke)
        return np.std(np.array(data))

    def random_scale_seq(self, data):
        random_scale_factor = 0.1
        x_scale_factor = (np.random.random() - 0.5) * 2 * random_scale_factor + 1.0
        y_scale_factor = (np.random.random() - 0.5) * 2 * random_scale_factor + 1.0
        result = np.copy(data)
        result[:, 0] *= x_scale_factor
        result[:, 1] *= y_scale_factor
        return result

    def seq_3d_to_5d(self, stroke, max_len=250):
        result = np.zeros((max_len + 1, 5), dtype=float)
        l = len(stroke)
        assert l <= max_len
        result[0:l, 0:2] = stroke[:, 0:2]
        result[0:l, 3] = stroke[:, 2]
        result[0:l, 2] = 1 - result[0:l, 3]
        result[l:, 4] = 1

        # put in the first token, as described in sketch-rnn methodology
        start_stroke_token = [0, 0, 1, 0, 0]  # S_0 in sketch-rnn paper
        result[1:, :] = result[:-1, :]
        result[0, :] = 0
        result[0, 2] = start_stroke_token[2]  # setting S_0 from paper.
        result[0, 3] = start_stroke_token[3]
        result[0, 4] = start_stroke_token[4]
        return result

    def __len__(self):
        return len(self.strokes)

    def __getitem__(self, idx):
        strokes_3d = self.strokes[idx]
        strokes_5d = self.seq_3d_to_5d(strokes_3d, self.max_seq_len)

        # absolute position embedding
        pre_pe = np.zeros([self.graph_number - 1, 512])  # 512 dimensions
        position = np.expand_dims(np.array(np.arange(self.graph_number - 1)), axis=1)
        div_term = np.exp(np.array(np.arange(0, 512, 2)) * (- math.log(10000.0) / 512))
        pre_pe[:, 0::2] = np.sin(position * div_term)
        pre_pe[:, 1::2] = np.cos(position * div_term)

        # relative distances
        row = np.array(np.arange(self.graph_number - 1))
        column = np.array(np.arange(self.graph_number - 1))
        relative_distance = np.zeros([self.graph_number - 1, self.graph_number - 1])
        for i in range(self.graph_number - 1):
            for j in range(self.graph_number - 1):
                relative_distance[i, j] = np.abs(row[i] - column[j])
        # relative_distance = np.clip(relative_distance, -np.inf, 5)  # max relative distance is 5

        # transform sketch sequences to images
        seed = np.load('/data/datasets/quickdraw/random_seed.npy', allow_pickle=True)   # a group of pre-determined random seeds for evaluation
        seed_id = 0
        _, graph, adj, _graph_len, mask_id, seed_id = make_graph_(strokes_3d, seed, seed_id, graph_num=self.graph_number,
                                                                  graph_picture_size=self.graph_picture_size,
                                                                  mask_prob=self.mask_prob, train=self.mode)
        if _graph_len == (self.graph_number - 1):
            adj_mask = np.ones([self.graph_number - 1, self.graph_number - 1])
        else:
            adj_mask = np.concatenate([np.concatenate([np.ones([_graph_len + 1, _graph_len + 1]),
                                                       np.zeros([self.graph_number - 2 - _graph_len, _graph_len + 1])], axis=0),
                                       np.zeros([self.graph_number - 1, self.graph_number - 2 - _graph_len])], axis=1)
        for id in mask_id:
            adj_mask[id, :] = 0
            adj_mask[:, id] = 0

        return (strokes_5d.astype(np.float32), graph.astype(np.float32), adj.astype(np.float32),
                adj_mask.astype(np.float32), pre_pe.astype(np.float32), relative_distance.astype(np.int32))

class GaussianMixtureReconstructionLoss(torch.nn.Module):
    def __init__(self, bs, eps=1e-6):
        super(GaussianMixtureReconstructionLoss, self).__init__()
        self.bs = bs
        self.eps = eps

    def get_density(self, x1, x2, mu1, mu2, s1, s2, rho):
        norm1 = x1 - mu1
        norm2 = x2 - mu2
        s1s2 = s1 * s2
        z = (norm1 / s1).square() + (norm2 / s2).square() - 2. * rho * norm1 * norm2 / s1s2
        neg_rho = 1. - rho.square()
        result = (-z / (2. * neg_rho)).exp()
        denom = 2 * np.pi * s1s2 * neg_rho.sqrt()
        result = result / denom
        return result

    def forward(self, pi, mu1, mu2, s1, s2, corr, pen_logits, x1_data, x2_data, pen_data, mode):
        result0 = self.get_density(x1_data, x2_data, mu1, mu2, s1, s2, corr)
        result1 = (result0 * pi).sum(dim=1)
        result1 = -(result1 + self.eps).log()  # Avoid log(0)

        masks = 1.0 - pen_data[:, 2]
        masks = masks.contiguous().reshape(-1)
        result1 = (result1 * masks).sum()
        result2 = torch.nn.functional.cross_entropy(pen_logits, pen_data, reduction='none')
        if mode == 'train':
            result2 = result2.sum()
        else:
            result2 = (result2 * masks).sum()

        return (result1 + result2) / self.bs

class Model:
    def __init__(self):
        self.encoder: nn.Module = model.Encoder(args=args).cuda()
        self.decoder: nn.Module = model.Decoder(args=args).cuda()

        self.lil_loss = GaussianMixtureReconstructionLoss(args.bs)

        self.enc_optimizer = torch.optim.Adam(self.encoder.parameters(), args.lr)
        self.dec_optimizer = torch.optim.Adam(self.decoder.parameters(), args.lr)

    def load(self, enc_name, dec_name):
        saved_enc = torch.load(enc_name)
        saved_dec = torch.load(dec_name)
        self.encoder.load_state_dict(saved_enc)
        self.decoder.load_state_dict(saved_dec)

    def save(self, epoch):
        torch.save(self.encoder.state_dict(), f'./sketch_model/enc_epoch_{epoch}.pth')
        torch.save(self.decoder.state_dict(), f'./sketch_model/dec_epoch_{epoch}.pth')

    def lr_decay(self, optimizer):
        """Decay learning rate by a factor of lr_decay"""
        for param_group in optimizer.param_groups:
            if param_group['lr'] > 1e-6:
                param_group['lr'] *= 0.9
        return optimizer

    def train(self, epoch, dataloader):
        self.encoder.train()
        self.decoder.train()

        step = 0
        start = time.time()
        for batch in dataloader:
            seqs, graphs, adjs, input_masks, abs_pe, rel_dis = batch
            seqs = seqs.cuda().float()
            graphs = graphs.cuda().float()
            # adjs = adjs.cuda().float()
            input_masks = input_masks.cuda().float()
            abs_pe = abs_pe.cuda().float()
            rel_dis = rel_dis.cuda().long()

            z, mu, logvar = self.encoder(graphs, input_masks, rel_dis, abs_pe)
            pi, mu1, mu2, sigma1, sigma2, corr, pen, pen_logits, _ = self.decoder(seqs[:, :-1], z)

            self.enc_optimizer.zero_grad()
            self.dec_optimizer.zero_grad()

            x1_data, x2_data, pen_data = torch.split(seqs[:, 1:].contiguous().reshape(-1, 5), [1, 1, 3], dim=-1)
            loss = self.lil_loss(pi, mu1, mu2, sigma1, sigma2, corr, pen_logits, x1_data, x2_data, pen_data, "train")
            loss.backward()

            grad_clip = 1.
            torch.nn.utils.clip_grad_norm(self.encoder.parameters(), grad_clip)
            torch.nn.utils.clip_grad_norm(self.decoder.parameters(), grad_clip)

            self.enc_optimizer.step()
            self.dec_optimizer.step()

            if (step % 20) == 0:
                end = time.time()
                time_taken = end - start
                start = time.time()

                print("Epoch: %d, Step: %d, Lil: %.2f, Time: %.2f" % (epoch, step, loss, time_taken))
            step += 1

    def eval(self, dataloader):
        self.encoder.eval()
        self.decoder.eval()

        loss_sum = 0
        count = 0
        for batch in dataloader:
            seqs, graphs, adjs, input_masks, abs_pe, rel_dis = batch

            seqs = seqs.cuda().float()
            graphs = graphs.cuda().float()
            # adjs = adjs.cuda().float()
            input_masks = input_masks.cuda().float()
            abs_pe = abs_pe.cuda().float()
            rel_dis = rel_dis.cuda().long()

            with torch.no_grad():
                z, mu, logvar = self.encoder(graphs, input_masks, rel_dis, abs_pe)
                pi, mu1, mu2, sigma1, sigma2, corr, pen, pen_logits, _ = self.decoder(seqs[:, :-1], z)

                x1_data, x2_data, pen_data = torch.split(seqs[:, 1:].contiguous().reshape(-1, 5), [1, 1, 3], dim=-1)
                loss = self.lil_loss(pi, mu1, mu2, sigma1, sigma2, corr, pen_logits, x1_data, x2_data, pen_data, "valid")
                loss_sum += loss
                count += 1

        return loss_sum / count

def train_model(args):
    # Initialize datasets
    trainset = Dataset(args.data_dir, categories=args.categories, graph_number=args.graph_number, graph_picture_size=256, mask_prob=args.mask_prob, mode="train")
    validset = Dataset(args.data_dir, categories=args.categories, graph_number=args.graph_number, graph_picture_size=256, mask_prob=0.0, mode="valid")
    train_dataloader = DataLoader(trainset, batch_size=args.bs, num_workers=8, shuffle=True)
    valid_dataloader = DataLoader(validset, batch_size=args.bs, num_workers=4, shuffle=False)

    if not os.path.exists('./sketch_model/'):
        os.makedirs('./sketch_model/')

    model = Model()

    best_lil = 1e5
    epoch = 0

    if args.epoch_load != 0:
        model.load(f'./sketch_model/enc_epoch_{args.epoch_load}.pth',
                   f'./sketch_model/dec_epoch_{args.epoch_load}.pth',)
        best_lil = np.load(f'./sketch_model/best_lil.npy')
        epoch = args.epoch_load
        for i in range(args.epoch_load):  # initialize the starting lr
            model.enc_optimizer = model.lr_decay(model.enc_optimizer)
            model.dec_optimizer = model.lr_decay(model.dec_optimizer)

    while epoch <= args.num_epochs:
        model.train(epoch, train_dataloader)
        current_lil = model.eval(valid_dataloader)
        print("Epoch: %d, Best-Lil: %.2f, Current-Lil: %.2f" % (epoch, best_lil, current_lil))

        if current_lil.cpu().numpy() < best_lil:
            epoch += 1
            best_lil = current_lil.cpu().numpy()
            np.save(f'./sketch_model/best_lil.npy', current_lil.cpu().numpy())
            print("Model %d saved." % epoch)
            model.save(epoch)
        else:
            print("Loading model %d." % (epoch - 1))
            model.load(f'./sketch_model/enc_epoch_%d.pth' % (epoch - 1),
                       f'./sketch_model/dec_epoch_%d.pth' % (epoch - 1))

class HParams:
    def __init__(self):
        self.data_dir = "/data/datasets/quickdraw/"
        self.categories = ["bee", "bus", "flower", "giraffe", "pig"]
        self.lr = 0.001         # initialized learning rate
        self.bs = 256           # mini-batch size
        self.num_epochs = 50    # max number pf training epoch
        self.num_gaussians = 20 # number of GMM components in decoder
        self.graph_number = 21  # each sketch is represented by 20 graph nodes and 1 full sketch image
        self.decoder_dim = 1024 # hidden state size of LSTM decoder
        self.zdim = 128         # latent code dimension
        self.mask_prob = 0.1    # masking probability
        self.epoch_load = 0     # load pre-trained network parameters

if __name__ == "__main__":
    args = HParams()
    train_model(args)
