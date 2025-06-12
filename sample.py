import argparse
import os, glob, re
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms
from PIL import Image
import model
from train import Dataset, HParams
from seq2img import make_graph_, seq_5d_to_3d
import cv2

EPOCH_LOAD = 20     # load pre-trained network parameters
NUM_PER_CATEGORY = 2500

def sort_paths(paths):
    idxs = []
    for path in paths:
        idxs.append(int(re.findall(r'\d+', path)[-1]))

    for i in range(len(idxs)):
        for j in range(i+1, len(idxs)):
            if idxs[i] > idxs[j]:
                tmp = idxs[i]
                idxs[i] = idxs[j]
                idxs[j] = tmp

                tmp = paths[i]
                paths[i] = paths[j]
                paths[j] = tmp
    return paths

class Eval_Model:
    def __init__(self):
        self.encoder: nn.Module = model.Encoder(args=args).cuda()
        self.decoder: nn.Module = model.Decoder(args=args).cuda()

    def load(self, enc_name, dec_name):
        saved_enc = torch.load(enc_name)
        saved_dec = torch.load(dec_name)
        self.encoder.load_state_dict(saved_enc)
        self.decoder.load_state_dict(saved_dec)

    def adjust_pdf(self, pi_pdf, temp):
        pi_pdf = np.log(pi_pdf + 1e-10) / temp
        pi_pdf -= np.max(pi_pdf)
        pi_pdf = np.exp(pi_pdf)
        pi_pdf /= np.sum(pi_pdf)
        return pi_pdf

    def get_pi_idx(self, x, pdf, temp=1.0, greedy=False):
        if greedy:
            return np.argmax(pdf)
        pdf = self.adjust_pdf(np.copy(pdf), temp)
        accumulate = 0
        for i in range(0, pdf.size):
            accumulate += pdf[i]
            if accumulate >= x:
                return i
        return -1

    def sample_gaussian_2d(self, mu1, mu2, s1, s2, rho, temp=1.0, greedy=False):
        if greedy:
            return mu1, mu2
        mean = [mu1, mu2]
        s1 *= temp * temp
        s2 *= temp * temp
        cov = [[s1 * s1, rho * s1 * s2], [rho * s1 * s2, s2 * s2]]
        x = np.random.multivariate_normal(mean, cov, 1)
        return x[0][0], x[0][1]

    def sample(self, z, sequences, temperature=0.24):
        start_of_seq = sequences[:, :1]

        max_steps = sequences.shape[1] - 1  # last step is an end-of-seq token

        output_sequences = torch.zeros_like(sequences)
        output_sequences[:, 0] = start_of_seq.squeeze(1)

        current_input = start_of_seq
        hidden_and_cell = None
        for step in range(max_steps):
            pi, mu1, mu2, sigma1, sigma2, corr, pen, pen_logits, hidden_and_cell = self.decoder(current_input, z, hidden_and_cell=hidden_and_cell)

            idx = self.get_pi_idx(random.random(), pi[0].cpu().numpy(), temperature)
            next_x1, next_x2 = self.sample_gaussian_2d(mu1[0][idx].cpu().numpy(), mu2[0][idx].cpu().numpy(),
                                                       sigma1[0][idx].cpu().numpy(), sigma2[0][idx].cpu().numpy(),
                                                       corr[0][idx].cpu().numpy(), np.sqrt(temperature))
            # generate stroke pen status
            idx_eos = self.get_pi_idx(random.random(), pen[0].cpu().numpy(), temperature)

            eos = np.zeros(3)
            eos[idx_eos] = 1

            output_sequences[0, step + 1, :] = torch.tensor([next_x1, next_x2, eos[0], eos[1], eos[2]]).cuda().float()

            current_input = np.array([next_x1, next_x2, eos[0], eos[1], eos[2]], dtype=np.float32)
            current_input = torch.tensor(current_input.reshape([1, 1, 5])).cuda().float()

        return output_sequences

    def eval(self, dataloader, category, args, seed_id):
        self.encoder.eval()
        self.decoder.eval()

        count = 0
        seed = np.load('/data/datasets/quickdraw/random_seed.npy', allow_pickle=True)

        for batch in dataloader:
            ori_seqs, graphs, adjs, input_masks, abs_pe, rel_dis = batch

            seqs = ori_seqs.cuda().float()
            graphs = graphs.cuda().float()
            input_masks = input_masks.cuda().float()
            abs_pe = abs_pe.cuda().float()
            rel_dis = rel_dis.cuda().long()

            # learn and store codes of the original unmasked sketches
            with torch.no_grad():
                gt_z, mu, logvar = self.encoder(graphs, input_masks, rel_dis, abs_pe)

                filepath = './sample/gt_%d_%d.npy' % (category, count)
                np.save(filepath, gt_z.cpu().numpy()[0])

            # apply the prepared masking seeds
            _, graph, adj, _graph_len, mask_id, seed_id = make_graph_(seq_5d_to_3d(ori_seqs[0]), seed, seed_id,  # bs = 1
                                                                      graph_num=args.graph_number,
                                                                      graph_picture_size=256,
                                                                      mask_prob=args.mask_prob, train='test')
            if _graph_len == (args.graph_number - 1):
                input_mask = np.ones([args.graph_number - 1, args.graph_number - 1])
            else:
                input_mask = np.concatenate([np.concatenate([np.ones([_graph_len + 1, _graph_len + 1]),
                                                             np.zeros([args.graph_number - 2 - _graph_len, _graph_len + 1])], axis=0),
                                             np.zeros([args.graph_number - 1, args.graph_number - 2 - _graph_len])], axis=1)
            for id in mask_id:
                input_mask[id, :] = 0
                input_mask[:, id] = 0

            graphs = torch.from_numpy(np.expand_dims(graph, axis=0))
            input_masks = torch.from_numpy(np.expand_dims(input_mask, axis=0))

            graphs = graphs.cuda().float()
            input_masks = input_masks.cuda().float()

            with torch.no_grad():
                z, mu, logvar = self.encoder(graphs, input_masks, rel_dis, abs_pe)
                out_seqs = self.sample(z, seqs)

                filepath = './sample/seq_%d_%d.npy' % (category, count)
                np.save(filepath, out_seqs.cpu().numpy()[0])

                temp = seq_5d_to_3d(out_seqs[0].cpu())
                if len(temp) < 2:
                    continue
                _, graph, _, _graph_len, mask_id, _ = make_graph_(temp, seed, seed_id=0, graph_num=args.graph_number,
                                                                  graph_picture_size=256, mask_prob=0.0, train=False)

                if _graph_len == (args.graph_number - 1):
                    input_mask = np.ones([args.graph_number - 1, args.graph_number - 1])
                else:
                    input_mask = np.concatenate([np.concatenate([np.ones([_graph_len + 1, _graph_len + 1]),
                                                                 np.zeros([args.graph_number - 2 - _graph_len, _graph_len + 1])], axis=0),
                                                 np.zeros([args.graph_number - 1, args.graph_number - 2 - _graph_len])], axis=1)
                for id in mask_id:
                    input_mask[id, :] = 0
                    input_mask[:, id] = 0

                path = os.path.join("./sample/%d_%d.png" % (category, count))
                g0 = 255. - (graph[0] + 1) / 2. * 255.
                cv2.imwrite(path, np.tile(g0, [1, 1, 3]))

                # learn and store codes of the generated sketches
                fake_z, _, _ = self.encoder(torch.from_numpy(np.expand_dims(graph, axis=0)).cuda().float(),
                                            torch.from_numpy(np.expand_dims(input_mask, axis=0)).cuda().float(), rel_dis, abs_pe)
                filepath = './sample/fake_%d_%d.npy' % (category, count)
                np.save(filepath, fake_z.cpu().numpy()[0])

            count += 1
            if count >= NUM_PER_CATEGORY:
                return seed_id

def sample(args):
    bs = 1
    seed_id = 0

    args.epoch_load = EPOCH_LOAD

    model = Eval_Model()
    model.load(f'./sketch_model/enc_epoch_{args.epoch_load}.pth',
               f'./sketch_model/dec_epoch_{args.epoch_load}.pth')

    if not os.path.exists('./sample/'):
        os.makedirs('./sample/')

    for category in range(len(args.categories)):
        print(args.categories[category])
        testset = Dataset(args.data_dir, categories=[args.categories[category]], graph_number=args.graph_number,
                          graph_picture_size=256, mask_prob=0.0, mode="test")
        test_dataloader = DataLoader(testset, batch_size=bs, num_workers=1, shuffle=False)

        seed_id = model.eval(test_dataloader, category, args, seed_id)


if __name__ == "__main__":
    args = HParams()
    args.mask_prob = 0.1
    sample(args)
