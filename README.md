# DC-gra2seq：Equipping sketch patches with context-aware positional encoding for graphic sketch representation

[SP-gra2seq](https://ojs.aaai.org/index.php/AAAI/article/view/26314) ([source code](https://github.com/CMACH508/SP-gra2seq)) learns graphic sketch representations by constructing graph edges according to synonymous proximity between sketch patches, but SP-gra2seq never considers the sequential information among drawing strokes from sketch drawing orders. Here we propose a variant-drawing-protected method, namely DC-gra2seq, to improve SP-gra2seq by benefiting graphic sketch representation with sketch drawing orders. 

<img src="https://github.com/sczang/blob/main/assets/PE_for_node.png" width="400" alt="PE_for_node"/>

DC-gra2seq equips sketch patches with context-aware positional encoding (PE) to make better use of drawing orders for sketch learning. Sinusoidal absolute PEs and learnable relative PEs are employed to embed the sequential positions in drawing orders and encode the unseen contextual relationships between patches, respectively. Note that both types of PEs never attend the construction of graph edges, but are injected into graph nodes to cooperate with the visual patterns captured from patches. After linking nodes by semantic proximity, during message aggregation via graph convolutional networks, each node receives both semantic features from patches and contextual information from PEs from its neighbors, which equips local patch patterns with global contextual information, further obtaining drawing-order-enhanced sketch representations.

<img src="https://github.com/sczang/blob/main/assets/overview.png" width="400" alt="overview"/>

This repo contains official source codes and [pre-trained models]() for `DC-gra2seq`, and its corresponding article can be found at [link](https://www.sciencedirect.com/science/article/pii/S1077314225001080).

# Training a DC-gra2seq

## Dataset

We use [QuickDraw dataset](https://quickdraw.withgoogle.com/data) to train our DC-gra2seq, and the function `make_graph_` in `seq2img.py` is utilized to translate original sequence-formed sketches into graphic structures.

## Required environments

1. Python 3.8
2. PyTorch 2.4.1

## Training

The training settings can be found at the class `HParams` in `train.py`.

```
data_dir = "/data/datasets/quickdraw/"      # dataset directory
categories = ['bee', 'bus'],         # Sketch categories for training
lr = 0.001                      # learning rate
bs = 256                        # mini-batch size
num_epochs = 50                # max number of training epoch
num_gaussians = 20            # number of GMM components in LSTM decoder
graph_number = 21            # each sketch is represented by 20 graph nodes and 1 full sketch image
decoder_dim = 1024            # hidden state size of LSTM decoder
zdim = 128                  # latent code size
mask_prob = 0.1            # masking probability. Recommend leaving it at 0.1 when model training
epoch_load = 0            # load a pre-trained model
```


You can simply run
```
python train.py
```
to starting network training. 

## Generating
```
python sample.py
```

With a pre-trained model, you can generate sketches based on the input (corrupted) sketches. In `sample.py`, `EPOCH_LOAD` at #Line 17# and `NUM_PER_CATEGORY` at #Line 18# denote which pre-trained model is utilized for sketch generation and how many sketches per category are going to generate, respectively. And you are able to adjust masking probability for sketch healing by editing the value of `mask_prob` in #Line 211#.


## Evaluation

The metrics **Rec** and **Ret** are used to testify whether a method learns accurate and robust sketch representations. For calculating **Rec**, you need to train a [Sketch_a_net](https://arxiv.org/pdf/1501.07873.pdf) for each dataset as the classifier. And for **Ret**, you can run `calculate_Ret.py` to obtain it with the generated sketches (2500 sketches per category). The following figure presents the detail calculations of both metrics for controllable sketch synthesis and sketch healing, respectively.
```
python calculate_Ret.py

* Please make sure both the metrics are computed with the entire test set (i.e., num_per_category=2500 in `sample.py`).

* We also provide the random seeds in `random_seed.npy` (stored in the .zip file in [link](https://jbox.sjtu.edu.cn/l/i193TY)) for creating the random masks for sketch healing. These seeds are the specific ones utilized in the article for the sketch healing performance evaluation. You can use them to make a fair comparison with the benchmarks in the article.

# Citation
If you find this project useful for academic purposes, please cite it as:
```
@Article{DC-gra2seq,
  Title                    = {Equipping sketch patches with context-aware positional encoding for graphic sketch representation},
  Author                   = {Sicong Zang and Zhijun Fang},
  Journal                  = {Computer Vision and Image Understanding},
  Volume                   = {258},
  Pages                    = {104385},
  Year                     = {2025},
  Doi                      = {https://doi.org/10.1016/j.cviu.2025.104385}
}
```

