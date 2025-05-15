# DC-gra2seq：Equipping sketch patches with context-aware positional encoding for graphic sketch representation

[SP-gra2seq](https://ojs.aaai.org/index.php/AAAI/article/view/26314) ([source code](https://github.com/CMACH508/SP-gra2seq)) learns graphic sketch representations by constructing graph edges according to synonymous proximity between sketch patches, but SP-gra2seq never considers the sequential information among drawing strokes from sketch drawing orders. Here we propose a variant-drawing-protected method, namely DC-gra2seq, to improve SP-gra2seq by benefiting graphic sketch representation with sketch drawing orders. DC-gra2seq equips sketch patches with context-aware positional encoding (PE) to make better use of drawing orders for sketch learning. Sinusoidal absolute PEs and learnable relative PEs are employed to embed the sequential positions in drawing orders and encode the unseen contextual relationships between patches, respectively. Note that both types of PEs never attend the construction of graph edges, but are injected into graph nodes to cooperate with the visual patterns captured from patches. After linking nodes by semantic proximity, during message aggregation via graph convolutional networks, each node receives both semantic features from patches and contextual information from PEs from its neighbors, which equips local patch patterns with global contextual information, further obtaining drawing-order-enhanced sketch representations.

This repo contains official source codes and pre-trained models for `DC-gra2seq`, and its correspondig article can be found at [link](https://www.sciencedirect.com/science/article/pii/S1077314225001080).

The source code will be released soon. 


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

