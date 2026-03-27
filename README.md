# hypergraph-MI

The file `function.py` contains all dependencies to run the hierarchy of network mutual information (NMI) measures of the manuscript [arXiv:2510.27411](https://arxiv.org/abs/2510.27411).

```
    .
    ├── README.md
    ├── data
    ├── results
    ├── figs
    ├── functions.py
    └── tutorial.ipynb
```

The folder `data/` contains scripts to load, generate and save real-world hypergraphs, and compute the their $\textrm{NMI}_{\rm cross}$ matrices.

The folder `results/` contains scripts to reproduce the results of the manuscript.

The folder `figs/` contains scripts to generate the main figures of the manuscript given the results found.

The file `tutorial.ipynb` is a brief tour of the NMI encodings applied to small hypergraphs.

All code was run using Python >= 3.10.

