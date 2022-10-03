# Understanding Interventional TreeSHAP: How and Why it works.

This code is a companion to the paper [Understanding TreeSHAP](https://arxiv.org/abs/2209.15123)
and acts as supplementary materials for the interested reader. Two C++ implementations of
Interventional TreeSHAP and Taylor-TreeSHAP are provided. Said implementations use the same
notation as the paper.
Our TreeSHAP implementation is not meant to be a replacement to the well-established
[SHAP](https://github.com/slundberg/shap) library. Is is rather intended as a tool to teach
the method or to drive new research for on the topic of explaining tree ensembles with game-theory 
indices.

---------------
## Setup

To setup the virtual environement, we suggest to use [Anaconda](https://www.anaconda.com/products/distribution). Once it is installed, run
```
conda env create --file environment.yml
```
Then, the C++ implementations of Interventional TreeSHAP and Taylor-TreeSHAP
can be complied with setuptools.
```
python setup.py build
```
If everything worked properly, you should see a directory **build/** that contains the shared library
`.so`.


---------------
## Tree Structure

TODO : discuss the tree structures and the way the sets $S_X$ and $S_Z$ are represented.

---------------
## Tutorials

TODO : present some of the basic tutorials.