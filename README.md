### Main analysis code for Zhang, Meschke, & Gallant. **A map of the cortical functional network mediating naturalistic navigation**
This repository contains code that fits voxelwise encoding models for the active navigation data, 
and applies boostrapped model connectivity to identify the network of cortical navigation regions.

##### External dependencies
- [joblib](https://joblib.readthedocs.io/en/stable/)
- [pytables](https://www.pytables.org)
- [h5py](https://docs.h5py.org/en/stable/index.html)
- [Numpy](https://numpy.org)
- [Scipy](https://scipy.org)
- [Scikit-learn](https://scikit-learn.org/stable/index.html)
- [Himalaya](https://github.com/gallantlab/himalaya)
- [pyTorch](https://pytorch.org)*
- [Fast Pairwise](https://github.com/gallantlab/Fast-Pairwise)*
- an Nvidia GPU*
- [Pycortex](https://github.com/gallantlab/pycortex) †
- [Plotly](https://plotly.com/python/) and [Kaleido](https://github.com/plotly/Kaleido) †
- [UMAP](https://umap-learn.readthedocs.io) †

*Not technically required, but accelerates things quite a bit.

†Not needed for analysis, only for visualization & plotting

##### Tested Environments:
- Ubuntu 18.04 with python 3.6, Ubuntu 22.04 with python 3.10

##### Installation
- Install external dependencies
- Clone this repo; code in this repo is intended to run out of this directory box without any further installations
- Set `BASE_DIRECTORY` in Config.py to point to a directory for data storage
- Clone data into the data directory

##### Example scripts
The numbered scripts are found in the "Driving Network" folder and are run in order:
1. Fit the voxelwise encoding models for one participant.
2. Bootstrap the model connectivity clustering for that participant.
3. Compute the navigation preference index for that participant.
4. Bootstrap the model connectivity clustering across the group.
5. Compute the group navigation preference index.
6. Compute the tuning bias profiles for the navigation ROIs.
7. Project the group encoding model weights with UMAP and compute the concreteness gradient.

Scripts #1-3 run on a single participant, set by the `participant` variable at the top of each script. 
Scripts #4-7 operate on the whole group and require that scripts #1-3 have been run for every participant first.

##### Runtime
The VM process takes approx. ~3 days per participant on a RTX A6000. 
The bootstrapped clustering takes approx. 1 week on two Xeon Gold 5220Rs sharing 1 TB RAM.