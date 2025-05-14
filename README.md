# Organoid Simulator

If you are using this code, please cite the following paper:

```
@article{vanaret2023detector,
  title={A detector-independent quality score for cell segmentation without ground truth in 3D live fluorescence microscopy},
  author={Vanaret, Jules and Dupuis, Victoria and Lenne, Pierre-Fran{\c{c}}ois and Richard, Fr{\'e}d{\'e}ric and Tlili, Sham and Roudot, Philippe},
  journal={IEEE Journal of Selected Topics in Quantum Electronics},
  volume={29},
  number={4: Biophotonics},
  pages={1--12},
  year={2023},
  publisher={IEEE}
}
```

## Table of contents

* [Installation](#Installation)
* [TODO](#TODO)
* [DONE](#DONE)

## Installation

Packages that should be installed manually:
* numba
* napari
* tqdm

After cloning the repository, you can install the package with pip:
```bash
# from repo root (the DOT "." is important !!!)
cd organo_simulator;pip install -e .
```