# Indoor Climbing Hold and Route Segmentation
This repository contains the source code for the [Indoor Climbing Hold and Route Segmentation](https://github.com/xiaoxiae/Indoor-Climbing-Hold-and-Route-Segmentation/tree/main/paper/main.pdf) project, created as a final assignment for the [Computer Vision: 3D Reconstruction](https://hci.iwr.uni-heidelberg.de/content/computer-vision-3d-reconstruction-ws-2223) course (University of Heidelberg).

## Obtaining data
The data used, including the best-performing model weights, can be obtained from Kaggle: https://www.kaggle.com/datasets/tomasslama/indoor-climbing-gym-hold-segmentation

## Running the code
A short demo of the implemented functionality can be found in `demo.ipynb`.

Alternatively, you can run the ML hold and route detection straight on Kaggle without downloading anything:

- Hold Segmentation: https://www.kaggle.com/code/tomasslama/hold-segmentation/
- Route Segmentation: https://www.kaggle.com/code/tomasslama/route-segmentation/

## Folder structure

```
.
├── data/
│   ├── bh/               # Bolderhaus camera VIA annotations
│   ├── bh-phone/         # Bolderhaus phone VIA annotations
│   ├── sm/               # Smíchoff camera VIA annotations
│   ├── *.py              # dataset utility scripts
│   └── statistics.ipynb  # dataset statistics
├── ml/                   # machine learning approach
├── std/                  # standard approach
├── paper/                # paper source code
└── demo.ipynb            # code demo
```
