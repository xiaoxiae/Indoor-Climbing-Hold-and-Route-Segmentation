# Indoor Climbing Hold and Route Segmentation
This repository contains the source code for the technical report [Indoor Climbing Hold and Route Segmentation](TODO), made as a final project for the [Computer Vision: 3D Reconstruction](https://hci.iwr.uni-heidelberg.de/content/computer-vision-3d-reconstruction-ws-2223) course (University of Heidelberg).

## Obtaining data
The data used, including the best-performing model weights, can be obtained from Kaggle: https://www.kaggle.com/datasets/tomasslama/indoor-climbing-gym-hold-segmentation

## Running the code
A short demo of the implemented functionality can be found in `demo.ipynb`.

Alternatively, you can run the ML hold detection straight on Kaggle without downloading anything: https://www.kaggle.com/code/tomasslama/indoor-climbing-gym-hold-segmentation-example

## Folder structure

```
.
├── data
│   ├── bh                # Bolderhaus camera
│   ├── bh-phone          # Bolderhaus phone
│   ├── sm                # Smíchoff camera
│   ├── *.py              # dataset utility scripts
│   └── statistics.ipynb  # dataset statistics
├── demo.ipynb            # code demo
├── ml                    # machine learning approach
├── paper                 # paper source code
└── std                   # standard approach
```
