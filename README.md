# ComplexEnvExploration-Modirshanechi2025

This repository contains the code and data for the results reported in the article:

A. Modirshanechi, W H. Lin, A. Xu, M.H. Herzog, and W. Gerstner, "Novelty as a drive of human exploration in complex stochastic environments", to appear in PNAS (2025); [preprint available on bioRxiv](https://www.biorxiv.org/content/10.1101/2022.07.05.498835v3.abstract).

Contact: [alireza.modirshanechi@helmholtz-munich.de](alireza.modirshanechi@helmholtz-munich.de)


## Dependencies and installation

* [Julia](https://julialang.org/) (1.10.3)

To install the necessary Julia packages, follow these steps:

1.  Navigate into the `IMRLExploration` folder.
2.	Open a Julia terminal, press `]` to enter the package management mode.
3.	In the package management mode, type `activate .`.
4.	In the package management mode, type `instantiate`.

All Julia packages and dependencies will be installed automatically within this environment. 


## Usage: Demo, Data, and Figures

### Demo

`IMRLExploration/HowTo.ipynb` presents a demo for reading and working with the data along with a general overview of computational models.

### Data 

Behavioral choices participant number `i` (with `i` between 1 and 63) are saved in `IMRLExploration/data/BehaveData_Si.mat`.
Graph reconstruction data of all participants is saved in `IMRLExploration/data/graphs.csv`.
The data structure is explained in `IMRLExploration/HowTo.ipynb` along with a demonstration of how to read and work with the data.

*If you would like to only use the data without using our code, you can find a minimal version of the raw data saved in `data/tidydata.CSV` (with the same notation as in the paper)*

### Figures

The folder `IMRLExploration/figures/` contains a series of `ipynb` notebooks for reproducing the results presented in different figures.


## Source files

* All analyses are based on the functions and structs that are written in the files in the root of `IMRLExploration/src/`
* Subfolders contain files for model-fitting (`IMRLExploration/src/01_ModelFitting`), Posterior Predictive Checks (`IMRLExploration/src/02_PPCSimulation`; also see `IMRLExploration/figures/Figure5_plus2DF.ipynb`), and model-recovery (`IMRLExploration/src/03_ModelRecovery`).
