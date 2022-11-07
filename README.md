# HyperSCI-KDD22:  Learning Causal Effects on Hypergraphs

Code for the KDD 2022 paper [*Learning Causal Effects on Hypergraphs*.](https://arxiv.org/pdf/2207.04049.pdf)

## Environment
```
Python 3.6
Pytorch 1.2.0
Scipy 1.3.1
Numpy 1.17.2
```

## Dataset
Datasets can be found in [link](https://virginia.box.com/s/zo47hdsavd0vvsnnxmqiitec3dsmbmbc).

## Run Experiment
### HyperSCI
```
python HyperSCI.py --path '../../data/Simulation/GR/GoodReads.mat'
```

The data preprocessing and simulation is in:
### Data Preprocessing
```
python data_preprocessing.py
```
### Data Simulation
```
python data_simulation.py
```

### Refenrences
Jing Ma, Mengting Wan, Longqi Yang, Jundong Li, Brent Hecht, Jaime Teevan, “Learning Causal Effects on Hypergraphs”, ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD), 2022. 

