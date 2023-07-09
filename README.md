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
Demo datasets with simulation can be found in [link](https://virginia.box.com/s/zo47hdsavd0vvsnnxmqiitec3dsmbmbc).

## Run Experiment
### HyperSCI
```
python HyperSCI.py --dataset 'contact' --path '../data/contact.mat'
```
With the demo ```contact.mat``` dataset and default parameter settings, the mean results ($\sqrt{\epsilon_{PEHE}}$ and $\epsilon_{ATE}$) of three runs for our method should be $12.16/9.55$. 

```
python HyperSCI.py --dataset 'GoodReads' --path '../data/GoodReads.mat'
```
With the demo ```GoodReads.mat``` dataset and default parameter settings, the mean results ($\sqrt{\epsilon_{PEHE}}$ and $\epsilon_{ATE}$) of three runs for our method should be $33.30/4.73$. 

The data preprocessing from raw data and simulation is in:
### Data Preprocessing
```
python data_preprocessing.py
```
### Data Simulation
```
python data_simulation.py
```

### References
Jing Ma, Mengting Wan, Longqi Yang, Jundong Li, Brent Hecht, Jaime Teevan, “Learning Causal Effects on Hypergraphs”, ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD), 2022. 

