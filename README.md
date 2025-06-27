# BaMFA_cemcon_Eng19
Bayesian Material Flow Analysis (BaMFA) codebase used for and which supplements the manuscript, 'Bayesian Material Flow Analysis and Life Cycle Assessment of the Cement and Concrete Life Cycle in England' by Mason et al.

## Bayesian material flow analysis

### Overview

The study uses BaMFA methodology for reconciliation of the available data to quantify uncertainty and improve the reliability of the quantitative results.

The BaMFA model was implemented in Python (v3.9.16), and `PyMC3` (v3.11.2) was used to conduct Bayesian inference via the No-U-Turn-Sampler algorithm. Refer to the manuscript for further information, results, and directions to input and output data.

### System requirements

This BaMFA code requires only a standard computer with enough RAM to support the in-memory operations. The analysis took 30 minutes to run using a standard Lenovo ThinkPad laptop.

### Documentation and installation guide

- See the BaMFA study by Wang et. al. <a href="https://doi.org/10.1111/jiec.13550" target="_blank" style=" text-decoration: none !important; color:red !important;">here &#10140;</a>
- See Python documentation <a href="https://docs.python.org/3/" target="_blank" style=" text-decoration: none !important; color:red !important;">here &#10140;</a>
- See `PyMC3` documentation <a href="https://pymc3-fork.readthedocs.io/en/latest/#" target="_blank" style=" text-decoration: none !important; color:red !important;">here &#10140;</a>

#### Python dependencies

```
pymc3
arviz
pandas
numpy
math
random
matplotlib
os
```

### Instructions

The BaMFA codebase consists of four Python and one Jupyter Notebook source files. The Python source files define the necessary functions to prepare the input data for analysis (‘preprocessingagg.py’), to construct prior distributions for analysis (‘prior.py’), to conduct material flow analysis using Bayes’ theorem (‘model.py’), and lastly to construct posterior predictive distributions and plots for model and data checking (‘posteriorpredictive.py’). An additional Python source file (‘outputforsankey.py’) is used to obtain the BaMFA model results for flows to make it easier to plot them in Sankey diagram format. The Jupyter Notebook file (‘runmodel.ipynb’) is used to run the complete BaMFA model and obtain outputs combining all the other source files.

The BaMFA model produces posterior distributions of all child stock changes and flows by combining the prior distribution and data (including mass balance for all processes) via Bayes’ Theorem, which includes quantifying and propagating uncertainties. The posterior distribution provides estimates for each stock change or flow of interest via the posterior mean, as well as uncertainty quantification through 95% credible intervals.
