# Minimal L2-Consistent Data-Transmission

The code accompaning the paper.

**Authors:** [Antoine Aspeel](https://aaspeel.github.io/), Laurent Bako and [Necmiye Ozay](https://web.eecs.umich.edu/~necmiye/)

The code reproduces the results in the section "Numerical Experiments" of the paper. 

## Setup
From the base directory of this repository, install dependencies with:
~~~~
pip install -r requirements.txt
~~~~

## Run
The code runs in two steps: first it does the computations and save the results; and then it makes the plots.

To compute the results, run
~~~~
python3 main_simulation.py
~~~~
This will save the results in `simulation_results/variables.pkl`.

Then, to make the plots, run
~~~~
python3 main_plot.py
~~~~
this will plot the results and save them in the folder `simulation_results/`.

## Appendix
The script `main_simulation.py` relies on the following additional scripts:
1. `load_system.py` is used to load an instance of the problem.
2. `SLSFinite.py` defines a class `SLSFinite` storing the optimization variables and parameters of the optimization problems. Methods of `SLSFinite` compute system level synthesis constraint.
3. `optimize_SLS.py` solve the optimization problem with reweighting nuclear norm heuristic.
4. `utils.py` contains the function to compute an approximate causal factorization.
5. `minimax.py` contains an implementation of the method proposed in _Balaghi, M. H., Antunes, D. J., & Heemels, W. M. (2019, December). An L 2-Consistent Data Transmission Sequence for Linear Systems. In 2019 IEEE 58th Conference on Decision and Control (CDC) (pp. 2622-2627). IEEE._

The script `main_plot.py` relies on `plots.py` which contains functions used for the plots.


