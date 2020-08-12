# Recurrent neural-linear posterior sampling

This software supplements the paper "Recurrent Neural-Linear Posterior Sampling for Non-Stationary Contextual Bandits".

The implementation focuses on clarity and flexibility rather than computational efficiency.

## Instructions

Remember to include this repository folder in your PYTHONPATH.

### Hyperparameter grid

Create config files for a specific bandit problem instance and the policies to be evaluated:

```bash
python3 rnlps/scripts/hgrid.py experiment_folder/
```

### Experiments

Run an individual experiment on a folder with a config file:
```bash
python3 rnlps/scripts/run.py experiment_folder/single_trial/

# Try one of the example configurations
python3 rnlps/scripts/run.py rnlps/examples/example_configs/sinusoidal_bernoulli/2/
```

Run multiple experiments in parallel with 10 jobs (requires tmux):

```bash
python3 rnlps/scripts/multirun.py experiment_folder/ 10
```

### Analysis

Create a csv file that summarizes the return (mean and standard deviation over independent runs) for the different policies:

```bash
python3 rnlps/scripts/create_summary.py experiment_folder/
```

Create a plot to analyse the sensitivity of neural policies across hyperparameters:

```bash
python3 rnlps/scripts/hp_sensitivity_plot.py experiment_folder/
```


Create a plot comparing the regret of the different policies:

```bash
python3 rnlps/scripts/regret_analysis.py experiment_folder/
```

## Dependencies

- tmux (for multirun.py)
- matplotlib
- numpy(1.16.3)
- pandas
- scipy
- seaborn
- tensorflow(1.13.1)
