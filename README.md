# Recurrent Neural-Linear Posterior Sampling for Non-Stationary Contextual Bandits
Code associated with the paper "Recurrent Neural-Linear Posterior Sampling for Non-Stationary Contextual Bandits".

## Instructions

### Hyperparameter grid

Create config files for a specific bandit problem instance and the policies to be evaluated:

```bash
python3 hgrid.py experiment_folder/
```

### Experiments

Run an individual experiment on a folder with a config file:
```bash
python3 run.py experiment_folder/single_experiment/
```

Run multiple experiments in parallel with 10 jobs (requires tmux):

```bash
python3 multirun.py experiment_folder/ 10
```

### Analysis 

Create a csv file that summarizes the return (mean and standard deviation over independent runs) for the different policies:

```bash
python3 create_summary.py experiment_folder/
```

Create a plot comparing the regret/return of different policies:

```bash
# Regret
python3 analyse.py experiment_folder/ --regret

# Return
python3 analyse.py experiment_folder/ --return
```

## Dependencies

- tmux (for multirun.py)

#### Python packages -

- matplotlib
- numpy
- pandas
- scipy
- seaborn
- tensorflow-1.x
 
