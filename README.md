# Agent-based Modeling of COVID19 using age distribution and family structure

- Code for the preprint: The Role of Age Distribution and Family Structure on COVID-19 Dynamics: A Preliminary Modeling Assessment for Hubei and Lombardy

## Running experiments

- To get started:
`python3 run_simulation.py`

- To run simulations of different age-based policies
`python3 simulate_agepolicies.py -h`

- To run large scale parameter sweeps
`python3 parameter_sweep.py -h`
  - Also see: README_parameter_sweep.txt

## The simulator

See `seir_individual.py`

## Fitting comorbidity models

See `comorbidity_inference.py`. Estimates from the paper are already saved, so
you can start doing simulations without rerunning this step unless you would like
to fit your own model.
