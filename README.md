# Agent-based Modeling of COVID19 using age distribution and family structure

Code for the [preprint](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3564800): The Role of Age Distribution and Family Structure on COVID-19 Dynamics: A Preliminary Modeling Assessment for Hubei and Lombardy

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

## Data sources needed to run for specific country


### Data already available in model (most countries)
1. **Age distribution**: [Source](https://population.un.org/household/resources/About_the_United_Nations_Database_on_the_Household_Size_and_Composition_2019.pdf)
2. **Age specific fertility of mothers**: [Source](https://www.un.org/en/development/desa/population/publications/dataset/fertility/age-fertility.asp)
3. **Contact Matrices**: [Source](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5609774/#pcbi.1005697.s001)

### Outside data needed:
4. **Distribution of household types**: single household, couple without children, single parent +1/2/3 children, couple +1/2/3 children, family without a nucleus, nucleus with other persons, households with two or more nuclei (a and b)
5. **Age specific rates of diabetes**
6. **Age specific rates of hypertension**




