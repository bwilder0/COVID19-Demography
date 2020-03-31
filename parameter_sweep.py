import numpy as np
from numba import jit
import global_parameters
import scipy.special
import csv
from datetime import date
import numba
from seir_individual import run_complete_simulation
import pandas as pd
import datetime
from sklearn.metrics import mean_squared_error
import os
import argparse
from itertools import product


EXP_ROOT_DIR = "parameter_sweeps"

# This script is meant to be launched by slurm job scripts.
# See job.distributed_sweep_lombardy.sh for the how this script will be called.

# You may want to run --do_the_math to see what range of indices to pass to the job script.

# Ultimately you will need to run a number of jobs equal to: total_combos * N_SIMS_PER_COMBO / N_SIMS_PER_JOB


# Real
N=10000000.
load_population = True
N_INFECTED_START = 5.

# Test
# N=10000.
# load_population = True
# N_INFECTED_START = 5.


parser = argparse.ArgumentParser(description='Run lombardy parameter sweeps in a distributed way.')
parser.add_argument('--do_the_math', action='store_true', default=False, help='Given N_SIMS_PER_COMBO,\
                N_SIMS_PER_JOB, and the parameter ranges (see source code), will report the range of\
                indexes that you should pass to the slurm job script')
parser.add_argument('N_SIMS_PER_COMBO', type=int, help='Number of simulations you plan to run per parameter combo')
parser.add_argument('N_SIMS_PER_JOB', type=int, help='Number of simulations to run per job. Must be divisible by N_SIMS_PER_COMBO.')

parser.add_argument('--index', type=int, default=-1, help='Index of the job to run. see --do_the_math')
parser.add_argument('--seed_offset', type=int, default=0, help='seed = index + index_offset. \
    So to run a new indepent set of simulations for the same parameter combos, this \
    value must be larger than the total number of simulations you have run so far, \
    i.e., total_combos * N_SIMS_PER_COMBO.')
parser.add_argument('--sim_name', type=str, default="lombardy_batch", help='Always specify a new directory for your jobs!')

args = parser.parse_args()

p_infect_given_contact_list = [round(0.02+0.001*i, 3) for i in range(12)]
mortality_multiplier_list = [1, 2, 3, 4, 5]
start_date_list = [10, 13, 16, 19, 22, 25]

# p_infect_given_contact_list = [round(0.02+0.001*i, 3) for i in range(3)]
# mortality_multiplier_list = [1, 2]
# start_date_list = [19, 22, 25]


num_p = len(p_infect_given_contact_list)
num_m = len(mortality_multiplier_list)
num_d = len(start_date_list)

N_SIMS_PER_COMBO = args.N_SIMS_PER_COMBO
N_SIMS_PER_JOB = args.N_SIMS_PER_JOB

total_combos = num_p*num_m*num_d

if N_SIMS_PER_COMBO % N_SIMS_PER_JOB != 0:
    raise ValueError('Please ensure that N_SIMS_PER_COMBO is divisible by N_SIMS_PER_JOB')

if args.do_the_math:

    n_indices = int(total_combos*N_SIMS_PER_COMBO/N_SIMS_PER_JOB)
    print()
    print("NUM COMBOS:",total_combos)
    print("NUM INDICES:",n_indices)
    print()
    print("RUNNING ON SLURM")
    print("----------------------------")
    print("Please launch jobs indexed from 0 to %s, e.g.,"%(n_indices-1))
    print("sbatch --array=0-%s job.parameter_sweep.sh %s %s %s %s"%(n_indices-1, N_SIMS_PER_COMBO, N_SIMS_PER_JOB, args.sim_name, args.seed_offset))
    print()
    print("However, please note that you can only queue 10000 jobs at a time on FASRC!")
    print("")
    print("Note: the larger N_SIMS_PER_JOB, the more time and memory it will take.")
    print("But the smaller N_SIMS_PER_JOB, the less time and memory it will take, but more jobs and CPUs.")
    exit()

elif args.index == -1:
    raise ValueError('Please set the index or pass --do_the_math.')


INDEX = args.index

# TODO: Pass in a parameter
SEED_OFFSET = args.seed_offset


"""Run Parameter Sweep
"""


JOBS_PER_COMB0 = N_SIMS_PER_COMBO//N_SIMS_PER_JOB

param_combo_index = INDEX//JOBS_PER_COMB0
TRIAL_NUMBER = INDEX%JOBS_PER_COMB0

seed = INDEX * N_SIMS_PER_JOB + SEED_OFFSET


sim_name = args.sim_name
path = os.path.join(EXP_ROOT_DIR, sim_name)
if not os.path.exists(path):
    os.makedirs(path)

master_combo_list = list(product(p_infect_given_contact_list, mortality_multiplier_list, start_date_list))

this_combo = master_combo_list[param_combo_index]

pigc_val = this_combo[0]
mm_val = this_combo[1]
sd_val = this_combo[2]

print('combo',param_combo_index)
print(this_combo)


# What dates are we simulating

d0 = date(2020, 1, sd_val) # start
d_lockdown = date(2020, 3, 8) # lockdown
d_end = date(2020, 3, 22) # stop

# Parse real data for computing mse during sweep
data = pd.read_csv('validation_data/italy/lombardy_data_deaths.csv')
dates = []
actual_deaths = []
for i in range(len(data)):
    dates.append(datetime.datetime.strptime(data['Date'][i], '%m/%d/%y %H:%M').date())
    actual_deaths.append(data['Deaths'][i])

time_from_d0 = []
for i in range(len(dates)):
    time_from_d0.append((dates[i] - d0).days)


"""1. SET FIXED SIMULATION PARAMETERS
"""
params = numba.typed.Dict()
params['n'] = N

n_ages = 101
"""int: Number of ages. Currently ages 0-100, with each age counted separately"""
params['n_ages'] = float(n_ages)


country_list=['China','France', 'Germany', 'Iran', 'Italy',  'Spain', 'Switzerland',
              'Republic of Korea', 'UK', 'USA']
"""list: List of modeled countries."""

country = 'Italy'
# params['country'] = country


params['seed'] = float(seed)
"""int: Seed for random draws"""

np.random.seed(seed)


params['T'] = float((d_end - d0).days + 1)
#params['T'] = 30
"""int: Number of timesteps"""

params['initial_infected_fraction'] = N_INFECTED_START/params['n']
params['t_lockdown'] = float((d_lockdown - d0).days)


factor = 1
if country == 'China':
    factor = 50.0
else:
    factor = 10.0

params['lockdown_factor'] = factor
lockdown_factor_age = ((0, 14, factor), (15, 24, factor), (25, 39, factor), (40, 69, factor), (70, 100, factor))
lockdown_factor_age = np.array(lockdown_factor_age)


d_stay_home_start = date(2020, 3, 8)
params['t_stayhome_start'] = float((d_stay_home_start - d0).days)

fraction_stay_home = np.zeros(n_ages)
fraction_stay_home[:] = 0

params['mean_time_to_isolate_asympt'] = np.inf

params['mean_time_to_isolate'] = 4.6
"""float: Time from symptom onset to isolation
https://www.nejm.org/doi/pdf/10.1056/NEJMoa2001316?articleTools=true
optimistic estimate is time to seek first medical care, pessimistic is time to hospital admission
"""

params['asymptomatic_transmissibility'] = 0.55
"""float: How infectious are asymptomatic cases relative to symptomatic ones
https://science.sciencemag.org/content/early/2020/03/13/science.abb3221
"""

params['p_infect_given_contact'] = pigc_val
"""float: Probability of infection given contact between two individuals
This is currently set arbitrarily and will be calibrated to match the empirical r0
"""

params['mortality_multiplier'] = mm_val
"""
float: increase probability of death for all ages and comorbidities by this amount
"""

params['contact_tracing'] = float(False)
params['p_trace_outside'] = 1.0
params['p_trace_household'] = 0.75
d_tracing_start = date(2020, 2, 10)
params['t_tracing_start'] = float((d_tracing_start - d0).days)
"""
Whether contact tracing happens, and if so the probability of successfully 
identifying each within and between household infected individual
"""        

mean_time_to_isolate_factor = ((0, 14, 1), (14, 24, 1), (25, 39, 1), (40, 69, 1), (70, 100, 1))
mean_time_to_isolate_factor = np.array(mean_time_to_isolate_factor)

"""TODO: Find documented probabilities, age distribution or mean_time"""
params['p_documented_in_mild'] = 0.0


"""2c. Set transition times between states
We assume that all state transitions are exponentially distributed.
"""

#for now, the time for all of these events will be exponentially distributed
#from https://www.who.int/docs/default-source/coronaviruse/who-china-joint-mission-on-covid-19-final-report.pdf
params['mean_time_to_severe'] = 7.
params['mean_time_mild_recovery'] = 14.

#guessing based on time to mechanical ventilation as 14.5 days from
#https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(20)30566-3/fulltext
#and subtracting off the 7 to get to critical. This also matches mortality risk
#starting at 2 weeks in the WHO report
params['mean_time_to_critical'] = 7.5

#WHO gives 3-6 week interval for severe and critical combined
#using 4 weeks as mean for severe and 5 weeks as mean for critical
params['mean_time_severe_recovery'] = 28. - params['mean_time_to_severe']
params['mean_time_critical_recovery'] = 35. - params['mean_time_to_severe'] - params['mean_time_to_critical'] 
#mean_time_severe_recovery = mean_time_critical_recovery = 21

#mean_time_to_death = 35 #taking the midpoint of the 2-8 week interval
#update: use 35 - mean time to severe - mean time to critical as the excess time
#to death after reaching critical
#update: use 18.5 days as median time onset to death from
#https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(20)30566-3/fulltext
params['mean_time_to_death'] = 18.5 - params['mean_time_to_severe'] - params['mean_time_to_critical'] 
#mean_time_to_death = 1 #this will now be critical -> death

#probability of exposed individual becoming infected each time step
#set based on https://annals.org/aim/fullarticle/2762808/incubation-period-coronavirus-disease-2019-covid-19-from-publicly-reported
params['time_to_activation_mean'] = 1.621
params['time_to_activation_std'] = 0.418


# DON'T CHANGE: we don't want p infect household to recalibrate for different policy what ifs on mean time to isolate
MEAN_TIME_TO_ISOLATE = 4.6 # DON'T CHANGE
p_infect_household = global_parameters.get_p_infect_household(int(params['n']), MEAN_TIME_TO_ISOLATE, params['time_to_activation_mean'], params['time_to_activation_std'], params['asymptomatic_transmissibility'])



overall_p_critical_death = 0.49
"""float: Probability that a critical individual dies. This does _not_ affect
overall mortality, which is set separately, but rather how many individuals
end up in critical state. 0.49 is from
http://weekly.chinacdc.cn/en/article/id/e53946e2-c6c4-41e9-9a9b-fea8db1a8f51
"""


"""2. LOAD AND CALCULATE NON-FREE PARAMETERS
"""



"""2a.  Construct contact matrices
Idea: based on his/her age, each individuals has a different probability
      of contacting other individuals depending on their age
Goal: construct contact_matrix, which states that an individual of age i
     contacts Poission(contact[i][j]) contacts with individuals of age j
The data we have for this is based on contacts between individuals in age
intervals and must be converted.
"""

contact_matrix_age_groups_dict = {
    'infected_1': '0-4', 'contact_1': '0-4', 'infected_2': '5-9',
    'contact_2': '5-9', 'infected_3': '10-14', 'contact_3': '10-14',
    'infected_4': '15-19', 'contact_4': '15-19', 'infected_5': '20-24',
    'contact_5': '20-24', 'infected_6': '25-29', 'contact_6': '25-29',
    'infected_7': '30-34', 'contact_7': '30-34', 'infected_8': '35-39',
    'contact_8': '35-39', 'infected_9': '40-44', 'contact_9': '40-44',
    'infected_10': '45-49', 'contact_10': '45-49', 'infected_11': '50-54',
    'contact_11': '50-54', 'infected_12': '55-59', 'contact_12': '55-59',
    'infected_13': '60-64', 'contact_13': '60-64', 'infected_14': '65-69',
    'contact_14': '65-69', 'infected_15': '70-74', 'contact_15': '70-74',
    'infected_16': '75-79', 'contact_16': '75-79'}
"""dict: Mapping from interval names to age ranges."""

def read_contact_matrix(country):
    """Create a country-specific contact matrix from stored data.

    Read a stored contact matrix based on age intervals. Return a matrix of
    expected number of contacts for each pair of raw ages. Extrapolate to age
    ranges that are not covered.

    Args:
        country (str): country name.

    Returns:
        float n_ages x n_ages matrix: expected number of contacts between of a person
            of age i and age j is Poisson(matrix[i][j]).
    """
    matrix = np.zeros((n_ages, n_ages))
    with open('Contact_Matrices/{}/All_{}.csv'.format(country, country), 'r') as f:
        csvraw = list(csv.reader(f))
    col_headers = csvraw[0][1:-1]
    row_headers = [row[0] for row in csvraw[1:]]
    data = np.array([row[1:-1] for row in csvraw[1:]])
    for i in range(len(row_headers)):
        for j in range(len(col_headers)):
            interval_infected = contact_matrix_age_groups_dict[row_headers[i]]
            interval_infected = [int(x) for x in interval_infected.split('-')]
            interval_contact = contact_matrix_age_groups_dict[col_headers[j]]
            interval_contact = [int(x) for x in interval_contact.split('-')]
            for age_infected in range(interval_infected[0], interval_infected[1]+1):
                for age_contact in range(interval_contact[0], interval_contact[1]+1):
                    matrix[age_infected, age_contact] = float(data[i][j])/(interval_contact[1] - interval_contact[0] + 1)

    # extrapolate from 79yo out to 100yo
    # start by fixing the age of the infected person and then assuming linear decrease
    # in their number of contacts of a given age, following the slope of the largest
    # pair of age brackets that doesn't contain a diagonal term (since those are anomalously high)
    for i in range(interval_infected[1]+1):
        if i < 65: # 0-65
            slope = (matrix[i, 70] - matrix[i, 75])/5
        elif i < 70: # 65-70
            slope = (matrix[i, 55] - matrix[i, 60])/5
        elif i < 75: # 70-75
            slope = (matrix[i, 60] - matrix[i, 65])/5
        else: # 75-80
            slope = (matrix[i, 65] - matrix[i, 70])/5

        start_age = 79
        if i >= 75:
            start_age = 70
        for j in range(interval_contact[1]+1, n_ages):
            matrix[i, j] = matrix[i, start_age] - slope*(j - start_age)
            if matrix[i, j] < 0:
                matrix[i, j] = 0

    # fix diagonal terms
    for i in range(interval_infected[1]+1, n_ages):
        matrix[i] = matrix[interval_infected[1]]
    for i in range(int((100-80)/5)):
        age = 80 + i*5
        matrix[age:age+5, age:age+5] = matrix[79, 79]
        matrix[age:age+5, 75:80] = matrix[75, 70]
    matrix[100, 95:] = matrix[79, 79]
    matrix[95:, 100] = matrix[79, 79]

    return matrix

contact_matrix = read_contact_matrix(country)
"""n_ages x n_ages matrix: expected number of contacts between of a person
    of age i and age j is Poisson(matrix[i][j]).
"""


"""2b. Construct transition probabilities between disease severities
There are three disease states: mild, severe and critical.
- Mild represents sub-hospitalization.
- Severe is hospitalization.
- Critical is ICU.

The key results of this section are:
- p_mild_severe: n_ages x 2 x 2 matrix. For each age and comorbidity state
    (length two bool vector indicating whether the individual has diabetes and/or
    hypertension), what is the probability of the individual transitioning from
    the mild to severe state.
- p_severe_critical, p_critical_death are the same for the other state transitions.

All of these probabilities are proportional to the base progression rate
for an (age, diabetes, hypertension) state which is stored in p_death_target
and estimated via logistic regression.
"""

p_mild_severe_cdc = np.zeros(n_ages)
"""n_ages vector: The probability of transitioning from the mild to
    severe state for a patient of age i is p_mild_severe_cdc[i]. We will match
    these overall probabilities.

Source: https://www.cdc.gov/mmwr/volumes/69/wr/mm6912e2.htm?s_cid=mm6912e2_w#T1_down
Using the lower bounds for probability of hospitalization, since that's more
consistent with frequency of severe infection reported in
https://www.nejm.org/doi/full/10.1056/NEJMoa2002032 (at a lower level of age granularity).
"""
p_mild_severe_cdc[0:20] = 0.016
p_mild_severe_cdc[20:45] = 0.143
p_mild_severe_cdc[45:55] = 0.212
p_mild_severe_cdc[55:65] = 0.205
p_mild_severe_cdc[65:75] = 0.286
p_mild_severe_cdc[75:85] = 0.305
p_mild_severe_cdc[85:] = 0.313


#overall probability of progression from critical to severe
#https://www.ecdc.europa.eu/sites/default/files/documents/RRA-sixth-update-Outbreak-of-novel-coronavirus-disease-2019-COVID-19.pdf
#taking midpoint of the intervals
overall_p_severe_critical = (0.15 + 0.2) / 2

# go back to using CDC hospitalization rates as mild->severe
severe_critical_multiplier = overall_p_severe_critical / p_mild_severe_cdc
critical_death_multiplier = overall_p_critical_death / p_mild_severe_cdc

# get the overall CFR for each age/comorbidity combination by running the logistic model
"""
Mortality model. We fit a logistic regression to estimate p_mild_death from
(age, diabetes, hypertension) to match the marginal mortality rates from TODO.
The results of the logistic regression are used to set the disease severity
transition probabilities.
"""
c_age = np.loadtxt('c_age.txt', delimiter=',').mean(axis=0)
"""float vector: Logistic regression weights for each age bracket."""
c_diabetes = np.loadtxt('c_diabetes.txt', delimiter=',').mean(axis=0)
"""float: Logistic regression weight for diabetes."""
c_hyper = np.loadtxt('c_hypertension.txt', delimiter=',').mean(axis=0)
"""float: Logistic regression weight for hypertension."""
intervals = np.loadtxt('comorbidity_age_intervals.txt', delimiter=',')

def age_to_interval(i):
    """Return the corresponding comorbidity age interval for a specific age.

    Args:
        i (int): age.

    Returns:
        int: index of interval containing i in intervals.
    """
    for idx, a in enumerate(intervals):
        if i >= a[0] and i < a[1]:
            return idx
    return idx

p_death_target = np.zeros((n_ages, 2, 2))
for i in range(n_ages):
    for diabetes_state in [0,1]:
        for hyper_state in [0,1]:
            if i < intervals[0][0]:
                p_death_target[i, diabetes_state, hyper_state] = 0
            else:
                p_death_target[i, diabetes_state, hyper_state] = scipy.special.expit(
                    c_age[age_to_interval(i)] + diabetes_state * c_diabetes +
                    hyper_state * c_hyper)

# p_death_target *= params['mortality_multiplier']
# p_death_target[p_death_target > 1] = 1


#calibrate the probability of the severe -> critical transition to match the
#overall CFR for each age/comorbidity combination
#age group, diabetes (0/1), hypertension (0/1)
progression_rate = np.zeros((n_ages, 2, 2))
p_mild_severe = np.zeros((n_ages, 2, 2))
"""float n_ages x 2 x 2 vector: Probability a patient with a particular age combordity
    profile transitions from mild to severe state."""
p_severe_critical = np.zeros((n_ages, 2, 2))
"""float n_ages x 2 x 2 vector: Probability a patient with a particular age combordity
    profile transitions from severe to critical state."""
p_critical_death = np.zeros((n_ages, 2, 2))
"""float n_ages x 2 x 2 vector: Probability a patient with a particular age combordity
    profile transitions from critical to dead state."""

for i in range(n_ages):
    for diabetes_state in [0,1]:
        for hyper_state in [0,1]:
            progression_rate[i, diabetes_state, hyper_state] = (p_death_target[i, diabetes_state, hyper_state]
                                                                / (severe_critical_multiplier[i]
                                                                   * critical_death_multiplier[i])) ** (1./3)
            p_mild_severe[i, diabetes_state, hyper_state] = progression_rate[i, diabetes_state, hyper_state]
            p_severe_critical[i, diabetes_state, hyper_state] = severe_critical_multiplier[i]*progression_rate[i, diabetes_state, hyper_state]
            p_critical_death[i, diabetes_state, hyper_state] = critical_death_multiplier[i]*progression_rate[i, diabetes_state, hyper_state]
#no critical cases under 20 (CDC)
p_critical_death[:20] = 0
p_severe_critical[:20] = 0
#for now, just cap 80+yos with diabetes and hypertension
p_critical_death[p_critical_death > 1] = 1

p_mild_severe *= params['mortality_multiplier']**(1/3)
p_severe_critical *= params['mortality_multiplier']**(1/3)
p_critical_death *= params['mortality_multiplier']**(1/3)
p_mild_severe[p_mild_severe > 1] = 1
p_severe_critical[p_severe_critical > 1] = 1
p_critical_death[p_critical_death > 1] = 1



# n x 1 datatypes
r0_total = np.zeros((N_SIMS_PER_JOB,1))
mse_list = np.zeros((N_SIMS_PER_JOB,1))

# n x T datatypes
T = int(params['T'])
S_per_time = np.zeros((N_SIMS_PER_JOB, T))
E_per_time = np.zeros((N_SIMS_PER_JOB, T))
D_per_time = np.zeros((N_SIMS_PER_JOB, T))

Mild_per_time = np.zeros((N_SIMS_PER_JOB, T))
Severe_per_time = np.zeros((N_SIMS_PER_JOB, T))
Critical_per_time = np.zeros((N_SIMS_PER_JOB, T))
R_per_time = np.zeros((N_SIMS_PER_JOB, T))
Q_per_time = np.zeros((N_SIMS_PER_JOB, T))

r_0_over_time = np.zeros((N_SIMS_PER_JOB, T))
cfr_over_time = np.zeros((N_SIMS_PER_JOB, T))
fraction_over_70_time = np.zeros((N_SIMS_PER_JOB, T))
fraction_below_30_time = np.zeros((N_SIMS_PER_JOB, T))
median_age_time = np.zeros((N_SIMS_PER_JOB, T))

# n x n_age_groups x T datatypes
age_groups = ((0, 14), (15, 24), (25, 39), (40, 69), (70, 100))
infected_by_age_by_time = np.zeros((N_SIMS_PER_JOB, len(age_groups), T))
total_age_by_time = np.zeros((N_SIMS_PER_JOB, len(age_groups), T))
CFR_by_age_by_time = np.zeros((N_SIMS_PER_JOB, len(age_groups), T))
dead_by_age_by_time = np.zeros((N_SIMS_PER_JOB, len(age_groups), T))



for i in range(N_SIMS_PER_JOB):

    params['seed']+=1

    S, E, Mild, Documented, Severe, Critical, R, D, Q, num_infected_by,time_documented, \
    time_to_activation, time_to_death, time_to_recovery, time_critical, time_exposed, num_infected_asympt,\
        age, time_infected, time_to_severe =  run_complete_simulation(int(params['seed']),country, contact_matrix, p_mild_severe, p_severe_critical, \
                                   p_critical_death, mean_time_to_isolate_factor, lockdown_factor_age, p_infect_household, \
                                   fraction_stay_home, params, load_population=load_population)

    S_per_time[i] = S.sum(axis=1)
    E_per_time[i] = E.sum(axis=1)
    D_per_time[i] = D.sum(axis=1)

    Mild_per_time[i] = Mild.sum(axis=1)
    Severe_per_time[i] = Severe.sum(axis=1)
    Critical_per_time[i] = Critical.sum(axis=1)
    R_per_time[i] = R.sum(axis=1)
    Q_per_time[i] = Q.sum(axis=1)

    r0_total[i] = num_infected_by[np.logical_and(time_exposed <= 20, time_exposed > 0)].mean()


    import time
    start = time.time()
    #This is all for analyzing the age distribution of infection, CFR over time, etc.
    for t in range(T):
        if (time_exposed == t).sum() > 0:
            r_0_over_time[i, t] = num_infected_by[time_exposed == t].mean()
            cfr_over_time[i, t] = D[-1, time_exposed == t].sum()/(D[-1, time_exposed == t].sum() + R[-1, time_exposed == t].sum())
            fraction_over_70_time[i, t] = (age[time_exposed == t] >= 70).mean()
            fraction_below_30_time[i, t] = (age[time_exposed == t] < 30).mean()
            median_age_time[i, t] = np.median(age[time_exposed == t])


        for idx, (lower, upper) in enumerate(age_groups):
            total = 0
            infected = 0
            age_group_array = np.logical_and(age >= lower, age <= upper)

            age_group_total = age_group_array.sum()

            age_group_susceptible_this_timestep = np.logical_and(S[t], age_group_array).sum()
            infected = age_group_total - age_group_susceptible_this_timestep

            infected_by_age_by_time[i, idx, t] = infected
            total_age_by_time[i, idx, t] = age_group_total

            d_end_per_age_per_timestep = np.logical_and(D[-1], time_exposed == t)
            d_end_per_age_per_timestep = np.logical_and(d_end_per_age_per_timestep, age_group_array).sum()

            r_end_per_age_per_timestep = np.logical_and(R[-1], time_exposed == t)
            r_end_per_age_per_timestep = np.logical_and(r_end_per_age_per_timestep, age_group_array).sum()

            CFR_by_age_by_time[i, idx, t] = d_end_per_age_per_timestep / (d_end_per_age_per_timestep + r_end_per_age_per_timestep)

            dead_by_age_by_time[i, idx, t] = np.logical_and(D[t], age_group_array).sum()

    end = time.time()
    print('age/time reporting took %s seconds'%(end-start))





# additionally we can only compute mse for days we have data
f = time_from_d0[0]
l = time_from_d0[-1]+1

for i,D in enumerate(D_per_time):

    mse = mean_squared_error(D[f:l], actual_deaths)
    mse_list[i] = mse




path = os.path.join(EXP_ROOT_DIR, sim_name)
fname = '%s_paramsweep_n%s_i%s_N%s_p%s_m%s_s%s'%(sim_name, params['n'], param_combo_index, TRIAL_NUMBER, pigc_val, mm_val, sd_val)
fname += '_%s.csv'

fname = path = os.path.join(path, fname)

datatypes_n_1 = [
    {
        'name':'r0_tot', 
        'data':r0_total
    },
    {
        'name':'mse',
        'data':mse_list
    }
]

datatypes_n_t = [
    {
        'name':'susceptible', 
        'data':S_per_time
    },
    {
        'name':'exposed', 
        'data':E_per_time
    },
    {
        'name':'deaths', 
        'data':D_per_time
    },
    {
        'name':'mild', 
        'data':Mild_per_time
    },
    {
        'name':'severe', 
        'data':Severe_per_time
    },
    {
        'name':'critical', 
        'data':Critical_per_time
    },
    {
        'name':'recovered', 
        'data':R_per_time
    },
    {
        'name':'quarantine', 
        'data':Q_per_time
    },


    {
        'name':'r0_time', 
        'data':r_0_over_time
    },
    {
        'name':'cfr_time', 
        'data':cfr_over_time
    },
    {
        'name':'frac_over_70_time', 
        'data':fraction_over_70_time
    },
    {
        'name':'frac_below_30_time', 
        'data':fraction_below_30_time
    },
    {
        'name':'median_age_time', 
        'data':median_age_time
    },
    
]

# These will be the biggest storage burden. Cut where you can.
datatypes_n_a_t = [
    {
        'name':'infected_age_time', 
        'data':infected_by_age_by_time
    },
    {
        'name':'total_age_time',
        'data':total_age_by_time
    },
    {
        'name':'cfr_age_time',
        'data':CFR_by_age_by_time
    },
    {
        'name':'dead_age_time',
        'data':dead_by_age_by_time
    }
]


for d in datatypes_n_1:
    df = pd.DataFrame(d['data'])
    df.to_csv(fname%d['name'],sep='\t',index=False,header=False, na_rep='NA')


for d in datatypes_n_t:
    df = pd.DataFrame(d['data'])
    df.to_csv(fname%d['name'],sep='\t',index=False,header=False, na_rep='NA')


for d in datatypes_n_a_t:
    # Flatten the data -- remember to unflatten for analysis
    d['data'] = d['data'].reshape(N_SIMS_PER_JOB, len(age_groups)*T)
    df = pd.DataFrame(d['data'])
    df.to_csv(fname%d['name'],sep='\t',index=False,header=False, na_rep='NA')


# import json
# with open('parameter_sweeps/%s/%s_paramsweep_n%s_i%s_p%s_m%s_s%s_plotdata.json'%(sim_name,sim_name, params['n'], param_combo_index, pigc_val, mm_val, sd_val), 'w') as outfile:
#     json.dump(result_dict, outfile)


