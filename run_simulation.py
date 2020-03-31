import numpy as np
import global_parameters
import scipy.special
import csv
import pickle
from datetime import date
import numba
from seir_individual import run_complete_simulation

"""1. SET FIXED SIMULATION PARAMETERS
"""


params = numba.typed.Dict()
params['n'] = 10000000.

n_ages = 101
"""int: Number of ages. Currently ages 0-100, with each age counted separately"""
params['n_ages'] = float(n_ages)


country_list=['China','France', 'Germany', 'Iran', 'Italy',  'Spain', 'Switzerland',
              'Republic of Korea', 'UK', 'USA']
"""list: List of modeled countries."""

country = 'Italy'

load_population = False
seed = 0
params['seed'] = float(seed)
"""int: Seed for random draws"""

np.random.seed(seed)

if country == 'China':
    d0 = date(2019, 11, 15)
    d_lockdown = date(2020, 1, 23)
    d_stay_home_start = date(3000, 3, 8) 
    d_end = date(2020, 3, 21) 
elif country == 'Republic of Korea':
    d0 = date(2020, 2, 1)
    d_lockdown = date(3000, 3, 21)
    d_stay_home_start = date(3000, 3, 8) 
    d_end = date(2020, 3, 21) 
elif country== 'Italy':
    # What dates are we simulating
    d0 = date(2020, 1, 22) # start
    d_lockdown = date(2020, 3, 8) # lockdown
    d_end = date(2020, 3, 22) # stop
    d_stay_home_start = date(3000, 3, 8) 

params['t_stayhome_start'] = float((d_stay_home_start - d0).days)

fraction_stay_home = np.zeros(n_ages)
fraction_stay_home[:] = 0

params['T'] = float((d_end - d0).days)
#params['T'] = 70
"""int: Number of timesteps"""

if country == 'China':
    params['initial_infected_fraction'] = 5./params['n']
elif country == 'Republic of Korea':
    params['initial_infected_fraction'] = 500./params['n']
elif country == 'Italy':
    params['initial_infected_fraction'] = 5./params['n']
    
params['t_lockdown'] = float((d_lockdown - d0).days)

factor = 1
if country == 'China':
    factor = 50.0
else:
    factor = 10.0

params['lockdown_factor'] = factor
lockdown_factor_age = ((0, 14, factor), (15, 24, factor), (25, 39, factor), (40, 69, factor), (70, 100, factor))
lockdown_factor_age = np.array(lockdown_factor_age)

params['mean_time_to_isolate_asympt'] = 10000

params['mean_time_to_isolate'] = 4.6
"""float: Time from symptom onset to isolation
https://www.nejm.org/doi/pdf/10.1056/NEJMoa2001316?articleTools=true
This is take from the mean time to first seek medical care.
"""

params['asymptomatic_transmissibility'] = 0.55
"""float: How infectious are asymptomatic cases relative to symptomatic ones
https://science.sciencemag.org/content/early/2020/03/13/science.abb3221
"""

if country == 'China' or country == 'Republic of Korea':
    params['p_infect_given_contact'] = 0.020
elif country == 'Italy':
    params['p_infect_given_contact'] = 0.029
"""float: Probability of infection given contact between two individuals
This is currently set arbitrarily and will be calibrated to match the empirical r0
"""
if country == 'Italy':
    params['mortality_multiplier'] = 4.0
else:
    params['mortality_multiplier'] = 1.0
"""
float: increase probability of death for all ages and comorbidities by this amount
"""

if country != 'Republic of Korea':
    params['contact_tracing'] = float(False)
else:
    params['contact_tracing'] = float(True)
params['p_trace_household'] = 1
params['p_trace_outside'] = 0.95

d_tracing_start = date(2020, 2, 28) 
params['t_tracing_start'] = float((d_tracing_start - d0).days)
"""
Whether contact tracing happens, and if so the probability of successfully 
identifying each within and between household infected individual
"""

cumulative_documentation_mild = 0.00000000001
"""Target for the cumulative fraction of mild cases which never become severe that 
are documnted. This is used to calibrate p_documented_in_mild, which is the per-
day probability"""

isolation_filename='./isolation_exps/reducedBy3_for_group5.pickle'
mean_time_to_isolate_factor = ((0, 14, 1), (14, 24, 1), (25, 39, 1), (40, 69, 1), (70, 100, 0.347))
mean_time_to_isolate_factor = np.array(mean_time_to_isolate_factor)


"""TODO: Find documented probabilities, age distribution or mean_time"""
params['p_documented_in_mild'] = 0.2



"""2c. Set transition times between states
We assume that all state transitions are exponentially distributed.
"""

#for now, the time for all of these events will be exponentially distributed
#from https://www.who.int/docs/default-source/coronaviruse/who-china-joint-mission-on-covid-19-final-report.pdf
params['mean_time_to_severe'] = 7.
params['mean_time_mild_recovery'] = 14.

params['p_documented_in_mild'] = global_parameters.calibrate_p_document_mild(cumulative_documentation_mild, country, None, params['mean_time_mild_recovery'], None)
"""
Probability of documentation (outside of contact tracing) for a mild case which
never becomes severe 
"""

#guessing based on time to mechanical ventilation as 14.5 days from
#https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(20)30566-3/fulltext
#and subtracting off the 7 to get to critical. This also matches mortality risk
#starting at 2 weeks in the WHO report
params['mean_time_to_critical'] = 7.5

#WHO gives 3-6 week interval for severe and critical combined
#using 4 weeks as mean for severe and 5 weeks as mean for critical
params['mean_time_severe_recovery'] = 28. - params['mean_time_to_severe']
params['mean_time_critical_recovery'] = 35. - params['mean_time_to_severe'] - params['mean_time_to_critical'] 

#update: use 18.5 days as median time onset to death from
#https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(20)30566-3/fulltext
params['mean_time_to_death'] = 18.5 - params['mean_time_to_severe'] - params['mean_time_to_critical'] 

#probability of exposed individual becoming infected each time step
#set based on https://annals.org/aim/fullarticle/2762808/incubation-period-coronavirus-disease-2019-covid-19-from-publicly-reported
params['time_to_activation_mean'] = 1.621
params['time_to_activation_std'] = 0.418


p_infect_household = global_parameters.get_p_infect_household(int(params['n']), 4.6, params['time_to_activation_mean'], params['time_to_activation_std'], params['asymptomatic_transmissibility'])


overall_p_critical_death = 0.49
"""float: Probability that a critical individual dies. This does _not_ affect
overall mortality, which is set separately, but rather how many individuals
end up in critical state. 0.49 is from
http://weekly.chinacdc.cn/en/article/id/e53946e2-c6c4-41e9-9a9b-fea8db1a8f51
"""


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

#p_death_target *= params['mortality_multiplier']
#p_death_target[p_death_target > 1] = 1

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

#scale up all transitions proportional to the mortality_multiplier parameter
p_mild_severe *= params['mortality_multiplier']**(1/3)
p_severe_critical *= params['mortality_multiplier']**(1/3)
p_critical_death *= params['mortality_multiplier']**(1/3)
p_mild_severe[p_mild_severe > 1] = 1
p_severe_critical[p_severe_critical > 1] = 1
p_critical_death[p_critical_death > 1] = 1


'''
3. Run the simulation and save out fine-grained results
'''
num_runs = 30
n = int(params['n'])

T = int(params['T'])
all_c = []
all_s = []
all_d = []
r_0_over_time = np.zeros((num_runs, int(params['T'])))
cfr_over_time = np.zeros((num_runs, int(params['T'])))
fraction_over_70_time = np.zeros((num_runs, int(params['T'])))
fraction_below_30_time = np.zeros((num_runs, int(params['T'])))
median_age_time = np.zeros((num_runs, int(params['T'])))
total_infections_time = np.zeros((num_runs, int(params['T'])))
total_documented_time = np.zeros((num_runs, int(params['T'])))
dead_by_age = np.zeros((num_runs, n_ages))
total_deaths_time = np.zeros((num_runs, int(params['T'])))
all_final_s = np.zeros((num_runs, int(params['n'])))
for i in range(num_runs):
    print(i)
    S, E, Mild, Documented, Severe, Critical, R, D, Q, num_infected_by, time_documented, \
    time_to_activation, time_to_death, time_to_recovery, time_critical, time_exposed, \
    num_infected_asympt, age, time_infected, time_to_severe \
        =  run_complete_simulation(seed + i,country, contact_matrix, p_mild_severe, p_severe_critical, \
                                   p_critical_death, mean_time_to_isolate_factor, \
                                   lockdown_factor_age, p_infect_household, fraction_stay_home, params, load_population)
        
        
    #This is all for analyzing the age distribution of infection, CFR over time, etc.
    for t in range(T):
        if (time_exposed == t).sum() > 0:
            r_0_over_time[i, t] = num_infected_by[time_exposed == t].mean()
            cfr_over_time[i, t] = D[-1, time_exposed == t].sum()/(D[-1, time_exposed == t].sum() + R[-1, time_exposed == t].sum())
            fraction_over_70_time[i, t] = (age[time_exposed == t] >= 70).mean()
            fraction_below_30_time[i, t] = (age[time_exposed == t] < 30).mean()
            median_age_time[i, t] = np.median(age[time_exposed == t])
    total_infections_time[i] = (params['n'] - S.sum(axis=1) - E.sum(axis=1))
    total_documented_time[i] = Documented.sum(axis=1)
    for patient_age in range(n_ages):
        dead_by_age[i, patient_age] = D[-1, age == patient_age].sum()
    total_deaths_time[i] = D.sum(axis=1)
    all_final_s[i] = S[-1]
            

    print(d0, params['p_infect_given_contact'], D.sum(axis=1)[-1], Documented.sum(axis=1)[-1], total_infections_time[i, -1])
#import pickle
pickle.dump((r_0_over_time, cfr_over_time, fraction_over_70_time, fraction_below_30_time, median_age_time, total_infections_time, total_documented_time, dead_by_age, total_deaths_time, all_final_s), open('results_{}_{}_{}.pickle'.format(country, params['p_infect_given_contact'], d0.day), 'wb'))
