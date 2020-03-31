import pandas as pd 
import numpy as np 
import os
import glob
from itertools import product
import datetime
import matplotlib.pyplot as plt

country = 'China'
OLD_FORMATTING = True

def ci(a, alpha):
    a = a.copy()
    a.sort()
    return a[int((alpha/2)*a.shape[0])], a[int((1-(alpha/2))*a.shape[0])]


if country == 'Italy':
    p_infect_given_contact_list = [round(0.02+0.001*i, 3) for i in range(12)]
    mortality_multiplier_list = [1, 2, 3, 4, 5]
    start_date_list = [10, 13, 16, 19, 22, 25]


elif country == 'China':
    p_infect_given_contact_list = [0.018, 0.019, 0.0195, 0.02, 0.021, 0.022, 0.023]
    mortality_multiplier_list = [1]
    start_date_list = [8, 10, 12, 13, 15, 17, 19]


if country == 'Italy':
    data = pd.read_csv('validation_data/italy/lombardy_data_deaths.csv')
elif country == 'China':
    data = pd.read_csv('validation_data/china/hubei.csv')
elif country == 'Republic of Korea':
    data = pd.read_csv('validation_data/south_korea/south_korea.csv')

dates = []
deaths = []
confirmed= [ ]
for i in range(len(data)):
    if country == 'Italy':
        dates.append(datetime.datetime.strptime(data['Date'][i], '%m/%d/%y %H:%M').date())
    if country == 'China' or country == 'Republic of Korea':
        dates.append(datetime.datetime.strptime(data['Date'][i], '%m/%d/%y').date())
    deaths.append(data['Deaths'][i])
    if country == 'Republic of Korea':
        confirmed.append(data['Confirmed'][i])


N = 10000000

all_final_deaths = []
all_final_infected = []
all_r0 = []

if country == 'China':
    target_p = 0.020
    target_start = 13
    target_mult = 1
elif country == 'Italy':
    target_p = 0.029
    target_start = 22
    target_mult = 4


if country == 'Italy':

    N_COMBOS = len(list(product(p_infect_given_contact_list, mortality_multiplier_list, start_date_list)))
    NUM_DATES = len(start_date_list)
    runs = ['lombardy_327_combined_2']
    
    # how many jobs were in each run? i.e., N_SIMS_PER_COMBO / N_SIMS_PER_JOB
    # For 'combined' directories, pass 1
    NUM_JOBS = [1] 


elif country == 'China':
    N_COMBOS = 49
    dirname = os.path.join('parameter_sweeps','hubei_distributed')
    name_template = 'hubei_distributed_paramsweep_n10000000.0_i%s_*_%s.csv'
    
    NUM_DATES = 7
    
    runs = [
        'hubei_distributed_independent_0',
        'hubei_distributed_independent_1',
        'hubei_distributed_independent_2',
    ]
    NUM_JOBS = [1,1,1] 


# For new formatting that does include N
name_templates = []
for run_name in runs:
    dirname = os.path.join('parameter_sweeps',run_name)
    name_template = run_name+'_paramsweep_n10000000.0_i%s_N%s_p%s_m%s_s%s_%s.csv'
    name_templates.append(os.path.join(dirname, name_template))


# For old formatting that dind't include N (i.e. our China data)
if OLD_FORMATTING:
    name_templates = []
    for run_name in runs:
        dirname = os.path.join('parameter_sweeps',run_name)
        name_template = run_name+'_paramsweep_n10000000.0_i%s_p%s_m%s_s%s_%s.csv'
        name_templates.append(os.path.join(dirname, name_template))


data = {
    'mse': {'mean':np.zeros(N_COMBOS), 'std':np.zeros(N_COMBOS)},
    'infected': {'mean':np.zeros(N_COMBOS), 'std':np.zeros(N_COMBOS)}, 
    'percentile': {'mean':np.zeros(N_COMBOS), 'std':np.zeros(N_COMBOS)}, 
    'documentation': {'mean':np.zeros(N_COMBOS), 'std':np.zeros(N_COMBOS)}, 
    'r0': {'mean':np.zeros(N_COMBOS), 'std':np.zeros(N_COMBOS)},
    'cfr': {'mean':np.zeros(N_COMBOS), 'std':np.zeros(N_COMBOS)},
    'age': {'mean':np.zeros(N_COMBOS), 'std':np.zeros(N_COMBOS)}

}

master_combo_list = list(product(p_infect_given_contact_list, mortality_multiplier_list, start_date_list))


# num_d is the expected dimension of the input data:
# 1 --> numtrials x 1
# 2 --> numtrials x T
# 3 --> numtrials x num_ages x T
# at the time of writing num_ages is 5
def get_data(datatype, i, name_templates, num_d, num_ages=5):

    this_combo = master_combo_list[i]

    pigc_val = this_combo[0]
    mm_val = this_combo[1]
    sd_val = this_combo[2]

    trial_data = []
    for ind,name_template in enumerate(name_templates):
        for trial_num in range(NUM_JOBS[ind]):
            

            # TODO - RENAME FILES SO WE CAN GET RID OF COUNTRY SPECIFIC STUFF

            if datatype=='r0' and country=="Italy":
                datatype='r0_tot'

            # fname = ""
            # if country == "Italy":
            #     fname = name_template%(i, trial_num, pigc_val, mm_val, sd_val, datatype)
            # else:

            fname = None 
            if not OLD_FORMATTING:
                name_template%(i, 0, pigc_val, mm_val, sd_val, datatype)
            else:
                fname = name_template%(i, pigc_val, mm_val, sd_val, datatype)

            batch_data = pd.read_csv(fname,header=None,sep='\t').values

            if batch_data.shape[0]==0:
                batch_data = np.array([[0]])

            if num_d == 3:
                batch_data = batch_data.reshape(batch_data.shape[0], num_ages, -1)

            trial_data.append(batch_data)
    trial_data = np.concatenate(trial_data,axis=0)

    return trial_data


for i in range(N_COMBOS):

    # check MSE 
    datatype = 'mse'
    num_d = 1
    trial_data = get_data(datatype,i,name_templates,num_d)


    data[datatype]['mean'][i] = trial_data.mean()
    data[datatype]['std'][i] = trial_data.std()


    datatype = 'susceptible'
    num_d = 2
    susceptible_data = get_data(datatype,i,name_templates,num_d)

    datatype = 'exposed'
    num_d = 2
    exposed_data = get_data(datatype,i,name_templates,num_d)
    

    datatype = 'deaths'
    num_d = 2
    deaths_data = get_data(datatype,i,name_templates,num_d)
    if master_combo_list[i] == (target_p, target_mult, target_start):
        deaths_target_date = deaths_data

    data['percentile']['mean'][i] = (deaths[-1] >= deaths_data[:, -1]).mean()
    
    all_final_deaths.extend(deaths_data[:, -1])


    infected = N - susceptible_data[:,-1] - exposed_data[:,-1]


    trial_data = infected
    datatype='infected'
    data[datatype]['mean'][i] = trial_data.mean()
    data[datatype]['std'][i] = trial_data.std()
    all_final_infected.extend(infected)
    

    datatype = 'documentation'
    if country == 'Italy':
        confirmed_cases = 27206.
    elif country == 'China':
        confirmed_cases = 67800
    trial_data = confirmed_cases/infected
    
    data[datatype]['mean'][i] = np.median(trial_data)
    data[datatype]['std'][i] = trial_data.std()


    datatype = 'r0'
    num_d = 1
    trial_data = get_data(datatype,i,name_templates,num_d)

    # Convert NaNs to 0 from rare non-spreading trials
    trial_data = np.nan_to_num(trial_data)

    if trial_data.flatten().shape[0] != 100:
        raise Exception()
    all_r0.extend(trial_data.flatten())
    data[datatype]['mean'][i] = np.median(trial_data)
    data[datatype]['std'][i] = trial_data.std()


print(data['mse']['mean'])
print(len(data['mse']['mean']))


if country == 'Italy':
    colnames = start_date_list
    rownames = list(product(p_infect_given_contact_list, mortality_multiplier_list))
elif country == 'China':
    colnames = start_date_list
    rownames = list(product(p_infect_given_contact_list, mortality_multiplier_list))


datatype = 'mse'
data[datatype]['mean'] = data[datatype]['mean'].reshape(-1, NUM_DATES) # make the columns the dates
data[datatype]['std'] = data[datatype]['std'].reshape(-1, NUM_DATES) # make the columns the dates

df_mse_mean = pd.DataFrame(data[datatype]['mean'], columns=colnames, index=rownames)
df_mse_std = pd.DataFrame(data[datatype]['std'], columns=colnames, index=rownames)


datatype = 'infected'
data[datatype]['mean'] = data[datatype]['mean'].reshape(-1, NUM_DATES) # make the columns the dates
data[datatype]['std'] = data[datatype]['std'].reshape(-1, NUM_DATES) # make the columns the dates


df_infectd_mean = pd.DataFrame(data[datatype]['mean'], columns=colnames, index=rownames)
df_infected_std = pd.DataFrame(data[datatype]['std'], columns=colnames, index=rownames)

datatype = 'percentile'
data[datatype]['mean'] = data[datatype]['mean'].reshape(-1, NUM_DATES) # make the columns the dates
df_percentile_mean = pd.DataFrame(data[datatype]['mean'], columns=colnames, index=rownames)

datatype = 'documentation'
data[datatype]['mean'] = data[datatype]['mean'].reshape(-1, NUM_DATES) # make the columns the dates
df_documentation_mean = pd.DataFrame(data[datatype]['mean'], columns=colnames, index=rownames)

datatype = 'r0'
data[datatype]['mean'] = data[datatype]['mean'].reshape(-1, NUM_DATES) # make the columns the dates
data[datatype]['std'] = data[datatype]['std'].reshape(-1, NUM_DATES) # make the columns the dates

df_r0_mean = pd.DataFrame(data[datatype]['mean'], columns=colnames, index=rownames)
df_r0_std = pd.DataFrame(data[datatype]['std'], columns=colnames, index=rownames)



from mpl_heatmap_code import heatmap, annotate_heatmap
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib as mpl
mpl.rcParams['ps.useafm'] = True
mpl.rcParams['pdf.use14corefonts'] = True
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
to_plot = 'documentation'
#for current_mult in [1,2,3,4,5]:
for current_mult in mortality_multiplier_list:
    rows_include = []
    for i in range(len(df_percentile_mean.index)):
        if df_percentile_mean.index[i][1] == current_mult:
            rows_include.append(i)
        
    
    percentile_data = df_percentile_mean.iloc[rows_include].to_numpy()
    documentation_data = df_documentation_mean.iloc[rows_include].to_numpy()
    r0_data = df_r0_mean.iloc[rows_include].to_numpy()
    
    fig, ax = plt.subplots(figsize=(3.7, 6))
    plt.text(0, 1.07, r'$\mathbf{d_{\mathrm{mult}} = ' + str(current_mult) + r'}$', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=15)
    goodness = percentile_data*(1-percentile_data)
    #goodness = df_mse_mean.to_numpy()
    
    #axins1 = inset_axes(ax,
    #                    width="90%",  # width = 50% of parent_bbox width
    #                    height="1%",  # height : 5%
    #                    loc='upper right')
    col_labels = list(df_percentile_mean.iloc[rows_include].columns)
    if country == 'Italy':
        start_month = '1/'
    elif country == 'China':
        start_month = '11/'
    col_labels = [start_month + str(x) for x in col_labels]
    row_labels = list(df_percentile_mean.iloc[rows_include].index)
    row_labels = ['{0:.3f}'.format(x[0]) for x in row_labels]
    plt.xlabel(r'$t_0$', fontsize=20)
#    cb_kws = dict(use_gridspec=False, location='top right', aspect=50, shrink=0.75, norm=Normalize(vmin=0, vmax=0.25), ticks=[0, 0.05, 0.10, 0.15, 0.2, 0.25])
    
    axins = inset_axes(ax,
                   height="25%",  # width = 5% of parent_bbox width
                   width="75%",  # height : 50%
                   loc='upper right',
                   bbox_to_anchor=(0., 1.0, 0.95, .102),
                   bbox_transform=ax.transAxes,
                   borderpad=0,
                   )
    cb_kws = dict(cax = axins, norm=Normalize(vmin=0, vmax=0.25), ticks=[0, 0.05, 0.10, 0.15, 0.2, 0.25], orientation="horizontal")

    
    im, cbar = heatmap(goodness, row_labels, col_labels, ax=ax,
                       cmap="YlGn", cbarlabel="", clim = [0, 0.25], cbar_kw = cb_kws)
    #use_gridspec=False, location='top',
    #cax=axins1
    #im, cbar = heatmap(goodness, list(df_percentile_mean.index), list(df_percentile_mean.columns), ax=ax,
    #                   cmap="YlGn_r", cbarlabel="", norm=LogNorm(vmin=goodness.min(), vmax=goodness.max()))
    #ax.imshow(percentile_data)
    if to_plot == 'r0':
        data_threshold = np.inf
        label_data = r0_data
    elif to_plot == 'documentation':
        data_threshold = 1
        label_data = documentation_data
    
    texts = annotate_heatmap(im, data=label_data, color_data=goodness, valfmt="{x:.3f}", threshold=0.15, data_threshold = data_threshold)
    ax.set_ylabel(r'$p_{\mathrm{inf}}$', fontsize=20)
    fig.tight_layout()
    #plt.show()
    plt.savefig('img/heatmaps/heatmap_{}_{}_{}.pdf'.format(to_plot, country, current_mult))



print()
print('Plausible range for parameter values')
cutoff = 0.2
include = percentile_data*(1-percentile_data) >= cutoff
doc_data = df_documentation_mean.to_numpy()
r0_data = df_r0_mean.to_numpy()
print(cutoff, doc_data[include].min(), doc_data[include].max())
print(cutoff, r0_data[include].min(), r0_data[include].max())


from datetime import date

if country == 'China':
    d0 = date(2019, 11, target_start)
    d_lockdown = date(2020, 1, 23)
    d_end = date(2020, 3, 21)
elif country == 'Italy':
    d0 = date(2020, 1, target_start) # start
    d_lockdown = date(2020, 3, 8) # lockdown
    d_end = date(2020, 3, 22) # stop


t_lockdown = (d_lockdown - d0).days

plt.figure()
time_from_d0 = []
for i in range(len(dates)):
    time_from_d0.append((dates[i] - d0).days)


#plt.plot(D.sum(axis=1), lw=2)
for i in range(deaths_target_date.shape[0]):
    plt.plot(deaths_target_date[i], alpha = 0.075, c='C0')
plt.scatter(time_from_d0, deaths, color='k', s=10)
plt.plot(np.median(deaths_target_date, axis=0), color = 'g', lw = 2.5)
plt.ylim(0, 1.05*deaths_target_date[:, -1].max())
plt.xlim(0,deaths_target_date.shape[1]-1)
plt.ylabel('Total deaths', fontsize=23)
#plt.xlabel('t', fontsize=20)
if country != 'Republic of Korea':
    plt.vlines(t_lockdown, 0, 1.05*deaths_target_date[:, -1].max(), linestyles = '--', color='r')
else:
    plt.vlines(params['t_tracing_start'], 0, 1.05*max((np.max(deaths), D[-1].sum())), linestyles = '--', color='r')

date_labels = []
for i in range(deaths_target_date.shape[1]):
    new_date = d0 + datetime.timedelta(days=i)
    date_labels.append('{}/{}'.format(new_date.month, new_date.day))

plt.tick_params(axis='both', which='major', labelsize=17)
currticks = plt.xticks()
currticks = [int(i) for i in currticks[0] if i >= 0 and i < deaths_target_date.shape[1]]
plt.xticks(currticks)
plt.xticks(currticks, [date_labels[i] for i in currticks], rotation=45)
plt.xlabel('Date', fontsize=23)
plt.tight_layout()
plt.savefig('img/heatmaps/trajectory_{}.pdf'.format(country))

