import numpy as np
import torch

#diabetes prevalence for china
#https://jamanetwork.com/journals/jama/fullarticle/1734701
male = np.array([5.2755905511811, 8.346456692913385, 13.543307086614174, 17.95275590551181, 20.708661417322837, 21.653543307086615])
female = np.array([4.015748031496061, 5.1181102362204705, 9.05511811023622, 17.401574803149607, 24.488188976377955, 25.196850393700785])
male = male/100
female = female/100

#https://www.statista.com/statistics/282119/china-sex-ratio-by-age-group/
sex_ratio = np.zeros(101)
sex_ratio[0:5] = 113.91
sex_ratio[5:10] = 118.03
sex_ratio[10:15] = 118.62
sex_ratio[15:20] = 118.14
sex_ratio[20:25] = 112.89
sex_ratio[25:30] = 105.39
sex_ratio[30:35] = 101.05
sex_ratio[35:40] = 102.84
sex_ratio[40:45] = 103.75
sex_ratio[45:50] = 103.64
sex_ratio[50:55] = 102.15
sex_ratio[55:60] = 101.65
sex_ratio[60:65] = 100.5
sex_ratio[65:70] = 96.94
sex_ratio[70:75] = 94.42
sex_ratio[75:80] = 89.15
sex_ratio[80:85] = 76.97
sex_ratio[85:90] = 71.16
sex_ratio[90:95] = 48.74
sex_ratio[95:] = 40.07

#calculate male to female ratio within each age bucket, and use this to combine the male vs female
#prevalence numbers
sex_ratio = sex_ratio/(sex_ratio + 100)

age_distribution = [16113.281,16543.361,16875.302,17118.429,17282.064,17375.527,17408.145,17389.238,17328.13,17234.143,17117.175,16987.122,16850.435,16715.289,16592.73,16484.473,16388.786,16370.261,16460.9,16637.439,16866.861,17182.465,17477.132,17702.896,17928.813,18144.994,18201.129,18841.832,20387.657,22413.391,24308.028,26355.485,27269.657,26400.295,24405.505,22597.72,20719.355,19296.916,18726.536,18750.928,18640.938,18451.511,18716.505,19599.644,20865.548,22101.75,23374.699,24376.638,24907.095,25077.435,25250.357,25414.362,25172.526,24383.003,23225.134,22043.117,20795.729,19608.86,18589.082,17703.703,16743.545,15666.543,14988.213,14917.427,15198.411,15425.124,15749.105,15550.741,14503.063,12921.733,11444.972,9939.85,8651.521,7764.623,7148.723,6478.704,5807.535,5222.027,4729.055,4307.295,3931.038,3608.42,3272.336,2887.659,2481.964,2118.152,1783.88,1480.587,1215.358,983.8,739.561,551.765,453.96,342.463,217.275,145.809,122.178,96.793,69.654,40.759,74.692]
age_distribution = np.array(age_distribution)
intervals = [(18, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 100)]

percent_male_intervals = np.zeros(len(intervals))
for i in range(len(intervals)):
    age_frequency_within_interval = age_distribution[intervals[i][0]:intervals[i][1]]/age_distribution[intervals[i][0]:intervals[i][1]].sum()
    percent_male_intervals[i] = np.dot(age_frequency_within_interval, sex_ratio[intervals[i][0]:intervals[i][1]])

p_diabetes = male*percent_male_intervals + female*(1-percent_male_intervals)

#split 70+ into 70-80 and 80-100, and assume the same distribution within each
intervals = [(18, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 100)]
p_diabetes_expanded = np.zeros(len(intervals))
p_diabetes_expanded[:-1] = p_diabetes
p_diabetes_expanded[-1] = p_diabetes_expanded[-2]
p_diabetes = p_diabetes_expanded


#hypertension by age for china
#https://www.ahajournals.org/doi/full/10.1161/CIRCULATIONAHA.117.032380
p_hyp_data = np.zeros(101)
p_hyp_data[:18] = 0
p_hyp_data[18:25] = 4.0
p_hyp_data[25:35] = 6.1
p_hyp_data[35:45] = 15.0
p_hyp_data[45:55] = 29.6
p_hyp_data[55:65] = 44.6
p_hyp_data[65:75] = 55.7
p_hyp_data[75:] = 60.2

p_hyp_data = p_hyp_data/100

#convert this into the same age intervals as the diabetes data by assuming constant
#rate within each bucket of the hypertension data
p_hyp = np.zeros(len(intervals))
for i in range(len(intervals)):
    age_frequency_within_interval = age_distribution[intervals[i][0]:intervals[i][1]]/age_distribution[intervals[i][0]:intervals[i][1]].sum()
    p_hyp[i] = np.dot(age_frequency_within_interval, p_hyp_data[intervals[i][0]:intervals[i][1]])

#age distribution of adult covid patients from china CDC
p_age = torch.zeros(len(intervals)).double()
p_age[0] = 3619
p_age[1] = 7600
p_age[2] = 8571
p_age[3] = 10008
p_age[4] = 8583
p_age[5] = 3918
p_age[6] = 1408
p_age = p_age/p_age.sum()

#probability of having hypertension for each age group
p_hyp = torch.tensor(p_hyp)
#probability of having diabetes for each age group
p_diabetes = torch.tensor(p_diabetes)


#mortality rate for hypertensive patients, diabetic patients, and for each age group
#http://weekly.chinacdc.cn/en/article/id/e53946e2-c6c4-41e9-9a9b-fea8db1a8f51
target_diabetes = torch.tensor(0.073).double()
target_hyp = torch.tensor(0.06).double()
#n_70_80 = 3918.
#n_over_80 = 1408.
target_age = torch.tensor([0.2, 0.2, 0.4, 1.3, 3.6, 8.0, 14.8]).double()
target_age = target_age/100



#https://www-nature-com.ezp-prod1.hul.harvard.edu/articles/hr201767
p_hyp_given_diabetes = 0.5

#calculate conditional probabilities of various events for each age group
p_hyp_given_not_diabetes = (p_hyp - p_hyp_given_diabetes*p_diabetes)/(1 - p_diabetes)
p_diabetes_total = (p_age*p_diabetes).sum()  #
p_hyp_total = (p_age*p_hyp).sum()
p_diabetes_given_hyp = p_hyp_given_diabetes*p_diabetes/p_hyp
#probability of being in each age group given a comorbidity
p_age_given_diabetes = p_diabetes*p_age/p_diabetes_total
p_age_given_hyp = p_hyp*p_age/p_hyp_total

def model_age(c_age, c_hyp, c_diabetes, c_intercept):
    return p_diabetes*p_hyp_given_diabetes*(torch.sigmoid(c_age + c_hyp + c_diabetes + c_intercept)) + \
           (1 - p_diabetes)*p_hyp_given_not_diabetes*(torch.sigmoid(c_age + c_hyp + c_intercept)) + \
           p_diabetes*(1 - p_hyp_given_diabetes)*(torch.sigmoid(c_age + c_diabetes + c_intercept)) + \
           (1 - p_diabetes) * (1 - p_hyp_given_not_diabetes)*torch.sigmoid(c_age + c_intercept)
    
def model_diabetes(c_age, c_hyp, c_diabetes, c_intercept):
    preds_by_age = p_age_given_diabetes*p_hyp_given_diabetes*torch.sigmoid(c_age + c_hyp + c_diabetes + c_intercept) + \
                   p_age_given_diabetes*(1 - p_hyp_given_diabetes)*torch.sigmoid(c_age + c_diabetes + c_intercept)
    return preds_by_age.sum()

def model_hyp(c_age, c_hyp, c_diabetes, c_intercept):
    preds_by_age = p_age_given_hyp*p_diabetes_given_hyp*torch.sigmoid(c_age + c_hyp + c_diabetes + c_intercept) + \
                   p_age_given_hyp*(1 - p_diabetes_given_hyp)*torch.sigmoid(c_age + c_hyp + c_intercept)
    return preds_by_age.sum()

def loss(c_age, c_hyp, c_diabetes, c_intercept):
    preds_age = model_age(c_age, c_hyp, c_diabetes, c_intercept)
    preds_diabetes = model_diabetes(c_age, c_hyp, c_diabetes, c_intercept)
    preds_hyp = model_hyp(c_age, c_hyp, c_diabetes, c_intercept)
    return torch.nn.MSELoss()(preds_age, target_age) + torch.nn.MSELoss()(preds_diabetes, target_diabetes) + torch.nn.MSELoss()(preds_hyp, target_hyp)


num_restarts = 10
c_age_store = torch.zeros(num_restarts, len(intervals)).double()
c_diabetes_store = torch.zeros(num_restarts).double()
c_hyp_store = torch.zeros(num_restarts).double()

for restart in range(num_restarts):
    print(restart)
    num_iter = 10000
    c_age = torch.rand(len(intervals), requires_grad=True, dtype=torch.double)
    c_diabetes = torch.rand(1, requires_grad=True, dtype=torch.double)
    c_hyp = torch.rand(1, requires_grad=True, dtype=torch.double)
    c_intercept = torch.tensor(0., requires_grad=False, dtype=torch.double)
    optimizer = torch.optim.Adam([c_age, c_diabetes, c_hyp], lr=1e-1)
    for t in range(num_iter):
        loss_itr = loss(c_age, c_hyp, c_diabetes, c_intercept)
    #    print(loss_itr.item())
        optimizer.zero_grad()
        loss_itr.backward()
        optimizer.step()
    print(loss_itr)
    print(model_age(c_age, c_hyp, c_diabetes, c_intercept), model_diabetes(c_age, c_hyp, c_diabetes, c_intercept).item(), model_hyp(c_age, c_hyp, c_diabetes, c_intercept).item())
    c_age_store[restart] = c_age
    c_diabetes_store[restart] = c_diabetes
    c_hyp_store[restart] = c_hyp
#    print(c_age, c_diabetes.item(), c_hyp.item(), c_intercept.item())
np.savetxt('c_age.txt', c_age_store.detach().numpy(), delimiter = ',')
np.savetxt('c_diabetes.txt', c_diabetes_store.detach().numpy(), delimiter = ',')
np.savetxt('c_hypertension.txt', c_hyp_store.detach().numpy(), delimiter = ',')
np.savetxt('comorbidity_age_intervals.txt', intervals, delimiter = ',', fmt='%d')