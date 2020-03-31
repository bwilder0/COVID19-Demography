import numpy as np

def p_diabetes_china(age):
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

    p_diabetes_expanded = np.zeros(101)
    p_diabetes_expanded[:intervals[0][0]] = 0
    for i in range(len(intervals)):
        p_diabetes_expanded[intervals[i][0]:intervals[i][1]] = p_diabetes[i]
    p_diabetes_expanded[intervals[-1][1]:] = p_diabetes[-1]

    return p_diabetes_expanded

def p_hypertension_china(age):
    #https://www.ncbi.nlm.nih.gov/pubmed/29449338
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

    return p_hyp_data


def sample_joint(age, p_diabetes, p_hyp):
    #https://www-nature-com.ezp-prod1.hul.harvard.edu/articles/hr201767
    p_hyp_given_diabetes = 0.5
    p_hyp_given_not_diabetes = (p_hyp - p_hyp_given_diabetes*p_diabetes)/(1 - p_diabetes)
    diabetes_status = (np.random.rand(age.shape[0]) < p_diabetes[age]).astype(np.int)
    hyp_status = np.zeros(age.shape[0], dtype=np.int)
    hyp_status[diabetes_status == 1] = np.random.rand((diabetes_status == 1).sum()) < p_hyp_given_diabetes
    hyp_status[diabetes_status == 0] = np.random.rand((diabetes_status == 0).sum()) < p_hyp_given_not_diabetes[age[diabetes_status == 0]]
    return diabetes_status, hyp_status

def sample_joint_comorbidities(age, country='China'):
    """
    Default country is China.
    For other countries pass value for country from {us, Republic of Korea, japan, Spain, italy, uk, France}
    """

    return sample_joint(age, p_comorbidity(country, 'diabetes'), p_comorbidity(country, 'hypertension'))

def p_comorbidity(country, comorbidity, warning=False):

    """
    Input:
        -country: a string input belonging to- {us, Republic of Korea, japan, Spain, italy, uk, France}
        -comorbidity: a string input belonging to- {diabetes, hypertension}
        -warning: optional, If set to True, prints out the underlying assumptions/approximations
    Returns:
        -prevalence, sampled from a prevalence array of size 100, where prevalence[i] is the prevalence rate at age between {i, i+1}
    """

    prevalence = np.zeros(101)
    warning_string= " "


    ######################################  China #############################
    if country=='China':

        dummy_value_to_work_with_china_function=0
        if comorbidity=='diabetes':
            prevalence = p_diabetes_china(dummy_value_to_work_with_china_function)
        elif comorbidity=='hypertension':
            prevalence= p_hypertension_china(dummy_value_to_work_with_china_function)

    ######################################  Italy #############################
    if country=='Italy':

        if comorbidity=='diabetes':
            #from Global Burden of Disease study
            for i in range(101):
                if i <= 4:
                    prevalence[i] = 0.0001
                elif i <= 9:
                    prevalence[i] = 0.0009
                elif i <= 14:
                    prevalence[i] = 0.0024
                elif i <= 19:
                    prevalence[i] = 0.0091
                elif i <= 24:
                    prevalence[i] = 0.0264
                elif i <= 29:
                    prevalence[i] = 0.0356
                elif i <= 34:
                    prevalence[i] = 0.0392
                elif i <= 39:
                    prevalence[i] = 0.0428
                elif i <= 44:
                    prevalence[i] = 0.0489
                elif i <= 49:
                    prevalence[i] = 0.0638
                elif i <= 54:
                    prevalence[i] = 0.0893
                elif i <= 59:
                    prevalence[i] = 0.1277
                elif i <= 64:
                    prevalence[i] = 0.1783
                elif i <= 69:
                    prevalence[i] = 0.2106
                elif i <= 74:
                    prevalence[i] = 0.2407
                elif i <= 79:
                    prevalence[i] = 0.2851
                elif i <= 84:
                    prevalence[i] = 0.3348
                elif i <= 89:
                    prevalence[i] = 0.3517
                else:
                    prevalence[i] = 0.3354

        elif comorbidity=='hypertension':
            #https://www.ncbi.nlm.nih.gov/pubmed/28487768
            for i in range(101):
                if i<35:
                    prevalence[i]= 0.14*(i/35.)
                elif i<39:
                    prevalence[i]=0.14
                elif i<44:
                    prevalence[i]=0.1
                elif i<49:
                    prevalence[i]=0.16
                elif i<54:
                    prevalence[i]=0.3
                else:
                    prevalence[i]=0.34

    if warning:
        print ("Warning: \n", warning_string)

    return prevalence
