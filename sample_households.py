import numpy as np
import csv

'''
Code for sampling the household and age structure of a population of n
agents.
'''


def get_age_distribution(country):
    age_distribution=[]
    with open('World_Age_2019.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if row[0]==country:
                for i in range(101):
                    age_distribution.append(float(row[i+1]))
                break
    return np.array(age_distribution)


def get_mother_birth_age_distribution(country):
    mother_birth_age_distribution=[]
    with open('AgeSpecificFertility.csv',encoding='latin-1') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if row[0]==country:
                #15-19	20-24	25-29	30-34	35-39	40-44	45-49
                for i in range(7):
                    mother_birth_age_distribution.append(float(row[i+1]))
                break
    return np.array(mother_birth_age_distribution)


    

def sample_households_china(n):
    max_household_size = 4

    households = np.zeros((n, max_household_size), dtype=np.int)
    households[:] = -1
    age = np.zeros(n, dtype=np.int)
    n_ages = 101
    #estimates for china from 2020
    #from https://population.un.org/wpp/Download/Standard/Interpolated/
    age_distribution = [16113.281,16543.361,16875.302,17118.429,17282.064,17375.527,17408.145,17389.238,17328.13,17234.143,17117.175,16987.122,16850.435,16715.289,16592.73,16484.473,16388.786,16370.261,16460.9,16637.439,16866.861,17182.465,17477.132,17702.896,17928.813,18144.994,18201.129,18841.832,20387.657,22413.391,24308.028,26355.485,27269.657,26400.295,24405.505,22597.72,20719.355,19296.916,18726.536,18750.928,18640.938,18451.511,18716.505,19599.644,20865.548,22101.75,23374.699,24376.638,24907.095,25077.435,25250.357,25414.362,25172.526,24383.003,23225.134,22043.117,20795.729,19608.86,18589.082,17703.703,16743.545,15666.543,14988.213,14917.427,15198.411,15425.124,15749.105,15550.741,14503.063,12921.733,11444.972,9939.85,8651.521,7764.623,7148.723,6478.704,5807.535,5222.027,4729.055,4307.295,3931.038,3608.42,3272.336,2887.659,2481.964,2118.152,1783.88,1480.587,1215.358,983.8,739.561,551.765,453.96,342.463,217.275,145.809,122.178,96.793,69.654,40.759,74.692]
    age_distribution = np.array(age_distribution)
    age_distribution = age_distribution/age_distribution.sum()
    
    #single person, couple only, parents and unmarried children, 3-generation
    #from https://link.springer.com/article/10.1186/s40711-015-0011-0/tables/2
    #(2010 census, urban populations)
    household_probs = np.array([0.1703, .2117, 0.3557, 0.1126])
    household_probs /= household_probs.sum()
    num_generated = 0
    while num_generated < n:
        if n - num_generated < 5:
            i = 0
        else:
            i = np.random.choice(household_probs.shape[0], p=household_probs)
        #single person household
        #sample from age distribution
        if i == 0:
            age[num_generated] = np.random.choice(n_ages, p=age_distribution)
            generated_this_step = 1
        #couple, sample from age distribution conditioned on age >= 22
        elif i == 1:
            renormalized = age_distribution[22:]
            renormalized = renormalized/renormalized.sum()
            age[num_generated] = np.random.choice(n_ages-22, p=renormalized) + 22
            age[num_generated+1] = np.random.choice(n_ages-22, p=renormalized) + 22
            generated_this_step = 2
        #some information about mother's age at birth of first child
        #https://link.springer.com/article/10.1007/s42379-019-00022-9
        elif i == 2:
            renormalized = age_distribution[:22]
            renormalized = renormalized/renormalized.sum()
            child_age = np.random.choice(22, p=renormalized)
            age[num_generated] = child_age
            #super rough approximation, women have child at a uniformly random age between 23 and 33
            renormalized = age_distribution[23:34]
            renormalized = renormalized/renormalized.sum()
            mother_age_at_birth = np.random.choice(11, p=renormalized) + 23
            mother_current_age = mother_age_at_birth + child_age
            age[num_generated + 1] = mother_current_age
            age[num_generated + 2] = mother_current_age
            generated_this_step = 3
        elif i == 3:
            #start by generating parents/unmarried child
            renormalized = age_distribution[:22]
            renormalized = renormalized/renormalized.sum()
            child_age = np.random.choice(22, p=renormalized)
            age[num_generated] = child_age
            #super rough approximation, women have child at a uniformly random age between 23 and 33
            renormalized = age_distribution[23:34]
            renormalized = renormalized/renormalized.sum()
            mother_age_at_birth = np.random.choice(11, p=renormalized) + 23
            mother_current_age = mother_age_at_birth + child_age
            age[num_generated + 1] = mother_current_age
            age[num_generated + 2] = mother_current_age
            #add grandparents
            renormalized = age_distribution[23:34]
            renormalized = renormalized/renormalized.sum()
            grandmother_age_at_birth = np.random.choice(11, p=renormalized) + 23
            grandmother_current_age = grandmother_age_at_birth + mother_current_age
            age[num_generated + 3] = grandmother_current_age
            age[num_generated + 4] = grandmother_current_age
            generated_this_step = 5
        
        #update list of household contacts
        for i in range(num_generated, num_generated+generated_this_step):
            curr_pos = 0
            for j in range(num_generated, num_generated+generated_this_step):
                if i != j:
                    households[i, curr_pos] = j
                    curr_pos += 1
        num_generated += generated_this_step
    return households, age

def sample_households_italy(n):
    max_household_size = 6
    
    households = np.zeros((n, max_household_size), dtype=np.int)
    households[:] = -1
    age = np.zeros(n, dtype=np.int)    
    n_ages = 101
        
    age_distribution = get_age_distribution("Italy")
    age_distribution = np.array(age_distribution)
    age_distribution = age_distribution/age_distribution.sum()
        
    # List of household types: single household, couple without children, single parent +1/2/3 children, couple +1/2/3 children,
    # family without a nucleus, nucleus with other persons, households with two or more nuclei (a and b)
    household_probs = np.array([0.308179, 0.191000, 0.0694283, 0.0273065, 0.00450268, 0.152655, 0.132429, 0.0340969, 
                       0.043821, 0.033, 0.0150])
    household_probs /= household_probs.sum()
    
    num_generated = 0
    
    # from fertility data
    mother_birth_age_distribution=get_mother_birth_age_distribution("Italy")    
    renormalized_mother = mother_birth_age_distribution/mother_birth_age_distribution.sum()
    renormalized_adult = age_distribution[18:]
    renormalized_adult = renormalized_adult/renormalized_adult.sum()
    # 18 considered as majority age, maybe should consider that children may still live with parents until 30 or so
    renormalized_child = age_distribution[:30]
    renormalized_child = renormalized_child/renormalized_child.sum()
    
    renormalized_adult_older = age_distribution[30:]
    renormalized_adult_older /= renormalized_adult_older.sum()
    # 60 considered as retirement threshold, maybe should be larger, but reasonable for first pass
    renormalized_grandparent = age_distribution[60:]
    renormalized_grandparent = renormalized_grandparent/renormalized_grandparent.sum()
    
    while num_generated < n:
        if n - num_generated < (max_household_size+1):
            i = 0
        else:
            i = np.random.choice(household_probs.shape[0], p=household_probs)
        #single person household
        #sample from age distribution
        if i == 0:
            age[num_generated] = np.random.choice(n_ages-30, p=renormalized_adult_older) + 30
            generated_this_step = 1
        # couple, sample from age distribution conditioned on age >= 18
        elif i == 1:  
            age_adult = np.random.choice(n_ages-30, p=renormalized_adult_older) + 30
            age[num_generated] = age_adult
            age[num_generated+1] = min(n_ages-1,age_adult+3) # man three years older on average
            generated_this_step = 2
        # single parent, 1 child
        elif i == 2:            
            child_age = np.random.choice(30, p=renormalized_child)
            age[num_generated] = child_age
            #super rough approximation
            mother_age_at_birth = (np.random.choice(7, p=renormalized_mother) + 3)*5+np.random.randint(5)
            mother_current_age = min(n_ages-1,mother_age_at_birth + child_age)
            age[num_generated + 1] = mother_current_age
            generated_this_step = 2
        # single parent, 2 children
        elif i == 3:
            for j in range(2):                
                child_age = np.random.choice(30, p=renormalized_child)
                age[num_generated+j] = child_age
            mother_age_at_birth = (np.random.choice(7, p=renormalized_mother) + 3)*5+np.random.randint(5)
            mother_current_age = min(n_ages-1,mother_age_at_birth + max(age[num_generated:(num_generated+2)]))
            age[num_generated + 2] = mother_current_age
            generated_this_step = 3
        # single parent, 3 children
        elif i == 4:
            for j in range(3):                
                child_age = np.random.choice(30, p=renormalized_child)
                age[num_generated+j] = child_age
            mother_age_at_birth = (np.random.choice(7, p=renormalized_mother) + 3)*5+np.random.randint(5)
            mother_current_age = min(n_ages-1,mother_age_at_birth + max(age[num_generated:(num_generated+3)]))
            age[num_generated + 3] = mother_current_age
            generated_this_step = 4
            
        # couple, 1 child
        elif i == 5: 
            child_age = np.random.choice(30, p=renormalized_child)
            age[num_generated] = child_age
            #super rough approximation
            mother_age_at_birth = (np.random.choice(7, p=renormalized_mother) + 3)*5+np.random.randint(5)
            mother_current_age = min(n_ages-1,mother_age_at_birth + child_age)
            age[num_generated + 1] = mother_current_age
            age[num_generated + 2] = min(n_ages-1,mother_current_age+3)
            generated_this_step = 3
        
        # couple, 2 children
        elif i == 6:
            for j in range(2):                
                child_age = np.random.choice(30, p=renormalized_child)
                age[num_generated+j] = child_age
            mother_age_at_birth = (np.random.choice(7, p=renormalized_mother) + 3)*5+np.random.randint(5)
            mother_current_age = min(n_ages-1,mother_age_at_birth + max(age[num_generated:(num_generated+2)]))
            age[num_generated + 2] = mother_current_age
            age[num_generated + 3] = min(n_ages-1,mother_current_age+3)
            generated_this_step = 4            
        
        # couple, 3 children
        elif i == 7:
            for j in range(3):                
                child_age = np.random.choice(30, p=renormalized_child)
                age[num_generated+j] = child_age
            mother_age_at_birth = (np.random.choice(7, p=renormalized_mother) + 3)*5+np.random.randint(5)
            mother_current_age = min(n_ages-1,mother_age_at_birth + max(age[num_generated:(num_generated+3)]))
            age[num_generated + 3] = mother_current_age
            age[num_generated + 4] = min(n_ages-1,mother_current_age+3)
            generated_this_step = 5
        
        # family without nucleus
        elif i == 8:
            age[num_generated] = np.random.choice(n_ages-30, p=renormalized_adult_older) + 30
            age[num_generated+1] = np.random.choice(n_ages-30, p=renormalized_adult_older) + 30
            generated_this_step = 2         
                
        # nucleus with other persons (couple, 2 children, adult >= 60)
        elif i == 9:
            for j in range(2):                
                child_age = np.random.choice(30, p=renormalized_child)
                age[num_generated+j] = child_age
            mother_age_at_birth = (np.random.choice(7, p=renormalized_mother) + 3)*5+np.random.randint(5)
            mother_current_age = min(n_ages-1,mother_age_at_birth + max(age[num_generated:(num_generated+2)]))
            age[num_generated + 2] = mother_current_age
            age[num_generated + 3] = min(n_ages-1,mother_current_age+3)
            age[num_generated + 4] = np.random.choice(n_ages-60, p=renormalized_grandparent) + 60
            generated_this_step = 5
            
        # households with 2 or more nuclei
        # a - couple with same age for mother/father sampled from > 18 + 2 children <= 18 + 2 grand-parents
        # b - couple with same age for mother/father sampled from > 18 + 2 children <= 18 + 2 children from other marriage <= 18
        # scenario b removed for now 
        
        elif i == 10:
            for j in range(2):                
                child_age = np.random.choice(30, p=renormalized_child)
                age[num_generated+j] = child_age
            mother_age_at_birth = (np.random.choice(7, p=renormalized_mother) + 3)*5+np.random.randint(5)
            mother_current_age = min(n_ages-1,mother_age_at_birth + max(age[num_generated:(num_generated+2)]))
            age[num_generated + 2] = mother_current_age
            age[num_generated + 3] = min(n_ages-1,mother_current_age+3)
            #grandparent_age =  np.random.choice(n_ages-60, p=renormalized_grandparent) + 60
            grandmother_age_at_birth = (np.random.choice(7, p=renormalized_mother) + 3)*5+np.random.randint(5)
            grandmother_current_age = min(n_ages-1,grandmother_age_at_birth + mother_current_age)
            #age[num_generated + 4] = grandparent_age
            #age[num_generated + 5] = grandparent_age+3   
            age[num_generated + 4] = grandmother_current_age
            age[num_generated + 5] = min(n_ages-1,grandmother_current_age+3)   
            generated_this_step = 6
            
        #elif i == 11:
            #for j in range(4):                
                #child_age = np.random.choice(30, p=renormalized_child)
                #age[num_generated+j] = child_age
            #mother_age_at_birth = (np.random.choice(7, p=renormalized_mother) + 3)*5+np.random.randint(5)
            #mother_current_age = min(n_ages-1,mother_age_at_birth + max(age[num_generated:(num_generated+4)]))
            #age[num_generated + 4] = mother_current_age
            #age[num_generated + 5] = min(n_ages-1,mother_current_age+3)          
            #generated_this_step = 6
            
        #update list of household contacts
        for i in range(num_generated, num_generated+generated_this_step):
            curr_pos = 0
            for j in range(num_generated, num_generated+generated_this_step):
                if i != j:
                    households[i, curr_pos] = j
                    curr_pos += 1
        num_generated += generated_this_step
        
    return households, age