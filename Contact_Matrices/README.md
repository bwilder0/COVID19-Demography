For each country, you will find four different contact matrix files:
1. Home: For contacts within the household;
2. School: For school contacts, prior to t=school closure time for considered country;
3. Work: For contacts at the workplace, prior to t=telework deployment time for considered country;
4. Other: For other places.

The indicated value is the mean of the Poisson distribution for the number of contacts the infected individual has 
with someone in the age category of the contact person, if met at location type 1 (home), 2 (school), 3 (work) or 4 (other).

The following dictionary has the age groups corresponding with both the infected individual and the contact person (5-year age categories).
contact_matrix_age_groups_dict = {'infected_1': '0-4', 'contact_1': '0-4', 
                                'infected_2': '5-9', 'contact_2': '5-9', 
                                'infected_3': '10-14', 'contact_3': '10-14',
                                'infected_4': '15-19', 'contact_4': '15-19',
                                'infected_5': '20-24', 'contact_5': '20-24',
                                'infected_6': '25-29', 'contact_6': '25-29',
                                'infected_7': '30-34', 'contact_7': '30-34',
                                'infected_8': '35-39', 'contact_8': '35-39',
                                'infected_9': '40-44', 'contact_9': '40-44',
                                'infected_10': '45-49', 'contact_10': '45-49',
                                'infected_11': '50-54', 'contact_11': '50-54',
                                'infected_12': '55-59', 'contact_12': '55-59',
                                'infected_13': '60-64', 'contact_13': '60-64',
                                'infected_14': '65-69', 'contact_14': '65-69',
                                'infected_15': '70-74', 'contact_15': '70-74',
                                'infected_16': '75-79', 'contact_16': '75-79'}

Taken from:
Kiesha Prem, Alex Cook, and Mark Jit. Projecting social contact matrices in 152 countries using contact surveysand demographic data.PLoS Computational Biology, 13(9):e1005697, 2017.