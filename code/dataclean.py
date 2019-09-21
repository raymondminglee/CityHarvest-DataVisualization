# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 15:28:12 2019

@author: ray80
"""

import pandas as pd
import numpy as np

idx = pd.IndexSlice

data = pd.read_excel('num_data.xlsx')
data = data.fillna(method='ffill')

x =data.loc[idx[:], idx[:, 'Estimate']]
x = x.iloc[0:3][:]
x = x.transpose()

#x = x.sort_values(by=['Total population'], axis = 0, ascending=False)
#pop_list = []
#for i in range(0,195,1):
#    pop_list.append(x.[i][0])
    


df = pd.read_excel('acs_select_econ_08to12_ntas_clean.xlsx', header=[0,1], index_col=None)
un_per = df.loc[idx['   Unemployment Rate'], idx[:, 'Percent']]

income_household = df.iloc[42:47]
income_household = income_household.loc[idx[:], idx[:, 'Estimate']]
income_household = income_household.transpose()
income_household['per'] = np.ones(195)

for i in range(0,195,1):
    income_household.per[i] = (income_household.iloc[i][1]+income_household.iloc[i][2])/income_household.iloc[i][0]


income_family = df.iloc[59:64]
income_family = income_family.loc[idx[:], idx[:, 'Estimate']]
income_family = income_family.transpose()

tabulated = pd.concat([x, income_household.per], axis=1)

tabulated.to_csv('data_model.csv')
un_per.to_csv('unemployment.csv')

data = pd.read_csv('data_model.csv', delimiter=',')
data2 = pd.read_csv('unemployment.csv', delimiter=',', header=None, names=['nta', 'p', 'un'])
data = pd.concat([data, data2.un], axis=1)
data['Unnamed: 0'] = data['Unnamed: 0'].apply(lambda line: line.split()[0])
meal = pd.read_excel('NTA_meals_served.xlsx', delimiter=',', index_col=0)
data = data.sort_values(by=['Unnamed: 0'])
meal = meal.sort_values(by=['NTA'])

data['served'] = np.zeros(195)
for i in range(0, 119, 1):
    print(meal.iloc[i][0])
    index = data.loc[data['Unnamed: 0'] == meal.iloc[i][0]].index[0]
    data.served[index] = meal.iloc[i][1]
    

    
data.to_csv('data_model_3.csv', index=False)
