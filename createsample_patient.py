#!/usr/bin/env python3.6
# coding: utf-8

## Generate test set of syntetic patients

# Import libraries
import numpy as np
import pandas as pd

# Number of samples
nsamples = 50
#Generate nsamples a and p values
const = []
a = np.random.uniform(low=0.51, high=1, size=nsamples) #'probability' #DA ARTICOLO 
p = np.random.uniform(low=0.2, high=1, size=nsamples) #proliferation rate #
#Generate nsamples sex values
sex = np.random.randint(low=0, high=2, size=nsamples)
#Generate nsamples weight values
mu = 70
sigma = 0.25
weight = np.random.normal(loc=mu, scale=sigma*mu, size=nsamples)
for i in range(len(weight)):
	weight[i] = np.round(weight[i],1)
#Generate nsamples age values
mu = 50
sigma = 0.2
age = np.random.normal(loc=mu, scale=sigma*mu, size=nsamples)
for i in range(len(age)):
	age[i] = int(age[i])
#Generate nsamples ec50 values
mu = 0.1234
sigma = 0.2 #PRIOR CON ARTICOLO DA TROVARE
ec50 = np.random.normal(loc=mu, scale=sigma*mu, size=nsamples)
#Generate nsamples number of stem cells corrupted values
mu = 1000000 #(lower bound 10^5, upper bound 10^6)
sigma = 0.47 #ARTICOLO DA TROVARE
l0 = np.random.normal(loc=mu, scale=sigma*mu, size=nsamples)

#Create pandas dataframe
data = {'Sex':sex, 'Age':age, 'Weight':weight, 'a':a, 'p':p, 'EC(50)':ec50, 'l0':l0}
syntetic_patients = pd.DataFrame(data)

#Export to csv and to txt
syntetic_patients.to_csv(r'datasets/syntetic_patients.csv', index=False)
np.savetxt(r'datasets/syntetic_patients.txt', syntetic_patients.values, fmt='%f')
