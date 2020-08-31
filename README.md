# closedLoop-CT4TD
This repository contains the source code to replicate the case study presented in *A Closed-loop Optimization Framework for Personalized Cancer Therapy Design*

## Requirements
The source code is written in ```Python 3.6``` and it requires the following packages:
*`CASADI` (<https://web.casadi.org/docs/#document-install>)
*`numpy`
*`pandas`
*`matplotlib`
*`sklearn`
*`scipy`

## Reproducibility
To replicate the case study presented in the article, execute ```full_cycle.py```. At the end of its computation, you will find, for each value of \phi, a folder that contains the corresponding results for each patient (i.e. for \phi=50 and patient=*patient_1* the folder is *datasets/simulations/50/patient_1.csv*). 
In each result file you will find the following columns:
1. Number of measured CSCs.
1. Estimated decay (net growth rate) of CSCs (1/day).
1. Estimated value of EC<sub>50</sub> (mg/L).
1. Mean concentration of Imatinib in the blood (mg/L).
1. The AUC of the time point under study (mg*hours/L).

## New case study
If you wish to generate a new dataset, you need to run `createsample_patient.py` and then, you can run `full_cycle.py` to obtain the results. 
When you run `createsample_patient.py`, you will create a new file `synthetic_patients.csv`, which contains a line for each patient with the following columns:
1. Age (years).
1. Body weight (kg).
1. Ground truth EC<sub>50</sub> (mg/L).
1. Tumor parameter a<sub>1</sub>.
1. Tumor parameter p<sub>1</sub> (1/days).
1. Initial number of CSCs L<sub>1</sub>(0).

## Analysis of standard dosage
Finally, if you wish to simulate standard Imatinib dosage for each patient, you need to run `auc_std.py`.
