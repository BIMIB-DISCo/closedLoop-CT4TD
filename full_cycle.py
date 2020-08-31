#!/usr/bin/env python3.6
# coding: utf-8

################################### Imports ####################################
# Import libraries
import numpy as np
import pandas as pd
import math
import random #elimina
from casadi import *
from scipy import stats
# Import dataset
patients = pd.read_csv('datasets/syntetic_patients.csv')
################################################################################



################################## Functions ###################################

# Compute personal data (cl, v, ka) --------------------------------------------
def compute_parameters(sex, age, weight, phi):
	# compute weight and age factor to compute personal CL and V
	weight_factor = (weight - bw_mean) / bw_mean
	age_factor = (age - age_mean) / age_mean
	if(sex == 1):
		# Male
		cl = theta_a + theta_1*weight_factor + theta_2 + theta_3*age_factor + 1
		v = theta_b + theta_4
	else:
		#Female
		cl = theta_a + theta_1*weight_factor - theta_2 + theta_3*age_factor + 1
		v = theta_b - theta_4
	w1 = phi
	w2 = 1
	return cl, v, w1, w2
# ------------------------------------------------------------------------------

# Heaviside step function ------------------------------------------------------
def theta(tref, time):
	return (1 * (time-tref >= 0))
# ------------------------------------------------------------------------------

# Formulation of personalized optimal dosage problem ---------------------------
def personalized_problem_formulation(clearance, volume, a, p, ec50, weight1, weight2):
	# Model equations (ODE)
	xdot_summatory = 0
	for i in range(n_doses):
		xdot_summatory = xdot_summatory + (theta(i*intdoses,t)*const_f*d[i]*exp((-ka)*(t-i*intdoses)))
	xdot = ka * xdot_summatory - (clearance/volume) * xb
	# Objective term (Lagrange and Mayer terms)
	Lagrange = weight1*((math.log10(math.exp(1))*(((2*a-1)*p) - k_param*(emax*(xb/volume))/(ec50 + xb/volume)))) + weight2*(xb/volume)
	# Formulate discrete time dynamics
	DT = T/K/M
	f = Function('f', [xb, d, t], [xdot, Lagrange]) #f: x,u,t --> xdot,L
	X0 = MX.sym('X0')
	U = MX.sym('U', n_doses) #u=[u1,u2,...,n_doses]
	X = X0 #X new state (starts as X0)
	Q = 0 #new control
	for j in range(M): 
		k1, k1_q = f(X, U, t)
		k2, k2_q = f(X + DT/2 * k1, U, t)
		k3, k3_q = f(X + DT/2 * k2, U, t)
		k4, k4_q = f(X + DT * k3, U, t)
		X=X+DT/6*(k1 +2*k2 +2*k3 +k4)
		Q = Q + DT/6*(k1_q + 2*k2_q + 2*k3_q + k4_q)
	# Inputs [x_0,U,t] and outputs [X,Q] computed through Runge-Kutta
	F = Function('F', [X0, U, t], [X, Q],['x0','p', 't'],['xf','qf'])
	return F
# ------------------------------------------------------------------------------

# DMS solver -------------------------------------------------------------------
def dms_solve(evolutionFunction, volume, initial_guess = 0):
	initial_guess = volume*initial_guess
	# Initialize an empty NLP
	w=[] #vector of controls and state over time (u1,x1,u2,x2,...)
	w0 = [] #initial vector of w
	lbw = [] #lower bound for w
	ubw = [] #upper bound for w
	J = 0 #propagation of L
	g=[] #function on w (evolution over time)
	lbg = [] #lower bounds on g
	ubg = [] #upper bounds on g
	# "Lift" initial conditions for multiple shooting
	# State Xk
	Xk = MX.sym('X0') #k=0 (initial point)
	w += [Xk]
	lbw += [initial_guess]
	ubw += [initial_guess]
	w0 += [initial_guess]
	# Formulate the NLP problem
	for k in range(K):
		# Obtain time equivalent to k-interval
		t = (T/K)*k
		# New NLP variable for the control
		Uk = MX.sym('U_' + str(k), n_doses)
		w += [Uk]
		for i in range(n_doses):
			w0 += [0]
			lbw += [0]
			if(math.ceil(t) == i*intdoses or math.floor(t) == i*intdoses):
				ubw += [1000] #constraint control to be under lethal dosage
			else:
				ubw += [0] #constraint control to be 0 out of dosage time
		# Integrate till the end of the interval
		Fk = evolutionFunction(x0=Xk, p=Uk, t=t)
		Xk_end = Fk['xf']
		J = J + Fk['qf'] #Add Lagrange term on evolution of cost J
		# New NLP variable for state at end of interval
		Xk = MX.sym('X_'+str(k+1))
		w += [Xk]
		w0 += [0]
		lbw += [0]
		ubw += [inf]
		# Add equality constraint
		g += [Xk_end-Xk] #only of the interval k
		lbg += [0]
		ubg += [0]
	# Solve the NLP problem
	# Map NLP problem to ipopt NLP problem
	prob = {'f': J, 'x': vertcat(*w), 'g': vertcat(*g)}
	# Define a solver
	solver = nlpsol('solver', 'ipopt', prob, {'ipopt':{'print_frequency_time':60.0}});
	# Solve the NLP
	sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
	w_opt = sol['x'].full().flatten()
	# Get results
	# State Xb values
	state = w_opt[0::(n_doses+1)]/volume
	# Controls d values
	d_opt = []
	for i in range(n_doses):
		di_opt = w_opt[(i+1)::(n_doses+1)] #start from i+1 and select a value every numberOfDoses+1
		d_opt += [di_opt]
	# Summarize all dosage control variables in one single dosage series
	dosage = []
	for i in range(len(d_opt[0])):
		maxval = 0
		for j in range(len(d_opt)):
			if(d_opt[j][i] > maxval):
				maxval = d_opt[j][i]
		dosage = np.append(dosage, maxval)
	return state, dosage
# ------------------------------------------------------------------------------

# Compute observed net decay ---------------------------------------------------
def compute_decay(concentration, ec50, diseased_cells_number, a, p, delta_t):
	# Add normal noise to parameter to generate a measure
	a = np.random.normal(loc=a, scale=0.05*a, size=1)
	if(a<0.51):
		a = 0.51
	p = np.random.normal(loc=p, scale=0.05*p, size=1)
	lambda_parameter = (2*a-1)*p
	d_parameter = k_param*emax / (ec50/concentration + 1)
	decay = lambda_parameter - d_parameter
	new_diseased_cells_number = diseased_cells_number * (math.exp(decay*delta_t))
	return new_diseased_cells_number, decay
# ------------------------------------------------------------------------------

# Define the rule for a bayesian update steps ----------------------------------
def bayesian_update(prior_mean, prior_sigma, measure_mean, measure_sigma, n_measures):
	# Update step 
	updated_mean = ((measure_sigma**2)*prior_mean + n_measures*measure_mean*(prior_sigma**2)) / (n_measures*(prior_sigma**2) + (measure_sigma**2))
	updated_sigma = (measure_sigma**2)*(prior_sigma**2) / (n_measures*(prior_sigma**2) + (measure_sigma**2))
	updated_sigma = np.sqrt(updated_sigma)
	# Return updated measures
	return updated_mean, updated_sigma
# ------------------------------------------------------------------------------

################################################################################



################################## Constants ###################################
# fixed parameters from literature
ka = 0.437
theta_a = 12.8
theta_b = 258
theta_1 = 12.7
theta_2 = 0.8
theta_3 = -2.1
theta_4 = 61.0
bw_mean = 70
age_mean = 50
const_f = 1
ctg = 1
emax = 1.0
k_param = 0.377
T = 336. # 14 days
M = 4
K = 1000
n_doses = 14 
intdoses = 24.
nmeasures = 29 # 29 + ground truth (database) = 30 measures
# Model variables
t = MX.sym('t')
xb = MX.sym('xb')
d = MX.sym('d1')
for i in range(1,n_doses):
	di = MX.sym('d'+str(i+1))
	d = vertcat(d, di)
################################################################################



################################## Main Loop ###################################
# Main cycle
phi = [60, 65, 70, 75, 80]
for pi in range(len(phi)):	 
	for i in range(len(patients)):
		
		### Patient i ###
		
		# Read patients data
		sex = patients.loc[i,'Sex']
		age = patients.loc[i,'Age']
		weight = patients.loc[i,'Weight']
		a = patients.loc[i,'a']
		p = patients.loc[i,'p']
		ec50 = 0.1234 #prior 
		l_0 = patients.loc[i,'l0']

		# Compute personal parameters of the patient i 
		cl, v, w1, w2 = compute_parameters(sex=sex,
			                               age=age,
			                               weight=weight,
			                               phi=phi[pi])
		# Computer personalized optimal control problem
		function = personalized_problem_formulation(clearance=cl,
			                                        volume=v, 
			                                        a = a,
			                                        p = p, 
			                                        ec50=0.1234, #prior 
			                                        weight1=w1, 
			                                        weight2=w2)
		# Optimization: compute default mean concentration
		# Compute personal dosage through optimization
		concentration, dosage = dms_solve(evolutionFunction=function,
			                              volume=v,
			                              initial_guess=0)
		# Compute mean concentration (excluding firsts 4 days to stabilize it)
		mean_concentration = np.mean(concentration[int(4*24*(K/T)):])
		last_concentration = concentration[len(concentration)-1]
		# First measure: get ec50 from syntetic database
		ec50_truth = patients.loc[i,'EC(50)']
		
		# Create patient data arrays
		l_patient = [l_0]
		l_measures = [l_0]
		ec50_tilde_patient = [ec50_truth]
		ec50_updated = ec50_truth # Until the first bayesian update
		decay_patient = [(((2*a-1)*p) - k_param*emax / (ec50_truth/mean_concentration + 1))]
		cmean_patient = [mean_concentration]

		# Measure-Analysis-Update-Optimization cycle
		for j in range(nmeasures):
			# If number of CSC > 1 (elsewhere no more cancer cells)
			if(l_0>1):
				# Generate tumor evolution measure and compute observed net decay 
				l_1, decay = compute_decay(concentration = mean_concentration,
							               ec50 = ec50_truth,
							               diseased_cells_number = l_0,
							               a = a,
							               p = p,
							               delta_t = T/24) #Time between different time points
				l_patient += [l_1]
				l_measures += [l_1]
				decay_patient += [decay]
				
				# Do fitting-update-optimization only if we have 3 times point
				if(len(l_measures) == 3):
									
					# Data analysis: interpolate l_0 and l_i
					times = [(j-2)*30, (j-1)*30, j*30] #times in time point
					datas = [math.log10(l_measures[0]), math.log10(l_measures[1]), math.log10(l_measures[2])]
					slope, intercept, r_sqrd, p_value, std_err = stats.linregress(times, datas)
					lambda_patient_tilde = slope / np.log10(math.exp(1))
					d_patient_tilde = (2*a-1)*p - lambda_patient_tilde
					ec50_tilde = mean_concentration * ((k_param*emax / d_patient_tilde) - 1)
					# Bayesian update of ec50
					ec50_updated, _ = bayesian_update(prior_mean=0.1234, 
							                          prior_sigma=0.2*0.1234, 
							                          measure_mean=np.mean(np.append(ec50_tilde_patient, ec50_tilde)),
							                          measure_sigma=np.std(np.append(ec50_tilde_patient, ec50_tilde)),
							                          n_measures=len(np.append(ec50_tilde_patient, ec50_tilde)))
					ec50_tilde_patient += [ec50_updated]
					
					# Computer personalized optimal control problem
					function = personalized_problem_formulation(clearance=cl,
									                            volume=v,
									                            a=a,
									                            p=p, 
									                            ec50=ec50_updated, 
									                            weight1=w1, 
									                            weight2=w2)
					# Compute personal dosage through optimization
					concentration, dosage = dms_solve(evolutionFunction=function,
									                  volume=v,
									                  initial_guess=last_concentration)
					# Compute mean concentration (excluding firsts 4 days to stabilize it)
					mean_concentration = np.mean(concentration[int(4*24*(K/T)):])
					last_concentration = concentration[len(concentration)-1]
					cmean_patient += [mean_concentration]
					# Compute 14 dosage
					dosestime = []
					doses = []
					start = 0
					summ = 0
					count = 0
					for c in range(len(dosage)):
						if(dosage[c] > 1 and start == 0):
							start = 1
							summ = summ + dosage[c]
							count = count + 1 
						elif(dosage[c] > 1 and start != 0):
							summ = summ + dosage[c]
							count = count + 1
						elif(start != 0):
							#end of mean interval
							start = 0
							dosage[c-count:c] = 0
							dosage[c-1] = summ/count #mean
							dosestime += [math.floor((c-1)*(T/K))] 
							doses += [np.round(dosage[c-1])]
							count = 0
							summ = 0
					# Export doses table
					dos_data = {'Time (hours)':dosestime, 'Dosage (mg)':doses}
					dos_dataset = pd.DataFrame(dos_data)
					dos_dataset.to_csv(r'datasets/simulation/'+str(phi[pi])+'/patient_'+str(i)+'_measure_'+str(j+2)+'.csv',index=False)
					# OBS: #measure j+2 perchè la prima è quella da database e un altro in più perchè gli indici partono da 0
					# Reset measurements for the fitting (keep only last measure)
					l_measures = [l_1]
				else:
					# Keep mean concentration and ec50 
					ec50_tilde_patient += [ec50_updated]
					cmean_patient += [mean_concentration]
				
				# Update l number of diseased cells
				l_0 = l_1
			#else:
				# No more CSC cells. Lukemia eradicated
		
		# Export data
		data = {"L(i)_tilde":l_patient, "Decay":decay_patient, "Ec50_tilde":ec50_tilde_patient, "Mean concentration":cmean_patient}
		dataset = pd.DataFrame(data)
		dataset.to_csv(r'datasets/simulation/'+str(phi[pi])+'/patient_'+str(i)+'.csv',index=False)
	
################################################################################
