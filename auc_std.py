#!/usr/bin/env python3.6
# coding: utf-8

# Import libraries
import numpy as np
import pandas as pd
import math
import random #elimina
from casadi import *
from scipy import stats
from sklearn import metrics

# Import dataset
patients = pd.read_csv('datasets/syntetic_patients.csv')

def theta(tref, time):
	return (1 * (time-tref >= 0))

# fixed parameters from literature
ka = 0.437
const_f = 1
ctg = 1
T = 336. # 14 days
M = 4
K = 1000
n_doses = 14 
intdoses = 24.
dosage = 400
theta_a = 12.8
theta_b = 258
theta_1 = 12.7
theta_2 = 0.8
theta_3 = -2.1
theta_4 = 61.0
bw_mean = 70
age_mean = 50

auc_values = []
for i in range(len(patients)):
	# Read patients data
	sex = patients.loc[i,'Sex']
	age = patients.loc[i,'Age']
	weight = patients.loc[i,'Weight']
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
	# Model variables
	t = MX.sym('t')
	xb = MX.sym('xb')
	# Model equations (ODE)
	xdot_summatory = 0
	for i in range(n_doses):
		xdot_summatory = xdot_summatory + (theta(i*intdoses,t)*const_f*dosage*exp((-ka)*(t-i*intdoses)))
	xdot = ka * xdot_summatory - (cl/v) * xb
	# Objective term (Lagrange and Mayer terms)
	Lagrange = 0
	# Formulate discrete time dynamics
	DT = T/K/M
	f = Function('f', [xb, t], [xdot, Lagrange]) #f: x,u,t --> xdot,L
	X0 = MX.sym('X0')
	X = X0 #X new state (starts as X0)
	Q = 0 #new control
	for j in range(M): 
		k1, k1_q = f(X, t)
		k2, k2_q = f(X + DT/2 * k1, t)
		k3, k3_q = f(X + DT/2 * k2, t)
		k4, k4_q = f(X + DT * k3, t)
		X=X+DT/6*(k1 +2*k2 +2*k3 +k4)
		Q = Q + DT/6*(k1_q + 2*k2_q + 2*k3_q + k4_q)
	# Inputs [x_0,U,t] and outputs [X,Q] computed through Runge-Kutta
	F = Function('F', [X0, t], [X, Q],['x0', 't'],['xf','qf'])
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
	lbw += [0]
	ubw += [0]
	w0 += [0]
	# Formulate the NLP problem
	for k in range(K):
		# Obtain time equivalent to k-interval
		t = (T/K)*k
		# Integrate till the end of the interval
		Fk = F(x0=Xk, t=t)
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
	state = w_opt/v
	x_val = []
	y_val = []
	cmean = np.mean(state)
	for j in range(336):
		x_val += [j]
		y_val += [cmean]
	auc = metrics.auc(x_val, y_val)
	auc_values += [auc]
	
data = {'auc':auc_values}
dataset = pd.DataFrame(data)
dataset.to_csv(r'datasets/auc_std.csv', index=False)
