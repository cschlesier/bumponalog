#!/usr/bin/env python

__doc__ = '''Fit data (art files, or dst skim or histogram pickle files).

Accepts multiple files, but only of a single type per invocation.

Creates a directory ./save_fits/ and writes out some results.  This script
REFUSES to overwrite the save_fits directory, so either move it or delete
it with rm -rf.

- - - -
Bare fitting code--requires you to provide optimal model and initial values for fitting the particular dataset.

Produces a suite of plots:
	Diagnostic Plots:
		-t-method histogram
		-4 component histograms
		-lost muon spectrum
		-pileup spectrum
		-unfitted ratio histogram
	Analysis Plots:
		-ratio model with initial values
		-ratio fit result
		-ratio modulo 100 us
		-residuals
		-fft residuals
		-cbo amplitude vs time after injection
	Scans: 
		-loop over calo #
		-loop over start time
		-loop over end time
		-loop over energy threshold

'''

binwidth = 0.14915 # ns
make_diagnostic_plots = False
extrafit = False

#### load some things ####

import matplotlib
import matplotlib.pyplot as plt
from pylab import frange
import argparse

# system imports
import os,sys
import pickle
import math
import random
import ROOT as Root
import numpy as np
import scipy
import scipy.stats as sstats
import ctypes as ct

# our custom stuff
import Blinders
import fitresults
import utility as util

folder = 'fit_plots'
os.mkdir(folder)

def save_figure(fig,basename,extension='png'):
  if not type(fig)==matplotlib.figure.Figure:
	try:
	  fig = fig.figure
	  assert type(fig)==matplotlib.figure.Figure
	except: RuntimeError('save_figure() requires a matplotlib.figure.Figure (or at least something that has a data member ".figure" which references a matplotlib Figure)!')
  filename = os.path.join(folder,basename+'.'+extension)
  print 'writing out %s...'%filename
  fig.savefig(filename, bbox_inches='tight')


def scanplots(xaxis, chi2, chi2_er, pvalue, asymmetry, asymmetry_er, omega_aR, omega_aR_er, phi_a, phi_a_er, tau_mu, tau_mu_er, a_cbo, a_cbo_er, tau_cbo, tau_cbo_er, omega_cbo, omega_cbo_er, phi_cbo, phi_cbo_er, K, K_er, scanname):
	# Create plots for various scans
	#
	if scanname == 'start' or 'end':
		unit = 't [us]' # x-axis label
	if scanname == 'calo':
		unit = 'calo number'
	if scanname == 'energythreshold':
		unit = 'MeV'

	chi2_plot = plt.figure(figsize=(10,6), dpi=600).add_subplot(1,1,1,title='chi2/ndf',xlabel=str(unit), ylabel='')
	chi2_plot.errorbar(xaxis,chi2ndf_list, chi2_er, fmt='o')
	plt.axhline(y=1.0, color='red', linestyle='-.')
	save_figure(chi2_plot,'chi2_plot_'+str(scanname))

	p_plot = plt.figure(figsize=(10,6), dpi=600).add_subplot(1,1,1,title='p-value',xlabel=str(unit), ylabel='')
	p_plot.plot(xaxis,pvalue,'bo',label='p',color='blue')
	save_figure(p_plot,'p_plot_'+str(scanname))

	a_plot = plt.figure(figsize=(10,6), dpi=800).add_subplot(1,1,1,title='A',xlabel=str(unit), ylabel='')
	a_plot.errorbar(xaxis,asymmetry, asymmetry_er, fmt='o')
	if scanname == 'start' or 'end' or 'energythreshold':
		asigdiff = np.array([np.sqrt(asymmetry_er[i]**2 - asymmetry_er[0]**2) for i in range(len(asymmetry_er))])
		a_plot.fill_between(xaxis, asymmetry[0]-asigdiff, asymmetry[0]+asigdiff, color='b', alpha=.1)
	save_figure(a_plot,'a_plot_'+str(scanname))

	r_plot = plt.figure(figsize=(10,6), dpi=800).add_subplot(1,1,1,title='R',xlabel=str(unit), ylabel='ppm')
	r_plot.errorbar(xaxis,omega_aR, omega_aR_er, fmt='o')
	if scanname == 'start' or 'end' or 'energythreshold':
		rsigdiff = np.array([np.sqrt(omega_aR_er[i]**2 - omega_aR_er[0]**2) for i in range(len(omega_aR_er))])
		r_plot.fill_between(xaxis, omega_aR[0]-rsigdiff, omega_aR[0]+rsigdiff, color='b', alpha=.1)
	save_figure(r_plot,'r_plot_'+str(scanname))

	phi_plot = plt.figure(figsize=(10,6), dpi=800).add_subplot(1,1,1,title='phi_0',xlabel=str(unit), ylabel='rad')
	phi_plot.errorbar(xaxis,phi_a, phi_a_er, fmt='o')
	if scanname == 'start' or 'end' or 'energythreshold':
		phisigdiff = np.array([np.sqrt(phi_a_er[i]**2 - phi_a_er[0]**2) for i in range(len(phi_a_er))])
		phi_plot.fill_between(xaxis, phi_a[0]-phisigdiff, phi_a[0]+phisigdiff, color='b', alpha=.1)
	save_figure(phi_plot,'phi_plot_'+str(scanname))

	tau_mu_plot = plt.figure(figsize=(10,6), dpi=800).add_subplot(1,1,1,title='muon lifetime',xlabel=str(unit), ylabel='us')
	tau_mu_plot.errorbar(xaxis,tau_mu, tau_mu_er, fmt='o')
	if scanname == 'start' or 'end' or 'energythreshold':
		tausigdiff = np.array([np.sqrt(tau_mu_er[i]**2 - tau_mu_er[0]**2) for i in range(len(tau_mu_er))])
		tau_mu_plot.fill_between(xaxis, tau_mu[0]-tausigdiff, tau_mu[0]+tausigdiff, color='b', alpha=.1)
	save_figure(tau_mu_plot,'tau_mu_plot_'+str(scanname))

	acbo_plot = plt.figure(figsize=(10,6), dpi=800).add_subplot(1,1,1,title='A_cbo',xlabel=str(unit), ylabel='')
	acbo_plot.errorbar(xaxis,a_cbo, a_cbo_er, fmt='o')
	if scanname == 'start' or 'end' or 'energythreshold':
		acsigdiff = np.array([np.sqrt(a_cbo_er[i]**2 - a_cbo_er[0]**2) for i in range(len(a_cbo_er))])
		acbo_plot.fill_between(xaxis, a_cbo[0]-acsigdiff, a_cbo[0]+acsigdiff, color='b', alpha=.1)
	save_figure(acbo_plot,'acbo_plot_'+str(scanname))

	tcbo_plot = plt.figure(figsize=(10,6), dpi=800).add_subplot(1,1,1,title='tau_cbo',xlabel=str(unit), ylabel='us')
	tcbo_plot.errorbar(xaxis,tau_cbo, tau_cbo_er, fmt='o')
	if scanname == 'start' or 'end' or 'energythreshold':
		tcsigdiff = np.array([np.sqrt(tau_cbo_er[i]**2 - tau_cbo_er[0]**2) for i in range(len(tau_cbo_er))])
		tcbo_plot.fill_between(xaxis, tau_cbo[0]-tcsigdiff, tau_cbo[0]+tcsigdiff, color='b', alpha=.1)
	save_figure(tcbo_plot,'tcbo_plot_'+str(scanname))

	omegacbo_plot = plt.figure(figsize=(10,6), dpi=800).add_subplot(1,1,1,title='omega_cbo',xlabel=str(unit), ylabel='radMHz')
	omegacbo_plot.errorbar(xaxis,omega_cbo, omega_cbo_er, fmt='o')
	if scanname == 'start' or 'end' or 'energythreshold':
		ocsigdiff = np.array([np.sqrt(omega_cbo_er[i]**2 - omega_cbo_er[0]**2) for i in range(len(omega_cbo_er))])
		omegacbo_plot.fill_between(xaxis, omega_cbo[0]-ocsigdiff, omega_cbo[0]+ocsigdiff, color='b', alpha=.1)
	save_figure(omegacbo_plot,'omegacbo_plot_'+str(scanname))

	phicbo_plot = plt.figure(figsize=(10,6), dpi=800).add_subplot(1,1,1,title='phi_cbo',xlabel=str(unit), ylabel='')
	phicbo_plot.errorbar(xaxis,phi_cbo, phi_cbo_er, fmt='o')
	if scanname == 'start' or 'end' or 'energythreshold':
		pcsigdiff = np.array([np.sqrt(phi_cbo_er[i]**2 - phi_cbo_er[0]**2) for i in range(len(phi_cbo_er))])
		phicbo_plot.fill_between(xaxis, phi_cbo[0]-pcsigdiff, phi_cbo[0]+pcsigdiff, color='b', alpha=.1)
	save_figure(phicbo_plot,'phicbo_plot_'+str(scanname))

	K_plot = plt.figure(figsize=(10,6), dpi=800).add_subplot(1,1,1,title='K',xlabel=str(unit), ylabel='')
	K_plot.errorbar(xaxis,K, K_er, fmt='o')
	if scanname == 'start' or 'end' or 'energythreshold':
		ksigdiff = np.array([np.sqrt(K_er[i]**2 - K_er[0]**2) for i in range(len(K_er))])
		K_plot.fill_between(xaxis, K[0]-ksigdiff, K[0]+ksigdiff, color='b', alpha=.1)
	save_figure(K_plot,'K_plot_'+str(scanname))
	


#Command-line options  
parser = argparse.ArgumentParser(description='Provide name of dataset so the fitter knows which model and initial conditions to use.')
parser.add_argument('infile', type=str, nargs='+', help='Input data file') #single infile is good for calo, start, & end scans. Energy threshold requires a list of files.
parser.add_argument('dataset',type=str, default='endgame',help='Data set name--choose ONE from options: endgame, 9day, 60h, 2c.')
parser.add_argument('scans',type=str, default='none',help='Select which scans you would like to run--choose ONE from options: none, calo, start, end, energythreshold')
args = parser.parse_args()


#Default setting is no scans
do_calo_scan = False
do_start_time_scan = False
do_end_time_scan = False
do_energy_threshold_scan = False

if args.scans == 'calo':
	do_calo_scan = True
if args.scans == 'start':
	do_start_time_scan = True
if args.scans == 'end':
	do_end_time_scan = True
if args.scans =='energythreshold':
	do_energy_threshold_scan = True 

#Inital histogramming/fitting routine before scans 
filename = args.infile[0]
a = Root.TFile.Open(filename) 
tmeth = a.Get("MuonCoincidence/TIME_FULL")
tmethD = a.Get("MuonCoincidence/TIME_D")
tmethS1 = a.Get("MuonCoincidence/TIME_S1")
tmethS2 = a.Get("MuonCoincidence/TIME_S2")
vp = a.Get("MuonCoincidence/caloTimes_Vp_all")
vm = a.Get("MuonCoincidence/caloTimes_Vm_all")
up = a.Get("MuonCoincidence/caloTimes_Up_all")
um = a.Get("MuonCoincidence/caloTimes_Um_all")
d_um = a.Get("MuonCoincidence/DTimes_Um_all")
s1_um = a.Get("MuonCoincidence/S1Times_Um_all")
s2_um = a.Get("MuonCoincidence/S2Times_Um_all")
d_up = a.Get("MuonCoincidence/DTimes_Up_all")
s1_up = a.Get("MuonCoincidence/S1Times_Up_all")
s2_up = a.Get("MuonCoincidence/S2Times_Up_all")
d_vm = a.Get("MuonCoincidence/DTimes_Vm_all")
s1_vm = a.Get("MuonCoincidence/S1Times_Vm_all")
s2_vm = a.Get("MuonCoincidence/S2Times_Vm_all")
d_vp = a.Get("MuonCoincidence/DTimes_Vp_all")
s1_vp = a.Get("MuonCoincidence/S1Times_Vp_all")
s2_vp = a.Get("MuonCoincidence/S2Times_Vp_all")

#TMethod Plot -- under/overflow bins are cut out
events = np.array(tmeth)[1:-1]-np.array(tmethD)[1:-1]+np.array(tmethS1)[1:-1]+np.array(tmethS2)[1:-1] # pileup subtraction
nbins = tmeth.GetNbinsX() 
bins = np.linspace(tmeth.GetXaxis().GetBinCenter(1), tmeth.GetXaxis().GetBinCenter(nbins), num=nbins)

#Subtract Pileup
Um = np.array(um)[1:-1]-np.array(d_um)[1:-1]+np.array(s1_um)[1:-1]+np.array(s2_um)[1:-1] 
Up = np.array(up)[1:-1]-np.array(d_up)[1:-1]+np.array(s1_up)[1:-1]+np.array(s2_up)[1:-1] 
Vm = np.array(vm)[1:-1]-np.array(d_vm)[1:-1]+np.array(s1_vm)[1:-1]+np.array(s2_vm)[1:-1] 
Vp = np.array(vp)[1:-1]-np.array(d_vp)[1:-1]+np.array(s1_vp)[1:-1]+np.array(s2_vp)[1:-1] 

#Compose 'graphs' of U, V, and their linear combinations 
U = Um + Up
V = Vm + Vp
S = U + V
D = U - V
Ratio = D/S
variance = (1-Ratio**2) / S #Eqn. 22 from E821 Note 366

S_peak = np.argmax(S) # Get bin# of peak of exponential decay function -- determines starting point for Ratio plot
print "Unclipped plot begins at t="+str(bins[S_peak]) #start times will differ among calo scans if S_peak changes from calo to calo

#Set Start and End times based on dataset
if args.dataset  == '2c':
	starttime = 32.0 # microseconds
	endtime = 600.0
if args.dataset  == '9d':
	starttime = 30.2 # microseconds
	endtime = 600.0

firstbin = len([i for i, val in enumerate(bins) if val < starttime])
if firstbin < S_peak: #make sure firstbin happens after S_peak, otherwise set S_peak to start time
	firstbin = S_peak
lastbin = next(x for x, val in enumerate(bins) if val > endtime)

print 'For dataset '+ args.dataset +', fitting will begin at t='+str(bins[firstbin])
print 'For dataset '+ args.dataset +', fitting will begin at t='+str(bins[lastbin])

#Clip messy early and late times from Ratio plot; update range on variance array (will have to use later in residuals function)
bins_for_fit = bins[firstbin:lastbin]
Ratio_for_fit = Ratio[firstbin:lastbin]
tmeth_for_fit = events[firstbin:lastbin]
variance = variance[firstbin:lastbin]

#Diagnostic Plots
if make_diagnostic_plots:
	zerocrossing_plot = plt.figure(figsize=(10,6), dpi=800).add_subplot(1,1,1,title='Ratio at Fit Start',xlabel='t [us]', ylabel='')
	zerocrossing_plot.plot(bins,Ratio,'bo',label='Ratio at Fit Start')
	zerocrossing_plot.plot(bins[firstbin], Ratio[firstbin],'bo',color='red') 
	plt.xlim([29,32])
	plt.axhline(y=0.0, color='black', linestyle='-.')
	save_figure(zerocrossing_plot,'zerocrossing_plot')

	Tmethod_plot = plt.figure(figsize=(10,6), dpi=800).add_subplot(1,1,1,title='T-Method',xlabel='t [us]', ylabel='events per 0.149 us')
	Tmethod_plot.plot(bins,events,label='data',linewidth=0.5)
	save_figure(Tmethod_plot,'Tmethod_plot')

	UV_plot = plt.figure(figsize=(10,6), dpi=800).add_subplot(1,1,1,title='U & V',xlabel='t [us]', ylabel='events per 0.149 us')
	UV_plot.plot(bins,V,label='V',linewidth=0.5)
	UV_plot.plot(bins,U,label='U',linewidth=0.5)
	save_figure(UV_plot,'UV_plot')

	S_plot = plt.figure(figsize=(10,6), dpi=800).add_subplot(1,1,1,title='S',xlabel='t [us]', ylabel='')
	S_plot.plot(bins,S,label='S',linewidth=0.7)
	S_plot.axvline(x=bins[S_peak], linewidth=0.5, color='black',linestyle='dotted')
	S_plot.text(bins[S_peak],0,'t ='+'%.3f'%bins[S_peak])
	save_figure(S_plot,'S_plot')

	# Fit S_plot with an exponential
	def whatup(x, a, b, c):
		return a * np.exp(-x/b) + c

	popt, pcov = scipy.optimize.curve_fit(whatup, bins[S_peak:], S[S_peak:], p0=(10000000.0, 64.4, 0.0))
	sfit = plt.figure(figsize=(10,6), dpi=600).add_subplot(1,1,1,title='s fit',xlabel='t [us]', ylabel='')
	sfit.plot(bins[S_peak:], S[S_peak:], 'bo', label="S=U+V")
	sfit.plot(bins[S_peak:], whatup(bins[S_peak:], *popt), 'r-', label="Fitted with an exponential")
	save_figure(sfit, "sfit")

	D_plot = plt.figure(figsize=(10,6), dpi=800).add_subplot(1,1,1,title='D',xlabel='t [us]', ylabel='')
	D_plot.plot(bins,D,label='D',linewidth=0.7)
	D_plot.axvline(x=bins[S_peak], linewidth=0.5, color='black',linestyle='dotted')
	D_plot.text(bins[S_peak],0,'t ='+'%.3f'%bins[S_peak])
	save_figure(D_plot,'D_plot')

	variance_plot = plt.figure(figsize=(10,6), dpi=800).add_subplot(1,1,1,title='variance on each bin over fit range',xlabel='t [us]', ylabel='')
	variance_plot.plot(bins_for_fit,variance,label='variance',linewidth=0.5)
	save_figure(variance_plot,'variance_plot')

	RawRatio_plot = plt.figure(figsize=(10,6), dpi=800).add_subplot(1,1,1,title='Ratio',xlabel='t [us]', ylabel='')
	RawRatio_plot.plot(bins,Ratio,label='Ratio',linewidth=0.5)
	save_figure(RawRatio_plot,'RawRatio_plot')

print 'Bin width:'+str(bins_for_fit[1]-bins_for_fit[0]) # sanity check

#												#
# 	CHOOSE A PARAMS AND MODEL BASED ON DATASET 	#
#												#
if args.dataset  == 'endgame':
	initial_params = endgame_initial_params
	simple_initial_params = endgame_initial_params[0:3]
	simple_model = util.simple_model
	ratio_model = util.ratio_model
	ratio_param_names = ( 'A_dr','A','r','phi_0','omega_cbo','A_cbo_n','phi_cbo_n','A_vw','tau_vw','omega_vw','phi_vw')
	trackermodelparams = (2.337, 7.43, 95.1, 4.71, 9.0)

if args.dataset  == '9d':
	initial_params = ( 
	0.3639, -17.82, 2.08, 
	2.615,137.4,0.0039,2.2,0.0006,1.7,0.0008,4.3, 
	2.5, 
	0.00313, 28700., 14.5, -0.1,
	0.00022, -4.9   
	)
	simple_initial_params = _9d_initial_params[0:3]
	ratio_model = util.ratio_model
	ratio_param_names = ('A','r','phi_0','omega_cbo','tau_cbo','A_cbo_n','phi_cbo_n','A_cbo_a','phi_cbo_a','A_cbo_phi','phi_cbo_phi','K','A_vw','tau_vw','omega_vw','phi_vw','A_2cbo','phi_2cbo')
	trackermodelparams = ()

if args.dataset  == '60h':
	initial_params = (
	0.3637, -20.8, 2.09, 
	2.338, 175.2, 0.0043,-2.34,0.0005,-0.271,0.0008,-1.183, 
	8.9,
	0.00313, 28700.,14.5,-0.1,
	0.00019, 3.33
	)
	simple_initial_params = _60h_initial_params[0:3]
	ratio_model = util.ratio_model
	ratio_param_names = ('A','r','phi_0','omega_cbo','tau_cbo','A_cbo_n','phi_cbo_n','A_cbo_a','phi_cbo_a','A_cbo_phi','phi_cbo_phi','K','A_vw','tau_vw','omega_vw','phi_vw','A_2cbo','phi_2cbo')
	trackermodelparams = ()

if args.dataset  == '2c':
	initial_params = [0.37,-51.9,2.166,64.4, 	# A, r, phi_a, tau_mu
	0.003218, 237.150, 2.34, 6.5, 				# A_cbo, tau_cbo, omega_cbo, phi_cbo 
	0.053, 26.030, 14.2, -1.6, 					# A_omega_vw, tau_vw, omega_vw, phi_vw 
	-0.00016, 0.34] 							# A_2cbo, phi_2cbo
	t_initial_params = [0.37,-51.9,2.166,64.4, 	# A, r, phi_a, tau_mu
	0.003218, 237.150, 2.34, 6.5, 				# A_cbo, tau_cbo, omega_cbo, phi_cbo 
	0.001, 26.030, 14.2, -1.6, 					# A_omega_vw, tau_vw, omega_vw, phi_vw 
	-0.00016, 0.34, 4.5, 16000000]				# A_2cbo, phi_2cbo, K, N0
	sweigart_initial_params = [
	15700000, 0.37,-51.9,2.166,64.4, 	# N0, A, r, phi_a, tau_mu
	275.150, 2.34, 0.0032, 0.285, 		#'tau_cbo', 'omega_cbo', 'A_nx11', 'phi_nx11'
	50.0, 14.2, 0.00031, 0.8, 			#'tau_y','omega_vw', 'A_ny22', 'phi_ny22'
	0.000157, 3.76, 					#'A_nx22', 'phi_nx22'
	2.0, 								# K  
	0.0, 4.15, 2.37,  					#'A_ny11', 'phi_ny11' 
	0.0, 2.31, 							#'A_phix11', 'phi_phix11',
	0.0, 2.71 							#'A_ax11', 'phi_ax11',
	] 
	simple_initial_params = initial_params[0:3]
	simple_model = util.simple_model
	ratio_model = util._2c_model #sweigart model
	tmethod_model = util.t_full_function_2c
	mlp_2c = util.MuonLossProfile('run2c-20pDQC-Lt.pkl')
	lossseries = mlp_2c.intg_loss_series[:len(bins_for_fit)] #------------call once and thats it. feed to fit model
	ratio_param_names = ('A','r','phi_a','tau_mu','A_cbo','tau_cbo','omega_cbo','phi_cbo','A_vw','tau_vw','omega_vw','phi_vw','A_2cbo','phi_2cbo','K')
	t_param_names = ('A','r','phi_a','tau_mu','A_cbo','tau_cbo','omega_cbo','phi_cbo','A_vw','tau_vw','omega_vw','phi_vw','A_2cbo','phi_2cbo','K','N0')
	sweigart_param_names = ('N0', 'A', 'r', 'phi_a', 'tau_mu', 'tau_cbo', 'omega_cbo', 'A_nx11', 'phi_nx11', 'tau_y', 'omega_vw','A_ny22', 'phi_ny22', 'A_nx22', 'phi_nx22','K','A_ny11', 'phi_ny11', 'omega_y','A_phix11', 'phi_phix11','A_ax11', 'phi_ax11')
	sweigart_param_names_ratio = ('A', 'r', 'phi_a', 'tau_mu','tau_cbo', 'omega_cbo', 'A_nx11', 'phi_nx11', 'tau_y', 'omega_vw','A_ny22', 'phi_ny22', 'A_nx22', 'phi_nx22','K', 'A_ny11', 'phi_ny11', 'omega_y','A_phix11', 'phi_phix11','A_ax11', 'phi_ax11')


#																			#
# 	PERFORM 3-PARAM FIT AND FULL FIT (BOUNDED & FREE) FOR GIVEN DATASET 	#
#

# 1 Bounded T-Method Fit																		
tmethod_sw_result = scipy.optimize.least_squares( 
	util.residuals_fcn, 
	sweigart_initial_params,
	args=(util.sweigart_model, (bins_for_fit, tmeth_for_fit, tmeth_for_fit), lossseries) ,
	bounds = ([-np.inf, -np.inf,-np.inf,-np.inf,-np.inf,-np.inf, -np.inf, -np.inf, -np.inf, 0.0, -np.inf, 0.0003, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf]
		,[ np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
	)
tmethod_sw_result = fitresults.PFitResult(tmethod_sw_result, sweigart_initial_params, sweigart_param_names, shrink=False )
print 'T-Method fit using fundamental frequency model:\n'+str(tmethod_sw_result)

# 2 Auxiliary T-Method Fit (as needed)
if extrafit == True:
	tmethod_fit_result = scipy.optimize.least_squares( util.residuals_fcn, t_initial_params,
		args=(tmethod_model,(bins_for_fit,tmeth_for_fit,tmeth_for_fit)),
		bounds=([0.0,-np.inf,2.0,63.5,0.0029,200.0,2.2,-np.inf,-np.inf,0.0,14.0,-np.inf,-np.inf,-np.inf,-np.inf, 0.0],
			[0.45,np.inf,np.inf,65.0,1.0,300.0,2.5,np.inf,0.01,100.0,15.0,np.inf,1.0,np.inf,np.inf, np.inf])
		)
	tmethod_fit_result = fitresults.PFitResult(tmethod_fit_result, t_initial_params, t_param_names, shrink=False )
	print 'T-Method fit:\n'+str(tmethod_fit_result)

	#plot muon loss correction factor -- for debugging/verification purposes
	K_loss = tmethod_fit_result.getparam15()
	ML_plot = plt.figure(figsize=(10,6), dpi=800).add_subplot(1,1,1,title='Muon loss correction from run2c-20pDQC-Lt.pkl',xlabel='t [us]', ylabel='')
	ML_plot.plot(mlp_2c.diff_loss_xdata,1.-K_loss*mlp_2c.intg_loss_series,label='model',linewidth=0.9, linestyle='dotted', color='black')
	ML_plot.axvline(x=starttime, color='green',linestyle=':',label='fit start') 
	save_figure(ML_plot,'ML_plot')

# 3 Unbounded 3-Param Ratio Fit
simple_fit_result = scipy.optimize.leastsq( util.residuals_fcn,simple_initial_params,
	  args=(simple_model,(bins_for_fit,Ratio_for_fit,variance)),full_output=1, maxfev=6000 )

simple_fit_result = fitresults.FitResult(simple_fit_result, simple_initial_params, ('A','r','phi_0'), shrink=False )
#print 'Ratio fit:\n'+str(simple_fit_result)

# 4 Bounded Full Ratio Fit 
sweigart_initial_params.pop(0) #remove first entry (N0)
ttaumu = tmethod_sw_result.getparamN(4) #get tau_mu
tK_loss = tmethod_sw_result.getparamN(15) #get K 
print 'From the preliminary T-Method fit, grab muon lifetime and K_loss.\n'
print 'tau_mu = '+str(ttaumu)+'  K_loss = '+str(tK_loss)

sweigart_initial_params[3] = ttaumu
sweigart_initial_params[14] = tK_loss

fit_result = scipy.optimize.least_squares(
	  util.residuals_fcn,
	  sweigart_initial_params,
	  bounds = ([ -np.inf,-np.inf,-np.inf,ttaumu-0.01*ttaumu,-np.inf, -np.inf, -np.inf, -np.inf, 0.0, -np.inf, 0.0003, -np.inf, -np.inf, -np.inf, tK_loss-0.1*tK_loss, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf]
		       ,[  np.inf, np.inf, np.inf, ttaumu+0.01*ttaumu, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,  np.inf,  np.inf,  tK_loss+0.1*tK_loss,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf, np.inf]),
	  args=(ratio_model,(bins_for_fit,Ratio_for_fit,variance), lossseries) 
	)
fit_result = fitresults.PFitResult(fit_result,sweigart_initial_params,sweigart_param_names_ratio,shrink=False) 
print 'Bounded Ratio fit:\n'+str(fit_result)


# =============================== #
#
#	PLOTS (Fit Results, FFTs)	  #
#
# =============================== #

# Ratio Plot for unbounded fit
# -i took out the ratio plot for now-


#FFT of Residuals 
sampling_period = bins_for_fit[1]-bins_for_fit[0]
#fft_x_simple,fft_y_simple = util.do_residuals_fft(simple_fit_result.residuals,sampling_period)
fft_x_R,fft_y_R = util.do_residuals_fft(fit_result.residuals,sampling_period)			# Ratio Method
fft_x_S,fft_y_S = util.do_residuals_fft(tmethod_sw_result.residuals,sampling_period)	# T-Method

fig = plt.figure(figsize=(15,9))
fig.text(0.5,0.97,'title',size='x-large', horizontalalignment='center', verticalalignment='top', multialignment='center')
fig.subplots_adjust(left=0.08, bottom=0.08, right=0.93, top=0.88, hspace=0.)

fft_plot_top = fig.add_subplot(1,1,1, xlabel='frequency [rad Mhz]', ylabel='FT Magnitude')
fft_plot_top.xaxis.tick_bottom() # draw ticklabels at top
fft_plot_top.xaxis.set_ticks_position('both') # tickmarks at top and bottom
#fft_plot_top.plot(fft_x_simple, np.abs(fft_y_simple), color='black', label='3-param ratio fit')
#fft_plot_top.plot(fft_x, np.abs(fft_y), color='red', label='full ratio fit')
fft_plot_top.plot(fft_x_S, np.abs(fft_y_S), color='black', label='fundamental freq tmethod fit')
fft_plot_top.plot(fft_x_R, np.abs(fft_y_R), color='red', linewidth=0.8, label='full ratio fit')
#plt.xlim([-1, 20])
fft_plot_top.axvline(x=1.43948, color='pink',linestyle=':',label='omega_a') #omega_a
fft_plot_top.axvline(x=0.89, color='yellow',linestyle=':',label='cbo-omega_a') #cbo-omega_a
fft_plot_top.axvline(x=3.79, color='yellow',linestyle=':',label='omega_a+cbo') #cbo+omega_a
fft_plot_top.axvline(x=2.338, color='green',linestyle='-.',label='cbo') #cbo
fft_plot_top.axvline(x=4.64, color='green',linestyle=':',label='2cbo') #2cbo
fft_plot_top.axvline(x=14.27, color='pink',linestyle='-.',label='vw') #vw
fft_plot_top.axvline(x=5.22, color='pink',linestyle='-',label='2(cyc-x)') #2(fc-fx)
fft_plot_top.axvline(x=42.11-39.5, color='blue',linestyle='-.',label='cyc-x')#fc-fx
#fft_plot_top.axvline(x=42.11-13.8, color='red',linestyle='-.',label='cyc-y')#fc-fy
fft_plot_top.axvline(x=42.11-27.6, color='blue',linestyle='-.',label='cyc-2y')#fc-2*fy
#fft_plot_top.axvline(x=27.6, color='blue',linestyle='-.',label='2y')#2fy
fft_plot_top.axvline(x=13.8, color='red',linestyle=':',label='y') #fy
#fft_plot_top.axvline(x=39.5, color='blue',linestyle=':',label='x')#fx
#fft_plot_top.axvline(x=42.11, color='black',linestyle=':',label='cyc')#fc
#fft_plot_top.figure.text(0.4,0.8,'peak: %.6f rad Ghz'%(freq_at_peak,))
fft_plot_top.legend()

fft_plot_top.axis(xmax=fft_x_R[-1])
save_figure(fft_plot_top,'fit_residuals_fft')

# ================================================================= #
#
#								SCANS 								#
#	
# ================================================================= #
end_time_list, eth_list, calonum_list, start_time_list, chi2ndf_list, pvalue_list, a_list, r_list, phi0_list, tau_mu_list, A_cbo_list, t_cbo_list, omega_cbo_list, phi_cbo_list, K_list = ([] for i in range(15))
chi2_error, a_error, r_error, phi_error, tau_mu_error, acbo_error, tcbo_error, omegacbo_error, phicbo_error, K_error=([] for i in range(10))

if do_energy_threshold_scan:
	for eth in range(1200,1700,50):
		fileName = "/gm2/data/users/irma/run2/run2cprod_with20pDQC/EthScan/run2c-20pDQC_eth"+str(eth)+".root"
		a = Root.TFile.Open(fileName)
		tmeth = a.Get("MuonCoincidence/TIME_FULL")
		vp = a.Get("MuonCoincidence/caloTimes_Vp_all")
		vm = a.Get("MuonCoincidence/caloTimes_Vm_all")
		up = a.Get("MuonCoincidence/caloTimes_Up_all")
		um = a.Get("MuonCoincidence/caloTimes_Um_all")
		d_um = a.Get("MuonCoincidence/DTimes_Um_all")
		s1_um = a.Get("MuonCoincidence/S1Times_Um_all")              
		s2_um = a.Get("MuonCoincidence/S2Times_Um_all")
		d_up = a.Get("MuonCoincidence/DTimes_Up_all")
		s1_up = a.Get("MuonCoincidence/S1Times_Up_all") 
		s2_up = a.Get("MuonCoincidence/S2Times_Up_all")
		d_vm = a.Get("MuonCoincidence/DTimes_Vm_all")
		s1_vm = a.Get("MuonCoincidence/S1Times_Vm_all")
		s2_vm = a.Get("MuonCoincidence/S2Times_Vm_all")
		d_vp = a.Get("MuonCoincidence/DTimes_Vp_all")
		s1_vp = a.Get("MuonCoincidence/S1Times_Vp_all")
		s2_vp = a.Get("MuonCoincidence/S2Times_Vp_all")

		#TMethod Plot -- under/overflow bins are cut out
		events = np.array(tmeth)[1:-1]
		nbins = tmeth.GetNbinsX() 
		bins = np.linspace(tmeth.GetXaxis().GetBinCenter(1), tmeth.GetXaxis().GetBinCenter(nbins), num=nbins)

		#Subtract Pileup
		Um = np.array(um)[1:-1]-np.array(d_um)[1:-1]+np.array(s1_um)[1:-1]+np.array(s2_um)[1:-1] 
		Up = np.array(up)[1:-1]-np.array(d_up)[1:-1]+np.array(s1_up)[1:-1]+np.array(s2_up)[1:-1] 
		Vm = np.array(vm)[1:-1]-np.array(d_vm)[1:-1]+np.array(s1_vm)[1:-1]+np.array(s2_vm)[1:-1] 
		Vp = np.array(vp)[1:-1]-np.array(d_vp)[1:-1]+np.array(s1_vp)[1:-1]+np.array(s2_vp)[1:-1] 

		#Compose 'graphs' of U, V, and their linear combinations 
		U = Um + Up
		V = Vm + Vp
		S = U + V
		D = U - V
		Ratio = D/S
		variance = (1-Ratio**2) / S #Eqn. 22 from E821 Note 366

		S_peak = np.argmax(S) # Get bin# of peak of exponential decay function -- determines starting point for Ratio plot
		print "Unclipped plot begins at t="+str(bins[S_peak]) #start times will differ among calo scans if S_peak changes from calo to calo

		#GLOBAL: Set Start and End times based on dataset + dataset model and initial params
	

		firstbin = len([i for i, val in enumerate(bins) if val < starttime])
		if firstbin < S_peak: #make sure firstbin happens after S_peak, otherwise set S_peak to start time
			firstbin = S_peak
		lastbin = next(x for x, val in enumerate(bins) if val > endtime)

		#Clip messy early and late times from Ratio plot; update range on variance array (will have to use later in residuals function)
		bins_for_fit = bins[firstbin:lastbin]
		Ratio_for_fit = Ratio[firstbin:lastbin]
		variance = variance[firstbin:lastbin]

		fit_result = scipy.optimize.least_squares(
			util.residuals_fcn, sweigart_initial_params,
			bounds = ([ -np.inf,-np.inf,-np.inf,ttaumu-0.01*ttaumu,-np.inf, -np.inf, -np.inf, -np.inf, 0.0, -np.inf, 0.0003, -np.inf, -np.inf, -np.inf, tK_loss-0.1*tK_loss, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf]
					       ,[  np.inf, np.inf, np.inf, ttaumu+0.01*ttaumu, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,  np.inf,  np.inf,  tK_loss+0.1*tK_loss,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf, np.inf]),
			args=(ratio_model,(bins_for_fit,Ratio_for_fit,variance)) 
				)
		fit_result = fitresults.PFitResult(fit_result,initial_params,ratio_param_names,shrink=False) 
		print 'Ratio fit:\n'+str(fit_result)

		#VERIFY THAT PARAM # MATCHES PARAM LISTED ----------------------------------- !!!!!!!
		eth_list.append(eth)
		chi2ndf_list.append(fit_result.getgoodness())
		chi2_error.append( np.sqrt(2.0/fit_result.getndf()) ) #from kinnard thesis eqn 5.33
		pvalue_list.append(fit_result.getpvalue())
		a_list.append(fit_result.getparam0())
		a_error.append(fit_result.getparam0error())
		r_list.append(fit_result.getparam1())
		r_error.append(fit_result.getparam1error())
		phi0_list.append(fit_result.getparam2())
		phi_error.append(fit_result.getparam2error())
		tau_mu_list.append(fit_result.getparam3())
		tau_mu_error.append(fit_result.getparam3error())
		A_cbo_list.append(fit_result.getparam4()) #-------------
		acbo_error.append(fit_result.getparam4error())
		t_cbo_list.append(fit_result.getparam4())
		tcbo_error.append(fit_result.getparam4error())
		omega_cbo_list.append(fit_result.getparam5())
		omegacbo_error.append(fit_result.getparam5error())
		phi_cbo_list.append(fit_result.getparam6())
		phicbo_error.append(fit_result.getparam6error())
		K_list
		K_error

		# Ratio Plot
		Ratio_plot = plt.figure(figsize=(10,6), dpi=600).add_subplot(1,1,1,title='Ratio (Eth Scan)',xlabel='t [us]', ylabel='')
		Ratio_plot.plot(bins_for_fit,Ratio_for_fit,label='Ratio',linewidth=0.8, color='blue')
		Ratio_plot.plot(bins_for_fit,[ratio_model(t, initial_params) for t in bins_for_fit],label='model',linewidth=0.7, linestyle='dotted', color='green')
		Ratio_plot.plot(bins_for_fit,
			[ratio_model(t,fit_result.params) for t in bins_for_fit],
			label='model',linewidth=0.5, color='red', linestyle='dashed')
		save_figure(Ratio_plot,'Ratio_plot_with_model'+str(eth))
		plt.close()

	scanplots(eth_list, chi2ndf_list, chi2_error, pvalue_list, a_list, a_error, r_list, r_error, phi0_list, phi_error, tau_mu_list, tau_mu_error, A_cbo_list, acbo_error, t_cbo_list, tcbo_error, omega_cbo_list, omegacbo_error, phi_cbo_list, phicbo_error, K_list, K_error, args.scans )
	#end loop pver ethresh

	
#loop over calorimeter
calofailcount = 0
calofaillist = []

if do_calo_scan: 

	filename = args.infile[0]
	a = Root.TFile.Open(filename) 

	for calonum in range(0, 2): 
		
		#Grab component histograms from ROOT file
		#tmeth = a.Get("MuonCoincidence/TIME_FULL") #without pileup subtraction
		histname = "MuonCoincidence/TimeOriginal_All_calo"+str(calonum)
		tmeth = a.Get(histname)

		histname = "MuonCoincidence/TimeOriginal_Vp_calo"+str(calonum) #vp = a.Get("MuonCoincidence/caloTimes_Vp_all")
		vp = a.Get(histname) 
		histname = "MuonCoincidence/TimeOriginal_Vm_calo"+str(calonum) #vm = a.Get("MuonCoincidence/caloTimes_Vm_all")
		vm = a.Get(histname)
		histname = "MuonCoincidence/TimeOriginal_Up_calo"+str(calonum) #vm = a.Get("MuonCoincidence/caloTimes_Up_all")
		up = a.Get(histname)
		histname = "MuonCoincidence/TimeOriginal_Um_calo"+str(calonum) #vm = a.Get("MuonCoincidence/caloTimes_Um_all")
		um = a.Get(histname)

		histname = "MuonCoincidence/time_D_Um_calo_"+str(calonum)  #d_um = a.Get("MuonCoincidence/DTimes_Um_all")
		d_um = a.Get(histname) 
		histname = "MuonCoincidence/time_S1_Um_calo_"+str(calonum) #s1_um = a.Get("MuonCoincidence/S1Times_Um_all")
		s1_um = a.Get(histname) 
		histname = "MuonCoincidence/time_S2_Um_calo_"+str(calonum) #s2_um = a.Get("MuonCoincidence/S2Times_Um_all")
		s2_um = a.Get(histname) 

		histname = "MuonCoincidence/time_D_Up_calo_"+str(calonum) #d_up = a.Get("MuonCoincidence/DTimes_Up_all")
		d_up = a.Get(histname) 
		histname = "MuonCoincidence/time_S1_Up_calo_"+str(calonum) #s1_up = a.Get("MuonCoincidence/S1Times_Up_all")
		s1_up = a.Get(histname)
		histname = "MuonCoincidence/time_S2_Up_calo_"+str(calonum) #s2_up = a.Get("MuonCoincidence/S2Times_Up_all")
		s2_up = a.Get(histname)


		histname = "MuonCoincidence/time_D_Vm_calo_"+str(calonum)  #d_vm = a.Get("MuonCoincidence/DTimes_Vm_all")
		d_vm = a.Get(histname) 
		histname = "MuonCoincidence/time_S1_Vm_calo_"+str(calonum) #s1_vm = a.Get("MuonCoincidence/S1Times_Vm_all")
		s1_vm = a.Get(histname) 
		histname = "MuonCoincidence/time_S2_Vm_calo_"+str(calonum) #s2_vm = a.Get("MuonCoincidence/S2Times_Vm_all")
		s2_vm = a.Get(histname) 

		histname = "MuonCoincidence/time_D_Vp_calo_"+str(calonum) #d_vp = a.Get("MuonCoincidence/DTimes_Vp_all")
		d_vp = a.Get(histname) 
		histname = "MuonCoincidence/time_S1_Vp_calo_"+str(calonum) #s1_vp = a.Get("MuonCoincidence/S1Times_Vp_all")
		s1_vp = a.Get(histname) 
		histname = "MuonCoincidence/time_S2_Vp_calo_"+str(calonum) #s2_vp = a.Get("MuonCoincidence/S2Times_Vp_all")
		s2_vp = a.Get(histname) 
		print 'test'

		#TMethod Plot -- under/overflow bins are cut out
		events = np.array(tmeth)[1:-1]
		nbins = tmeth.GetNbinsX() 
		bins = np.linspace(tmeth.GetXaxis().GetBinCenter(1), tmeth.GetXaxis().GetBinCenter(nbins), num=nbins)

		#Subtract Pileup
		Um = np.array(um)[1:-1]-np.array(d_um)[1:-1]+np.array(s1_um)[1:-1]+np.array(s2_um)[1:-1] 
		Up = np.array(up)[1:-1]-np.array(d_up)[1:-1]+np.array(s1_up)[1:-1]+np.array(s2_up)[1:-1] 
		Vm = np.array(vm)[1:-1]-np.array(d_vm)[1:-1]+np.array(s1_vm)[1:-1]+np.array(s2_vm)[1:-1] 
		Vp = np.array(vp)[1:-1]-np.array(d_vp)[1:-1]+np.array(s1_vp)[1:-1]+np.array(s2_vp)[1:-1] 

		#Compose 'graphs' of U, V, and their linear combinations 
		U = Um + Up
		V = Vm + Vp
		S = U + V
		D = U - V
		Ratio = D/S
		variance = (1-Ratio**2) / S #Eqn. 22 from E821 Note 366
		print 'test 1'
		S_peak = np.argmax(S) # Get bin# of peak of exponential decay function -- determines starting point for Ratio plot

		# start and end times for calo scans will be different than for all calo sum
		firstcalobin = len([i for i, val in enumerate(bins) if val < 34.3])
		if firstcalobin < S_peak: #make sure firstbin happens after S_peak, otherwise set S_peak to start time
			firstcalobin = S_peak
		lastcalobin  = next(x for x, val in enumerate(bins) if val > 600.0)

		#Clip messy early and late times from Ratio plot; update range on variance array (will have to use later in residuals function)
		bins_for_fit = bins[firstcalobin:lastcalobin]
		Ratio_for_fit = Ratio[firstcalobin:lastcalobin]
		variance = variance[firstcalobin:lastcalobin]

		#diagnostic plots can go here if needed
		print 'test 2'

		# 											#
		#  Perform bounded full fit for each calo 	#
		#    										#
		fit_result = scipy.optimize.least_squares(
			  util.residuals_fcn,
			  sweigart_initial_params,
			  bounds=([ -np.inf,-np.inf,-np.inf,ttaumu-0.01*ttaumu,-np.inf, -np.inf, -np.inf, -np.inf, 0.0, -np.inf, 0.0003, -np.inf, -np.inf, -np.inf, tK_loss-0.1*tK_loss, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf]
			  	,[  np.inf, np.inf, np.inf, ttaumu+0.01*ttaumu, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,  np.inf,  np.inf,  tK_loss+0.1*tK_loss,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf, np.inf]),
			  args=(ratio_model,(bins_for_fit,Ratio_for_fit,variance)) 
			)




			# fit is taking foreverrrrr --  figure out why lost muon piece is so slow












		fit_result = fitresults.PFitResult(fit_result,sweigart_initial_params,sweigart_param_names_ratio,shrink=False) 
		#print 'Ratio fit:\n'+str(fit_result)
		print 'test 3'
		calonum_list.append(calonum)
		chi2ndf_list.append(fit_result.getgoodness())
		chi2_error.append( np.sqrt(2.0/fit_result.getndf()) ) #from kinnard thesis eqn 5.33
		pvalue_list.append(fit_result.getpvalue())
		a_list.append(fit_result.getparamN(0))
		a_error.append(fit_result.getparamNerror(0))
		r_list.append(fit_result.getparamN(1))
		r_error.append(fit_result.getparamNerror(1))
		phi0_list.append(fit_result.getparamN(2))
		phi_error.append(fit_result.getparamNerror(2))
		tau_mu_list.append(fit_result.getparamN(3))
		tau_mu_error.append(fit_result.getparamNerror(3))
		A_cbo_list.append(fit_result.getparamN(6))
		acbo_error.append(fit_result.getparamNerror(6))
		t_cbo_list.append(fit_result.getparamN(4))
		tcbo_error.append(fit_result.getparamNerror(4))
		omega_cbo_list.append(fit_result.getparamN(5))
		omegacbo_error.append(fit_result.getparamNerror(5))
		phi_cbo_list.append(fit_result.getparamN(7))
		phicbo_error.append(fit_result.getparamNerror(7))
		K_list.append(fit_result.getparamN(14))
		K_error.append(fit_result.getparamNerror(14))
		print 'test4'
		# A=0, r=1, phi_a=2, tau_mu=3, 
	    #tau_cbo=4, omega_cbo=5, A_nx11=6, phi_nx11=7, 
	    #tau_y, omega_vw, A_ny22, phi_ny22, 
	    #A_nx22, phi_nx22, 
	    #K=14, 
	    #A_ny11, phi_ny11, omega_y, 
	    #A_phix11, phi_phix11, 
	    #A_ax11, phi_ax11 
		
		# Ratio Plot
		'''
		Ratio_plot = plt.figure(figsize=(10,6), dpi=800).add_subplot(1,1,1,title='Ratio',xlabel='t [us]', ylabel='')
		Ratio_plot.plot(bins_for_fit,Ratio_for_fit,label='Ratio',linewidth=0.8, color='blue')
		Ratio_plot.plot(bins_for_fit,[ratio_model(t, sweigart_initial_params) for t in bins_for_fit],label='model',linewidth=0.7, linestyle='dotted', color='green')
		Ratio_plot.plot(bins_for_fit,
			[ratio_model(t,fit_result.params) for t in bins_for_fit],
			label='model',linewidth=0.5, color='red', linestyle='dashed')
		save_figure(Ratio_plot,'Ratio_plot_with_model'+str(calonum))
		plt.close()
		'''

		#FFT of Residuals 
		sampling_period = bins_for_fit[1]-bins_for_fit[0]
		fft_x,fft_y = util.do_residuals_fft(fit_result.residuals,sampling_period)

		fig = plt.figure(figsize=(15,9))
		fig.text(0.5,0.97,'title',size='x-large', horizontalalignment='center', verticalalignment='top', multialignment='center')
		fig.subplots_adjust(left=0.08, bottom=0.08, right=0.93, top=0.88, hspace=0.)

		fft_plot_top = fig.add_subplot(1,1,1, xlabel='frequency [rad Mhz]', ylabel='FT Magnitude')
		fft_plot_top.xaxis.tick_bottom() # draw ticklabels at top
		fft_plot_top.xaxis.set_ticks_position('both') # tickmarks at top and bottom
		fft_plot_top.plot(fft_x, np.abs(fft_y), color='red', label='full ratio fit')
		plt.xlim([-1, 17])
		fft_plot_top.axvline(x=0.89, color='yellow',linestyle=':',label='cbo-omega_a') #cbo-omega_a    
		fft_plot_top.axvline(x=2.338, color='green',linestyle='-.',label='cbo') #cbo 					THESE LINES CORRESPOND			 
		fft_plot_top.axvline(x=4.64, color='green',linestyle=':',label='2cbo') #2cbo 						TO RUN 1 VALUES
		fft_plot_top.axvline(x=14.58, color='blue',linestyle=':',label='f_y') #fy 						FOR THESE FREQUENCIES
		fft_plot_top.axvline(x=39.405, color='blue',linestyle=':',label='f_x') #fy
		fft_plot_top.axvline(x=3.79, color='yellow',linestyle=':',label='omega_a+cbo') #cbo+omega_a
		fft_plot_top.axvline(x=1.43948, color='pink',linestyle=':',label='omega_a') #omega_a
		fft_plot_top.axvline(x=12.936, color='blue',linestyle='-.',label='vw') #vw
		fft_plot_top.legend()

		fft_plot_top.axis(xmax=fft_x[-1])
		  
		save_figure(fft_plot_top,'ratio_residuals_ft'+str(calonum))
		plt.close()
		print 'test 5'
		if not fit_result.success:
			calofailcount+=1
			calofaillist.append(calonum)
		
	print(calofaillist) 

	#Calo Scan Plots
	scanplots(calonum_list, chi2ndf_list, chi2_error, pvalue_list, a_list, a_error, r_list, r_error, phi0_list, phi_error, tau_mu_list, tau_mu_error, A_cbo_list, acbo_error, t_cbo_list, tcbo_error, omega_cbo_list, omegacbo_error, phi_cbo_list, phicbo_error, K_list, K_error, args.scans )
	

							
# 	Start Time Scan Fits 	
if do_start_time_scan:
	count=0
	
	for x in range(0,500,7): #bin number for start time
		bins_for_startscan=bins_for_fit[x:]
		Ratio_for_startscan=Ratio_for_fit[x:]
		variance_for_startscan=variance[x:]

		print 'Start Time: ' +str(bins_for_startscan[0])

		fit_result = scipy.optimize.leastsq(
			  util.residuals_fcn,
			  initial_params,
			  args=(ratio_model,(bins_for_startscan,Ratio_for_startscan,variance_for_startscan)),
			  full_output=1, 
			  maxfev=6000
			)
		fit_result = fitresults.FitResult(fit_result,initial_params,ratio_param_names,shrink=False) 
		print 'Ratio fit:\n'+str(fit_result)

		start_time_list.append(bins_for_startscan[0])
		chi2ndf_list.append(fit_result.getgoodness())
		chi2_error.append( np.sqrt(2.0/fit_result.getndf()) ) #from kinnard thesis eqn 5.33
		pvalue_list.append(fit_result.getpvalue())
		a_list.append(fit_result.getparam0())
		a_error.append(fit_result.getparam0error())
		r_list.append(fit_result.getparam1())
		r_error.append(fit_result.getparam1error())
		phi0_list.append(fit_result.getparam2())
		phi_error.append(fit_result.getparam2error())
		tau_mu_list
		tau_mu_error
		A_cbo_list.append(fit_result.getparam3())
		acbo_error.append(fit_result.getparam3error())
		t_cbo_list.append(fit_result.getparam4())
		tcbo_error.append(fit_result.getparam4error())
		omega_cbo_list.append(fit_result.getparam5())
		omegacbo_error.append(fit_result.getparam5error())
		phi_cbo_list.append(fit_result.getparam6())
		phicbo_error.append(fit_result.getparam6error())
		K_list
		K_error

		if not fit_result.success:
			count+=1
	print(count)

	#Start Time Scan Plots
	scanplots(start_time_list, chi2ndf_list, chi2_error, pvalue_list, a_list, a_error, r_list, r_error, phi0_list, phi_error, tau_mu_list, tau_mu_error, A_cbo_list, acbo_error, t_cbo_list, tcbo_error, omega_cbo_list, omegacbo_error, phi_cbo_list, phicbo_error, K_list, K_error, args.scans )

# 	Start Time Scan Fits 	
if do_end_time_scan:
	count=0
	
	for x in range(3500,3900,7): #bin number for start time
		bins_for_endscan=bins_for_fit[firstbin:x]
		Ratio_for_endscan=Ratio_for_fit[firstbin:x]
		variance_for_endscan=variance[firstbin:x]

		print 'End time: '+str(bins_for_startscan[x])

		fit_result = scipy.optimize.leastsq(
			  util.residuals_fcn,
			  initial_params,
			  args=(ratio_model,(bins_for_endscan,Ratio_for_endscan,variance_for_endscan)),
			  full_output=1, 
			  maxfev=6000
			)
		fit_result = fitresults.FitResult(fit_result,initial_params,ratio_param_names,shrink=False) 
		print 'Ratio fit:\n'+str(fit_result)

		start_time_list.append(bins_for_startscan[0])
		chi2ndf_list.append(fit_result.getgoodness())
		chi2_error.append( np.sqrt(2.0/fit_result.getndf()) ) #from kinnard thesis eqn 5.33
		pvalue_list.append(fit_result.getpvalue())
		a_list.append(fit_result.getparam0())
		a_error.append(fit_result.getparam0error())
		r_list.append(fit_result.getparam1())
		r_error.append(fit_result.getparam1error())
		phi0_list.append(fit_result.getparam2())
		phi_error.append(fit_result.getparam2error())
		tau_mu_list
		tau_mu_error
		A_cbo_list.append(fit_result.getparam3())
		acbo_error.append(fit_result.getparam3error())
		t_cbo_list.append(fit_result.getparam4())
		tcbo_error.append(fit_result.getparam4error())
		omega_cbo_list.append(fit_result.getparam5())
		omegacbo_error.append(fit_result.getparam5error())
		phi_cbo_list.append(fit_result.getparam6())
		phicbo_error.append(fit_result.getparam6error())
		K_list
		K_error

		if not fit_result.success:
			count+=1
	print(count)

	#Start Time Scan Plots
	scanplots(end_time_list, chi2ndf_list, chi2_error, pvalue_list, a_list, a_error, r_list, r_error, phi0_list, phi_error, tau_mu_list, tau_mu_error, A_cbo_list, acbo_error, t_cbo_list, tcbo_error, omega_cbo_list, omegacbo_error, phi_cbo_list, phicbo_error, K_list, K_error, args.scans )



'''
# Wrapped Ratio Plot AXES NEEDS ADJUSTING
fig, ax = plt.subplots(7)
plt.subplots_adjust(hspace=0.0)
fig.suptitle('Ratio Fit (from 35-600us)')
for i in range(0,7):
  ax[i].set_ylim(-1,1)
  rxlist = bins_for_fit[550*i:550*(i+1)] 
  rylist = Ratio_for_fit[550*i:550*(i+1)] 
  ax[i].plot(rxlist, rylist)
  ax[i].plot(rxlist, [ ratio_model(t,fit_result.params) for t in rxlist ], linewidth=0.5, linestyle='dashed') 
for i in range(1,6):
  ax[i].spines['top'].set_visible(False) 
  ax[i].spines['bottom'].set_visible(False)
  ax[i].get_xaxis().set_visible(False)
  ax[i].get_yaxis().set_visible(False) 
ax[0].spines['bottom'].set_visible(False) 
ax[6].get_yaxis().set_visible(False) 
ax[6].spines['top'].set_visible(False)
fig.text(0.5, 0.04, 'Time modulo x [us]', ha='center')
fig.text(0.01, 0.5, 'Ratio Value', va='center', rotation='vertical')
fig.savefig('fullratio_wrap.png', dpi=600)


#Fit Residual/Bin Error vs Time
r_pull = fit_result.residuals #residuals are actually already scaled by sqrt-variance in definition
pulls_plot = plt.figure(figsize=(10,6), dpi=800).add_subplot(1,1,1,title='Pulls',xlabel='t [us]', ylabel='')
pulls_plot.plot(bins_for_fit,r_pull,'bo',label='Pulls')
save_figure(pulls_plot,'pulls_plot')

#Fit Residual/Bin Error Histogram 
mean = sstats.tmean(r_pull)
sd = sstats.tstd(r_pull) 
fig = plt.hist(r_pull, bins=100, alpha=0.5)
textstring = "mean: %s \nstd dev: %s" % ("%.5f" % mean, "%.5f" %sd)
plt.text(3,100,textstring,bbox=dict(facecolor='red', alpha=0.5))
plt.xlim([-8, 8])
plt.title('Ratio Fit Pull')
plt.xlabel('Fit Residual/Bin Error')
plt.ylabel('counts')
plt.savefig('pull_histogram.png')

#FFT of Residuals --- Plot fft of full model on same axis as fft of 3-param fit for comparison

sampling_period = bins_for_fit[1]-bins_for_fit[0]
fft_x,fft_y = util.do_residuals_fft(fit_result.residuals,sampling_period)
fft_x_simple,fft_y_simple = util.do_residuals_fft(simple_fit_result.residuals,sampling_period)

fig = plt.figure(figsize=(15,9))
fig.text(0.5,0.97,'title',size='x-large', 
horizontalalignment='center', verticalalignment='top', 
multialignment='center')
fig.subplots_adjust(
left=0.08, bottom=0.08, right=0.93, top=0.88, 
hspace=0.)

fft_plot_top = fig.add_subplot(1,1,1, xlabel='frequency [rad Mhz]', ylabel='FT Magnitude')
fft_plot_top.xaxis.tick_bottom() # draw ticklabels at top
fft_plot_top.xaxis.set_ticks_position('both') # tickmarks at top and bottom
fft_plot_top.plot(fft_x_simple, np.abs(fft_y_simple), color='black', label='3-param ratio fit')
fft_plot_top.plot(fft_x, np.abs(fft_y), color='red', label='full ratio fit')
plt.xlim([-1, 17])
fft_plot_top.axvline(x=0.89, color='yellow',linestyle=':',label='cbo-omega_a') #cbo-omega_a
fft_plot_top.axvline(x=2.338, color='green',linestyle='-.',label='cbo') #cbo
fft_plot_top.axvline(x=4.64, color='green',linestyle=':',label='2cbo') #2cbo
fft_plot_top.axvline(x=13.76, color='blue',linestyle=':',label='y') #fy
fft_plot_top.axvline(x=3.79, color='yellow',linestyle=':',label='omega_a+cbo') #cbo+omega_a
fft_plot_top.axvline(x=1.43948, color='pink',linestyle=':',label='omega_a') #omega_a
fft_plot_top.axvline(x=14.27, color='blue',linestyle='-.',label='vw') #vw
#fft_plot_top.figure.text(0.4,0.8,'peak: %.6f rad Ghz'%(freq_at_peak,))
fft_plot_top.legend()

fft_plot_top.axis(xmax=fft_x[-1])
  
save_figure(fft_plot_top,'ratio_residuals_ft')

'''
