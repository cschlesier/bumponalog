
import math
import random
import os
import pickle
import numpy as np
import ROOT as Root
import scipy
import scipy.stats as sstats
import matplotlib
import matplotlib.pyplot as plt
from pylab import frange
from matplotlib import gridspec
import scipy.stats
from scipy.stats import gaussian_kde
from scipy import signal

from fitresult import *

import Blinders

my_blinder = None
try: 
  my_blinder = Blinders.Blinders(
    Blinders.FitType.Omega_a, 
    os.environ['OMEGA_BLINDERS_PHRASE']
  )
except: 
  raise RuntimeError('''
    Failed to create Blinders object!
    Did you export OMEGA_BLINDERS_PHRASE in your environment?
  '''
  )
print (my_blinder)

#omega_a0 = 0.00145
#omega_a0 = 2.*math.pi/4370. # rad*GHz 
omega_a0 = my_blinder.reference_value()
half_T_a0 = math.pi/omega_a0

#           #
# MODEL FUNCTIONS   #
# ---------------   # 

def simpleratio_function(t, params):
  '''
  A 3-param ratio--to be used for comparison purposes.
  '''
  A_mu,r,phi_0 = params

  freq = my_blinder.param_to_freq_radMHz(r)

  return (1. + A_mu*np.cos(freq*t + phi_0))

def irma_function(t,params, **kwargs):

  #tau_cbo and K are fixed in the IRMA endgame model

  A_dr,A_mu,r,phi_0, omega_cbo, A_cbo_n, phi_cbo_n,A_vw,tau_vw,omega_vw,phi_vw = params
  freq = my_blinder.param_to_freq_radMHz(r)

  _trackermodelparams = 0
  if 'trackermodelparams' in kwargs:
    _trackermodelparams = kwargs.get("trackermodelparams")

  return (1-A_dr*np.exp(-t/19.6))*(1. + A_mu*np.cos(freq*t + phi_0))\
    * (1. + A_cbo_n*np.exp(-t/206.2)*np.cos(phi_cbo_n + omega_cbo*omega_cbo_dynamic(t,_trackermodelparams)*t)) \
    * (1. + np.exp(-t/tau_vw) * A_vw * np.cos(omega_vw*t+phi_vw))


def t_full_function_2c(t, params): # T-Method formulation
  # Constant CBO frequency

  A,r,phi_a,tau_mu,A_cbo,tau_cbo,omega_cbo,phi_cbo,A_omega_vw,tau_vw,omega_vw,phi_vw, A_2cbo,phi_2cbo, K, N0 = params 

  freq = my_blinder.param_to_freq_radMHz(r)

  return N0*np.exp(-t/tau_mu)*(1. + A*np.cos(freq*t + phi_a) ) \
    * (1. + A_cbo*np.exp(-t/tau_cbo)*np.cos(omega_cbo*t + phi_cbo) + A_2cbo*np.exp(-2*t/tau_cbo)*np.cos(2*omega_cbo*t + phi_2cbo) ) \
    * (1. + np.exp(-t/tau_vw) * A_omega_vw * np.cos(omega_vw*t+phi_vw)) \
    * (1.-K*mlp_2c.intg_loss_series[:len(t)]) # print out plot of intg_loss_series to verify that it looks ok 
    # also -- how can u verify that fit start time matches with t0 for lost muon series 


#/pnfs/GM2/scratch/users/sganguly/testlist/DAQ/Ana/ ---energy threshold scan might be here

def sweigart_model(t, params, *args): # T-Method formulation 

  if args: #if args exist -- should the model remove K from params if loss_series isn't present?
    loss_series = args[0] 

  N0, A, r, phi_a, tau_mu, tau_cbo, omega_cbo, A_nx11, phi_nx11, tau_y, omega_vw, A_ny22, phi_ny22, A_nx22, phi_nx22, K, A_ny11, phi_ny11, omega_y, A_phix11, phi_phix11, A_ax11, phi_ax11 = params  
  freq = my_blinder.param_to_freq_radMHz(r)

  return N0*np.exp(-t/tau_mu)*(1.+ A * (1. + np.exp(-t/tau_cbo)*A_ax11*np.cos(omega_cbo*t + phi_ax11) ) * np.cos(freq*t + phi_a * ( 1. + np.exp(-t/tau_cbo)*A_phix11*np.cos(omega_cbo*t + phi_phix11)) )) \
    * (1. + np.exp(-t/tau_cbo)*A_nx11 * np.cos(omega_cbo*t + phi_nx11) + np.exp(-2*t/tau_cbo)*A_nx22 * np.cos(2*omega_cbo*t + phi_nx22) ) \
    * (1. + np.exp(-2*t/tau_y)*A_ny22 * np.cos(omega_vw*t + phi_ny22)  + np.exp(-t/tau_y)*A_ny11 * np.cos(omega_y*t + phi_ny11) ) \
    * (1. - K*loss_series)


def sweigart_model_ratio(t, params, *args): # Ratio formulation i.e. no N0

  if args: 
    loss_series = args[0]
    print('yes') #-- args dont pass thru for some reason ???
  else:
    loss_series = mlp_2c.intg_loss_series[:len(t)] 

  A, r, phi_a, tau_mu, tau_cbo, omega_cbo, A_nx11, phi_nx11, tau_y, omega_vw, A_ny22, phi_ny22, A_nx22, phi_nx22, K, A_ny11, phi_ny11, omega_y, A_phix11, phi_phix11, A_ax11, phi_ax11 = params  
  freq = my_blinder.param_to_freq_radMHz(r)

  return np.exp(-t/tau_mu)*(1.+ A * (1. + np.exp(-t/tau_cbo)*A_ax11*np.cos(omega_cbo*t + phi_ax11) ) * np.cos(freq*t + phi_a * ( 1. + np.exp(-t/tau_cbo)*A_phix11*np.cos(omega_cbo*t + phi_phix11)) )) \
    * (1. + np.exp(-t/tau_cbo)*A_nx11 * np.cos(omega_cbo*t + phi_nx11) + np.exp(-2*t/tau_cbo)*A_nx22 * np.cos(2*omega_cbo*t + phi_nx22) ) \
    * (1. + np.exp(-2*t/tau_y)*A_ny22 * np.cos(omega_vw*t + phi_ny22)  + np.exp(-t/tau_y)*A_ny11 * np.cos(omega_y*t + phi_ny11) ) \
    * (1. - K*loss_series) 


def full_function_2c(t, params):
  # Constant CBO frequency

  A,r,phi_a,tau_mu,A_cbo,tau_cbo,omega_cbo,phi_cbo,A_omega_vw,tau_vw,omega_vw,phi_vw,A_2cbo,phi_2cbo,K = params #K

  freq = my_blinder.param_to_freq_radMHz(r)

  return np.exp(-t/tau_mu)*(1. + A*np.cos(freq*t + phi_a) )\
    * (1. + A_cbo*np.exp(-t/tau_cbo)*np.cos(omega_cbo*t + phi_cbo)) \
    * (1. + np.exp(-t/tau_vw) * A_omega_vw * np.cos(omega_vw*t+phi_vw)) \
    * (1. + A_2cbo*np.exp(-2*t/tau_cbo)*np.cos(2*omega_cbo*t + phi_2cbo)) \
    * (1.- K * mlp_2c.intg_loss_series[:len(t)])  

def drawing_function_2c(t, params):
  # Constant CBO frequency
  # Plot without lost muon profile because this slows down the plotting process to an unreasonable extent

  A,r,phi_a,tau_mu,A_cbo,tau_cbo,omega_cbo,phi_cbo,A_omega_vw,tau_vw,omega_vw,phi_vw,A_2cbo,phi_2cbo,K = params #K

  freq = my_blinder.param_to_freq_radMHz(r)

  return np.exp(-t/tau_mu)*(1. + A*np.cos(freq*t + phi_a) )\
    * (1. + A_cbo*np.exp(-t/tau_cbo)*np.cos(omega_cbo*t + phi_cbo)) \
    * (1. + np.exp(-t/tau_vw) * A_omega_vw * np.cos(omega_vw*t+phi_vw)) \
    * (1. + A_2cbo*np.exp(-2*t/tau_cbo)*np.cos(2*omega_cbo*t + phi_2cbo)) 
    #* (1.-K*mlp_2c.intg_loss_series[:3808])

def full_function(t, params, trackermodelparams):
  '''
  N0 and muon lifetime are not part of the function because they are expected to cancel in the ratio.
  Function includes dynamic tracker-based CBO model. (All Run 1 data suffers from time-dependent CBO frequency, model by tracker data.)
  '''
  #A_mu,r,phi_0,omega_cbo,tau_cbo,A_cbo_n,phi_cbo_n,A_cbo_a,phi_cbo_a,A_cbo_phi,phi_cbo_phi,K,A_vw,tau_vw,omega_vw,phi_vw,A_2cbo,phi_2cbo = params

  A_mu,r,phi_0,omega_cbo,A_cbo_n,phi_cbo_n,A_cbo_a,phi_cbo_a,A_cbo_phi,phi_cbo_phi,A_2cbo,phi_2cbo = params #everything bu K, VW, tau_cbo
  # fix tau_cbo = 200.3

  freq = my_blinder.param_to_freq_radMHz(r)

  return (1. + (A_mu*(1. + A_cbo_a*np.exp(-t/200.3)*np.cos(omega_cbo*omega_cbo_dynamic(t,trackermodelparams)*t + phi_cbo_a) )) \
    * np.cos(freq*t + (phi_0 + A_cbo_phi *np.exp(-t/200.3)*np.cos(omega_cbo*omega_cbo_dynamic(t,trackermodelparams)*t + phi_cbo_phi ) )) ) \
    * (1. + A_cbo_n*np.exp(-t/200.3)*np.cos(omega_cbo*omega_cbo_dynamic(t, trackermodelparams)*t + phi_cbo_n )) \
    * (1. + A_2cbo*np.exp(-2*t/200.3)*np.cos(2*omega_cbo_dynamic(t, trackermodelparams)*t + phi_2cbo)) 
    # * mlp_eg.correction_factor(t,lnC=K) 
    # * (1. + np.exp(-t/tau_vw) * A_vw * np.cos(omega_vw*t+phi_vw)) \
          
#                #
#   LOST MUONS   #
#   ----------   #
import scipy.interpolate
class MuonLossProfile(object):
  '''Create muon loss profile function from differential loss data.
  
  Requirements:
    * diff_loss xdata,ydata are sorted by increasing x
    * x points are evenly-spaced
  
  Either pass
  '''
  def __init__(self,
      infile='run2c-20pDQC-Lt.pkl',
      diff_loss_xdata=[],diff_loss_ydata=[], #keys ['ylist', 'plot_filename', 'pkl_filename', 'infilename', 'xlist']
      t0=32.0, 
      tau_mu=64.423
    ):
    if len(diff_loss_xdata)==0 and len(diff_loss_ydata)==0 and len(infile)>0:
      infile_handle = open(infile,'rb')
      infile_data = pickle.load(infile_handle)
      infile_handle.close()
      if 'xlist' not in infile_data.keys() or 'ylist' not in infile_data.keys():
        raise ValueError('ERROR: the file "' + infile 
          + '" does not contain a dictionary with keys "xlist" and "ylist"!'
        )
      diff_loss_xdata = infile_data['xlist']
      diff_loss_ydata = infile_data['ylist']
    elif len(diff_loss_xdata)>0 and len(diff_loss_ydata)>0 and len(infile)==0:
      pass # they've been read from the file for us
    else: raise RuntimeError(
      'specify either infile OR (xdata AND ydata) in the MuonLossProfile'
      + ' constructor, but not both!'
    )
    if t0<diff_loss_xdata[0]: raise ValueError(
      't0 (%f) must be greater than first x value (%f)'%(t0,diff_loss_xdata[0])
    )
    if len(diff_loss_xdata)!=len(diff_loss_ydata): raise ValueError(
      'diff_loss xdata and ydata are not equal length!' \
      + ' (%d vs. %d)'%(len(diff_loss_xdata),len(diff_loss_ydata))
    )

    self.infile = infile
    self.diff_loss_xdata = np.array(diff_loss_xdata)
    self.dx = diff_loss_xdata[1]-diff_loss_xdata[0]
    self.diff_loss_ydata = np.array(diff_loss_ydata)
    self.t0 = t0
    self.tau_mu = tau_mu
    self.diff_fn_interp = None
    self.intg_loss_series = None
    self.intg_loss_fn = None
    self.create_intg_loss_series()

  def create_intg_loss_series(self,norm=True):
    '''Creates an 'integral' of L(t')*exp(t'/tau) using time series.

    Puts y values in intg_loss_series.

    This corresponds to the integral from p11 of Sudeshna's denver talk.
    '''
    sum_ylist = []
    lasty = 0.
    for x,y in zip(self.diff_loss_xdata,self.diff_loss_ydata):
      if x<self.t0:
        sum_ylist += [ 0. ]
        continue
      sum_ylist += [ y*np.exp(x/self.tau_mu)*self.dx + lasty ]
      lasty += y*self.dx
    self.intg_loss_series = np.array(sum_ylist)
    if norm: self.intg_loss_series /= np.sum(self.intg_loss_series)
    return self.intg_loss_series


mlp_2c = MuonLossProfile('run2c-20pDQC-Lt.pkl') #where does this need to be?

#                       #
#   TRACKER-CBO MODEL   #
# Values aquired from J. Mott Elog 184. 
#   -----------------   #

trackermodelparams = (2.337, 7.43, 95.1, 4.71, 9.0) #endgame tracker 12

def omega_cbo_dynamic(t, trackermodelparams):

  '''Tracker model of dynamic CBO frequency. Allows omega_cbo to be set (or float) in the full function. '''
  omega0,A, tau_A, B, tau_B = trackermodelparams
  retval = 1. + (A/omega0*t)*np.exp(-t/tau_A) + (B/omega0*t)*np.exp(-t/tau_B)
  return retval


#               #
#   GENERAL RATIO COMPOSITION   # make sure these functions work as intended -- verify time units
# ------------------------- #
def ratio_composition(t,params,Tfn, half_T_a0=half_T_a0,*args,**kwargs):
  '''Ratio composition (U-V divided by U+V) for arbitrary T-hist model function.
  
  t = time
  params = whatever parameters the model takes
  Tfn = model function, with call signature Tfn(t,params,*args,**kwargs)
  
  Calls ratio_composition_top and ratio_composition_bottom.  All *args and 
  **kwargs are passed through to the model function.
  '''
  return ratio_composition_top(t,params,Tfn,half_T_a0=half_T_a0,*args,**kwargs) \
    / ratio_composition_bottom(t,params,Tfn,half_T_a0=half_T_a0,*args,**kwargs)

def ratio_composition_top(t,params,Tfn, half_T_a0=half_T_a0,*args,**kwargs):
  '''U-V for arbitrary T-hist model functions.
  
  t = time
  params = whatever parameters the model takes
  Tfn = model function, with call signature Tfn(t,params,*args,**kwargs)
  
  U = Tfn(t-Ta/2) + Tfn(t+Ta/2)
  V = 2Tfn(t)
  
  All *args and **kwargs are passed through to the model function.
  '''
  return (
    Tfn(t-half_T_a0,params,*args,**kwargs) \
    + Tfn(t+half_T_a0,params,*args,**kwargs) \
    - 2.*Tfn(t,params,*args,**kwargs) )


def ratio_composition_bottom(t,params,Tfn,half_T_a0=half_T_a0,*args,**kwargs):
  '''U+V for arbitrary T-hist model functions.
  
  t = time
  params = whatever parameters the model takes
  Tfn = model function, with call signature Tfn(t,params,*args,**kwargs)
  
  U = Tfn(t-Ta/2) + Tfn(t+Ta/2)
  V = 2Tfn(t)
  
  All *args and **kwargs are passed through to the model function.
  '''
  return (
    Tfn(t-half_T_a0,params,*args,**kwargs) \
    + Tfn(t+half_T_a0,params,*args,**kwargs) \
    + 2.*Tfn(t,params,*args,**kwargs) )


#                                 #
#   RATIO COMPOSITION BY DATASET  #
# ----------------------------    #
def irma_model(t,params):
  return ratio_composition(t,params,irma_function) 

def simple_model(t,params):
  return ratio_composition(t,params,simpleratio_function)

def _2c_model(t,params, *args): 
  return ratio_composition(t,params,sweigart_model_ratio, *args)

def drawing_2c_model(t,params):
  return ratio_composition(t,params,drawing_function_2c) 

#                        #
#   RESIDUALS FUNCTION   #
#   ------------------   #

#least_squares(residuals, initialparams, args=(a, y))
#def residuals(bins, a, y):
   #return fndslfn


def residuals_fcn(params, model_function, data, *args): # Minimizing this function is consistent with the definition of chi2 in the Fit Result
  #loss series is passed through *model_args

  bins_for_fit,data_for_fit,variance = data 
  retval = (data_for_fit-model_function(bins_for_fit, params, *args)) / np.sqrt(variance)

  #retval = (data_for_fit-model_function(bins_for_fit,params)) / variance
  return retval


def do_residuals_fft(
    residuals, # NOT the residuals scaled by stat error like in the FitResult!
    sampling_period # put in 149.2 to get rad*GHz out
  ):
  #print(sampling_period)
  sampling_period = float(sampling_period)
  fft_x = 2.*math.pi*np.fft.rfftfreq(len(residuals), d=sampling_period)
  fft_y = np.fft.rfft(residuals)
  return fft_x,fft_y



