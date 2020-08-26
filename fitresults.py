import math
import numpy as np
import scipy
import matplotlib.pyplot as plt

class FitResult(object):
  def __init__(me, leastsq_fit, p0, param_names=None, shrink=False):
    # grab stuff from leastsq fit, save other arguments
    me.params,me.cov_frac,infodict,me.status_description,me.status = leastsq_fit

    me.qtf,me.nfev,me.fjac,me.fvec,me.ipvt = [
      infodict[k] for k in ('qtf', 'nfev', 'fjac', 'fvec', 'ipvt')
    ]
    me.initial_params = p0
    me.param_names = param_names
    me.error_message = ''
    me.residuals = me.fvec
    
    # set defaults
    me.cov = np.array([])
    me.chi_square = None
    me.param_errors = None
    me.reduced_chi_square = None
    me.P_chi_square = None
    me.N_points = None
    me.N_params = None
    me.N_dof = None
    me.success = False
    me.correlation = None

    me.check_and_compute()
    
    if shrink==True: 
      me.fjac = None
      me.fvec = None
      

    if len(me.error_message)>0:
      print ('Error message(s):\n'+me.error_message)
  
  def check_and_compute(me):
    if me.param_names==None: 
      try:
        me.param_names = tuple([ 'p'+str(i) for i in range(len(me.initial_params)) ])
      except:
        me.error_message += 'Failed to generate automatic parameter names..?\n'
        return False
    try:
      me.N_points = len(me.fvec)
      me.N_params = len(me.params)
      me.N_dof = me.N_points - me.N_params
    except: 
      me.error_message += 'Could not count points, parameters, or degrees of freedom.  (Try\n'
      me.error_message += 'checking fvec or params.)\n'
      return False
    try: 
      me.cov = me.cov_frac * np.sum(me.fvec**2) / me.N_dof
      v = np.sqrt(np.diag(me.cov))
      outer_v = np.outer(v, v)
      me.correlation = me.cov / outer_v
      me.correlation[me.cov == 0] = 0
    except:
      me.error_message += 'Fractional covariance matrix ill-formed!\n'
      return False
    try:
      me.chi_square = sum(me.fvec**2)
      me.reduced_chi_square = me.chi_square/me.N_dof
      me.P_chi_square = 1.-scipy.stats.chi2.cdf(me.chi_square,me.N_dof)
    except: 
      me.error_message += 'Failed to compute chi_square!  (Check fvec.)\n'
      return False
    try:
      me.param_errors = [ math.sqrt(me.cov[i][i]) for i in range(len(me.cov)) ]
    except:
      me.error_message += 'Failed to compute parameter errors! (Check fvec and cov.)\n'
      return False
    if me.status not in (1,2,3,4): 
      me.error_message += 'FAILURE indicated by status '+str(me.status)+'\n'
      return False
    me.success = True # if we got this far then we're probably good
    return me.success

  def failed(me):
    if me.success:
      return False
    else:
      return True

  def getgoodness(me):
    if me.success:
      return me.reduced_chi_square
    else: 
      return 0.0

  def getndf(me):
    return me.N_dof

  def getpvalue(me):
    if me.success:
      return me.P_chi_square
    else: 
      return -0.1

  def getparam0(me):
    return float(me.params[0])

  def getparam0error(me):
    if me.success:
      return math.sqrt(me.cov[0][0])
    else: 
      return 0.0

  def getparam1(me):
    return float(me.params[1])

  def getparam1error(me):
    if me.success:
      return math.sqrt(me.cov[1][1])
    else: 
      return 0.0

  def getparam2(me):
    return float(me.params[2])

  def getparam2error(me):
    if me.success:
      return math.sqrt(me.cov[2][2])
    else: 
      return 0.0

  def getparam3(me):
    return float(me.params[3])

  def getparam3error(me):
    if me.success:
      return math.sqrt(me.cov[3][3])
    else: 
      return 0.0

  def getparam4(me):
    return float(me.params[4])

  def getparam4error(me):
    if me.success:
      return math.sqrt(me.cov[4][4])
    else: 
      return 0.0

  def getparam5(me):
    return float(me.params[5])

  def getparam5error(me):
    if me.success:
      return math.sqrt(me.cov[5][5])
    else: 
      return 0.0

  def getparam6(me):
    return float(me.params[6])

  def getparam6error(me):
    if me.success:
      return math.sqrt(me.cov[6][6])
    else: 
      return 0.0

  def getparam15(me):
    return float(me.params[15])

  def fit_result_for_plot(me):
    retval = 'Fit Result \n'
    for i in range(me.N_params): 
      param_str = '% 12s:  % 6.4g'%(
        me.param_names[i], me.params[i]
      )
      if me.cov.any():
        param_str += ' +- % 6.4g'%math.sqrt(me.cov[i][i])
      if len(me.initial_params)>0: 
        param_str += '  (initial: %-6.4g)'%(me.initial_params[i],)
      retval += ' '+param_str+'\n'
    retval += '  DoF:  %d bins - %d parameters = %d\n'%(
      me.N_points, 
      me.N_params, 
      me.N_dof
    )
    if me.chi_square!=None:
      retval += '  chi^2:  %f\n'%(me.chi_square,)
      retval += '  reduced chi^2:  %f\n'%(me.reduced_chi_square)
      if me.P_chi_square>1e-05: retval += '  P: %f\n'%me.P_chi_square
      else: retval += '  P: %6.4g'%me.P_chi_square
    return retval
  
  def __str__(me):
    retval = '--------------------------------------------------------------------------\n'
    retval += '  Fit status: %d (%s)\n  %s'%(
      me.status, 
      'success' if me.success else 'FAILED!', 
      me.status_description.replace('\n','\n  ')
    )
    retval += '\n  parameters:\n'
    for i in range(me.N_params): 
      #param_str = '% 12s:  % 6.4g +- % 6.4g'%(
      #  me.param_names[i], me.params[i], math.sqrt(me.cov[i][i])
      #)
      param_str = '% 12s:  % 6.4g'%(
        me.param_names[i], me.params[i]
      )
      if me.cov.any():
        param_str += ' +- % 6.4g'%math.sqrt(me.cov[i][i])
      if len(me.initial_params)>0: 
        param_str += '  (initial: %-6.4g)'%(me.initial_params[i],)
      retval += ' '+param_str+'\n'  
    retval += '  DoF:  %d bins - %d parameters = %d\n'%(
      me.N_points, 
      me.N_params, 
      me.N_dof
    )
    if me.chi_square!=None:
      retval += '  chi^2:  %f\n'%(me.chi_square,)
      retval += '  reduced chi^2:  %f\n'%(me.reduced_chi_square)
      if me.P_chi_square>1e-05: retval += '  P: %f\n'%me.P_chi_square
      else: retval += '  P: %6.4g\n'%me.P_chi_square
    #if me.cov.any():
    if False:
      retval += '  covariance:\n'
      for row in me.cov:
        row_str = '    '
        for elem in row:
          #row_str += '%6.4g    '%(elem,)
          #row_str += '% 4.3g    '%(elem,)
          num_str = '%f'%(elem,)
          if len(num_str)>7: num_str = '%4.3g'%(elem,)
          row_str += '% 8s    '%(num_str,)
        row_str.rstrip('    ') # for loop adds whitespace at end
        retval += row_str + '\n'
    
    if len(me.error_message)>0:
      retval += '  Error message(s):\n'
      retval += me.error_message.replace('\n','\n    ')
    retval += '--------------------------------------------------------------------------'

    from matplotlib import rcParams
    rcParams['font.family'] = 'serif'
    rcParams['font.size'] = 10

    if False:
      A = me.correlation 
      X = me.param_names
      Y = me.param_names

      m = len(Y)
      n = len(X)
     
      plt.figure(figsize=(n + 1, m + 1)) 
      for krow, row in enumerate(A):
          plt.text(5, 10*krow + 15, Y[krow],    #plt.text(x,y position)
                   horizontalalignment='center',
                   verticalalignment='center')
          for kcol, num in enumerate(row):
              if krow == 0:
                  plt.text(10*kcol + 15, 5, X[kcol],
                           horizontalalignment='center',
                           verticalalignment='center')
              num = '{:.2e}'.format(num)
              plt.text(10*kcol + 15, 10*krow + 15, num,
                       horizontalalignment='center',
                       verticalalignment='center')

      plt.axis([0, 10*(n + 1), 10*(m + 1), 0])
      plt.xticks(np.linspace(0, 10*(n + 1), n + 2), []) # start, stop, number of lines
      plt.yticks(np.linspace(0, 10*(m + 1), m + 2), [])
      plt.grid(linestyle="solid")
      plt.savefig("correlation_table.png", dpi=200)

    return retval

# ------------------------------------------------------------------------------------------------------


# Create a class around scipy.least_squares in order to do preliminary fits which allow for bounds and fixed params

class PFitResult(object): # P for Preliminary
  def __init__(me, fitresult, p0, param_names=None, shrink=False):
    # grab stuff from leastsq fit, save other arguments

    me.params = fitresult.x
    me.residuals = fitresult.fun
    me.jac = fitresult.jac
    #me.grad, me.optimality, me.active_mask, me.nfev, me.njev, 
    me.status = fitresult.status
    me.message = fitresult.message
    me.success = fitresult.success

    me.initial_params = p0
    me.param_names = param_names
    me.error_message = ''
    
    # set defaults
    me.chi_square = None
    me.param_errors = None
    me.reduced_chi_square = None
    me.P_chi_square = None
    me.N_points = None
    me.N_params = None
    me.N_dof = None
    me.correlation = None
    #me.success = False

    me.check_and_compute()
      

    if len(me.error_message)>0:
      print ('Error message(s):\n'+me.error_message)
  
  def check_and_compute(me):
    if me.param_names==None: 
      try:
        me.param_names = tuple([ 'p'+str(i) for i in range(len(me.initial_params)) ])
      except:
        me.error_message += 'Failed to generate automatic parameter names..?\n'
        return False
    try:
      me.N_points = len(me.residuals)
      me.N_params = len(me.params)
      me.N_dof = me.N_points - me.N_params
    except: 
      me.error_message += 'Could not count points, parameters, or degrees of freedom.  (Try\n'
      me.error_message += 'checking fvec or params.)\n'
      return False
    try: 
      J = me.jac
      me.cov_frac = np.linalg.inv(J.T.dot(J))
      me.cov = me.cov_frac * np.sum(me.residuals**2) / me.N_dof # check that this formula is correct

      v = np.sqrt(np.diag(me.cov))
      outer_v = np.outer(v, v)
      me.correlation = me.cov / outer_v
      me.correlation[me.cov == 0] = 0
    except:
      me.error_message += 'Fractional covariance matrix ill-formed!\n'
      return False
    try:
      me.chi_square = sum(me.residuals**2)
      me.reduced_chi_square = me.chi_square/me.N_dof
      me.P_chi_square = 1.-scipy.stats.chi2.cdf(me.chi_square,me.N_dof)
    except: 
      me.error_message += 'Failed to compute chi_square!  (Check residuals.)\n'
      return False
    try:
      me.param_errors = [ math.sqrt(me.cov[i][i]) for i in range(len(me.cov)) ]
    except:
      me.error_message += 'Failed to compute parameter errors! (Check fvec and cov.)\n'
      return False
    if me.status not in (1,2,3,4): 
      me.error_message += 'FAILURE indicated by status '+str(me.status)+'\n'
      return False

  def failed(me):
    if me.success:
      return False
    else:
      return True

  def getgoodness(me):
    if me.success:
      return me.reduced_chi_square
    else: 
      return 0.0

  def getndf(me):
    return me.N_dof

  def getpvalue(me):
    if me.success:
      return me.P_chi_square
    else: 
      return -0.1

    # A, r, phi_a, tau_mu, 
    #tau_cbo, omega_cbo, A_nx11, phi_nx11, 
    #tau_y, omega_vw, A_ny22, phi_ny22, 
    #A_nx22, phi_nx22, 
    #K, 
    #A_ny11, phi_ny11, omega_y, 
    #A_phix11, phi_phix11, 
    #A_ax11, phi_ax11   

  def getparamN(me, param_index):
    return float(me.params[param_index])

  def getparamNerror(me, param_index):
    if me.success:
      return math.sqrt(me.cov[param_index][param_index])
    else: 
      return 0.0
 

  def fit_result_for_plot(me):
    retval = 'Fit Result \n'
    for i in range(me.N_params): 
      param_str = '% 12s:  % 6.4g'%(
        me.param_names[i], me.params[i]
      )
      if me.cov.any():
        param_str += ' +- % 6.4g'%math.sqrt(me.cov[i][i])
      if len(me.initial_params)>0: 
        param_str += '  (initial: %-6.4g)'%(me.initial_params[i],)
      retval += ' '+param_str+'\n'
    retval += '  DoF:  %d bins - %d parameters = %d\n'%(
      me.N_points, 
      me.N_params, 
      me.N_dof
    )
    if me.chi_square!=None:
      retval += '  chi^2:  %f\n'%(me.chi_square,)
      retval += '  reduced chi^2:  %f\n'%(me.reduced_chi_square)
      if me.P_chi_square>1e-05: retval += '  P: %f\n'%me.P_chi_square
      else: retval += '  P: %6.4g'%me.P_chi_square
    return retval
  
  def __str__(me):
    retval = '--------------------------------------------------------------------------\n'
    retval += 'Prelimary Fit Worked \n'
    if me.chi_square!=None:
      retval += '  chi^2:  %f\n'%(me.chi_square,)
      retval += '  reduced chi^2:  %f\n'%(me.reduced_chi_square)
      if me.P_chi_square>1e-05: retval += '  P: %f\n'%me.P_chi_square
      else: retval += '  P: %6.4g\n'%me.P_chi_square
    retval += '\n  parameters:\n'
    for i in range(me.N_params): 
      param_str = '% 12s:  % 6.4g'%(
        me.param_names[i], me.params[i]
      )
      if me.cov.any():
        param_str += ' +- % 6.4g'%math.sqrt(me.cov[i][i])
      if len(me.initial_params)>0: 
        param_str += '  (initial: %-6.4g)'%(me.initial_params[i],)
      retval += ' '+param_str+'\n'  
    retval += '  DoF:  %d bins - %d parameters = %d\n'%(
      me.N_points, 
      me.N_params, 
      me.N_dof
    )
    if len(me.error_message)>0:
      retval += '  Error message(s):\n'
      retval += me.error_message.replace('\n','\n    ')
    retval += '--------------------------------------------------------------------------'
    retval += '--------------------------------------------------------------------------'

    return retval
    





