from scipy import stats
from scipy.signal import correlate2d
import matplotlib.pyplot as plt

import numpy as np
import Plotting
from random import randint

import ABC,Models

"""
    HELPER METHODS
"""
def __record_results(fitting_model,fitted_model,removed,error) -> float:
    if type(fitting_model) is Models.SIRModel: # SIRModel._calc requires a list rather than a single value
        fitted_val=fitted_model._calc([removed[0]])[0]
    else:
        fitted_val=fitted_model._calc(removed[0])

    add_error=ABC.l2_norm(fitted_val,removed[1])
    error+=ABC.l2_norm(fitted_val,removed[1])

    print("Predicted - ",fitted_val,". Actual - ",removed[1],". Error - {:.3f}".format(add_error),sep="")
    # print("New Error - {:.3f}\nTotal so far - {:,.0f}\n".format(add_error,error))

    return error

"""
    CROSS-VALIDATION
"""
def LOO_CV_abc_rejection(n_obs:int,x_obs:[[float]],y_obs:[[float]],fitting_model:Models.Model,priors:["stats.Distribution"],sampling_details:dict,summary_stats=None) -> float:

    error=0
    for i in range(len(y_obs)):
        if (i==0) and (type(fitting_model) is Models.LinearModel): continue
        removed=(x_obs[i],y_obs[i])
        print("{}/{}. ".format(i,len(y_obs)),end="")

        x_minus=x_obs[:i]+x_obs[i+1:]
        y_minus=y_obs[:i]+y_obs[i+1:]

        n_minus=n_obs-1

        model_minus=fitting_model.copy([1 for _ in range(len(priors))])
        model_minus.x_obs=x_minus
        model_minus.n_obs=n_minus

        fitted_model,_=ABC.abc_rejcection(n_minus,y_minus,model_minus,priors,sampling_details,summary_stats=summary_stats,show_plots=False,printing=False)

        error=__record_results(fitting_model,fitted_model,removed,error)

    return error

def LOO_CV_abc_mcmc(n_obs:int,x_obs:[[float]],y_obs:[[float]],fitting_model:Models.Model,priors:["stats.Distribution"],
        perturbance_kernels:["func"],acceptance_kernel:"func",scaling_factor:int,
        chain_length=10000,summary_stats=None) -> float:

    error=0
    for i in range(len(y_obs)):
        if (i==0) and (type(fitting_model) is Models.LinearModel): continue
        removed=(x_obs[i],y_obs[i])
        print("{}/{}. ".format(i,len(y_obs)),end="")

        x_minus=x_obs[:i]+x_obs[i+1:]
        y_minus=y_obs[:i]+y_obs[i+1:]

        n_minus=n_obs-1

        model_minus=fitting_model.copy([1 for _ in range(len(priors))])
        model_minus.x_obs=x_minus
        model_minus.n_obs=n_minus

        fitted_model,_=ABC.abc_mcmc(n_obs=n_minus,y_obs=y_minus,fitting_model=model_minus,priors=priors,chain_length=chain_length,perturbance_kernels=perturbance_kernels,acceptance_kernel=acceptance_kernel,scaling_factor=scaling_factor,summary_stats=summary_stats,show_plots=False,printing=False)

        error=__record_results(fitting_model,fitted_model,removed,error)

    return error

def LOO_CV_abc_smc(n_obs:int,x_obs:[[float]],y_obs:[[float]],fitting_model:Models.Model,priors:["stats.Distribution"],
        perturbance_kernels:["func"],perturbance_kernel_probability:"[function]",acceptance_kernel:"func",scaling_factors:[float],
        num_steps=10,sample_size=100,summary_stats=None) -> float:

    error=0
    for i in range(len(y_obs)):
        if (i==0) and (type(fitting_model) is Models.LinearModel): continue
        removed=(x_obs[i],y_obs[i])
        print("{}/{}. ".format(i,len(y_obs)),end="")

        x_minus=x_obs[:i]+x_obs[i+1:]
        y_minus=y_obs[:i]+y_obs[i+1:]

        n_minus=n_obs-1

        model_minus=fitting_model.copy([1 for _ in range(len(priors))])
        model_minus.x_obs=x_minus
        model_minus.n_obs=n_minus

        fitted_model,_=ABC.abc_smc(n_obs=n_minus,y_obs=y_minus,fitting_model=model_minus,priors=priors,
            num_steps=num_steps,sample_size=sample_size,scaling_factors=scaling_factors,
            perturbance_kernels=perturbance_kernels,perturbance_kernel_probability=perturbance_kernel_probability,
            acceptance_kernel=ABC.gaussian_kernel,show_plots=False,printing=False)

        error=__record_results(fitting_model,fitted_model,removed,error)

    return error

def LOO_CV_adaptive_abc_smc(n_obs:int,x_obs:[[float]],y_obs:[[float]],fitting_model:Models.Model,priors:["stats.Distribution"],
        max_steps:int,max_simulations:int,alpha:float,terminal_scaling_factor=None,sample_size=100,summary_stats=None) -> float:

    error=0
    for i in range(len(y_obs)):
        if (i==0) and (type(fitting_model) is Models.LinearModel): continue
        removed=(x_obs[i],y_obs[i])
        print("{}/{}. ".format(i,len(y_obs)),end="")

        x_minus=x_obs[:i]+x_obs[i+1:]
        y_minus=y_obs[:i]+y_obs[i+1:]

        n_minus=n_obs-1

        model_minus=fitting_model.copy([1 for _ in range(len(priors))])
        model_minus.x_obs=x_minus
        model_minus.n_obs=n_minus

        fitted_model,_=ABC.adaptive_abc_smc(n_obs=n_minus,y_obs=y_minus,fitting_model=model_minus,priors=priors,
            max_steps=max_steps,sample_size=sample_size,alpha=alpha,max_simulations=max_simulations,terminal_scaling_factor=terminal_scaling_factor,
            acceptance_kernel=ABC.uniform_kernel,show_plots=False,printing=False)

        error=__record_results(fitting_model,fitted_model,removed,error)

    return error
"""
    CV SEMI-AUTO
"""
def LOO_CV_abc_rejection_semi_auto(n_obs:int,x_obs:[[float]],y_obs:[[float]],fitting_model:Models.Model,priors:["stats.Distribution"],sampling_details:dict,pilot_distance_measure=ABC.l2_norm,n_pilot_samples=10000,n_pilot_acc=1000,n_params_sample_size=500,summary_stats=None) -> float:

    error=0
    for i in range(len(y_obs)):
        if (i==0) and (type(fitting_model) is Models.LinearModel): continue
        removed=(x_obs[i],y_obs[i])
        print("{}/{}. ".format(i,len(y_obs)),end="")

        x_minus=x_obs[:i]+x_obs[i+1:]
        y_minus=y_obs[:i]+y_obs[i+1:]

        n_minus=n_obs-1

        model_minus=fitting_model.copy([1 for _ in range(len(priors))])
        model_minus.x_obs=x_minus
        model_minus.n_obs=n_minus

        print("*",end="")
        summary_stats_minus,coefs=ABC.abc_semi_auto(n_obs=n_minus,y_obs=y_minus,fitting_model=model_minus,priors=priors,distance_measure=pilot_distance_measure,n_pilot_samples=n_pilot_samples,n_pilot_acc=n_pilot_acc,n_params_sample_size=n_params_sample_size,summary_stats=summary_stats,printing=False)
        print("*",end="")
        fitted_model,_=ABC.abc_rejcection(n_minus,y_minus,model_minus,priors,sampling_details,summary_stats=summary_stats_minus,show_plots=False,printing=False)
        print("* ",end="")

        error=__record_results(fitting_model,fitted_model,removed,error)

    return error

def LOO_CV_abc_mcmc_semi_auto(n_obs:int,x_obs:[[float]],y_obs:[[float]],fitting_model:Models.Model,priors:["stats.Distribution"],
        perturbance_kernels:["func"],acceptance_kernel:"func",scaling_factor:int,chain_length=10000,
        pilot_distance_measure=ABC.l2_norm,n_pilot_samples=10000,n_pilot_acc=1000,n_params_sample_size=500,
        summary_stats=None) -> float:

    error=0
    for i in range(len(y_obs)):
        if (i==0) and (type(fitting_model) is Models.LinearModel): continue
        removed=(x_obs[i],y_obs[i])
        print("{}/{}. ".format(i,len(y_obs)),end="")

        x_minus=x_obs[:i]+x_obs[i+1:]
        y_minus=y_obs[:i]+y_obs[i+1:]

        n_minus=n_obs-1

        model_minus=fitting_model.copy([1 for _ in range(len(priors))])
        model_minus.x_obs=x_minus
        model_minus.n_obs=n_minus

        print("*",end="")
        summary_stats_minus,coefs=ABC.abc_semi_auto(n_obs=n_minus,y_obs=y_minus,fitting_model=model_minus,priors=priors,distance_measure=pilot_distance_measure,n_pilot_samples=n_pilot_samples,n_pilot_acc=n_pilot_acc,n_params_sample_size=n_params_sample_size,summary_stats=summary_stats,printing=False)
        print("*",end="")
        fitted_model,_=ABC.abc_mcmc(n_obs=n_minus,y_obs=y_minus,fitting_model=model_minus,priors=priors,chain_length=chain_length,perturbance_kernels=perturbance_kernels,acceptance_kernel=acceptance_kernel,scaling_factor=scaling_factor,summary_stats=summary_stats,show_plots=False,printing=False)
        print("* ",end="")

        error=__record_results(fitting_model,fitted_model,removed,error)

    return error

def LOO_CV_abc_smc_semi_auto(n_obs:int,x_obs:[[float]],y_obs:[[float]],fitting_model:Models.Model,priors:["stats.Distribution"],
        perturbance_kernels:["func"],perturbance_kernel_probability:"[function]",acceptance_kernel:"func",scaling_factors:[float],
        num_steps=10,sample_size=100,
        pilot_distance_measure=ABC.l2_norm,n_pilot_samples=10000,n_pilot_acc=1000,n_params_sample_size=500,
        summary_stats=None) -> float:

    error=0
    for i in range(len(y_obs)):
        if (i==0) and (type(fitting_model) is Models.LinearModel): continue
        removed=(x_obs[i],y_obs[i])
        print("{}/{}. ".format(i,len(y_obs)),end="")

        x_minus=x_obs[:i]+x_obs[i+1:]
        y_minus=y_obs[:i]+y_obs[i+1:]

        n_minus=n_obs-1

        model_minus=fitting_model.copy([1 for _ in range(len(priors))])
        model_minus.x_obs=x_minus
        model_minus.n_obs=n_minus

        print("*",end="")
        summary_stats_minus,coefs=ABC.abc_semi_auto(n_obs=n_minus,y_obs=y_minus,fitting_model=model_minus,priors=priors,distance_measure=pilot_distance_measure,n_pilot_samples=n_pilot_samples,n_pilot_acc=n_pilot_acc,n_params_sample_size=n_params_sample_size,summary_stats=summary_stats,printing=False)
        print("*",end="")
        fitted_model,_=ABC.abc_smc(n_obs=n_minus,y_obs=y_minus,fitting_model=model_minus,priors=priors,
            num_steps=num_steps,sample_size=sample_size,scaling_factors=scaling_factors,
            perturbance_kernels=perturbance_kernels,perturbance_kernel_probability=perturbance_kernel_probability,
            acceptance_kernel=ABC.gaussian_kernel,show_plots=False,printing=True)
        print("* ",end="")

        error=__record_results(fitting_model,fitted_model,removed,error)

    return error
