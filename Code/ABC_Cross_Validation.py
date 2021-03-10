from scipy import stats
from scipy.signal import correlate2d
import matplotlib.pyplot as plt

import numpy as np
import Plotting
from random import randint

import ABC,Models


def LOO_CV_abc_rejection(n_obs:int,x_obs:[[float]],y_obs:[[float]],fitting_model:Models.Model,priors:["stats.Distribution"],sampling_details:dict,summary_stats=None) -> float:

    error=0
    for i in range(len(y_obs)):
        removed=(x_obs[i],y_obs[i])
        print("{}/{} ([{}],[{}]) ".format(i,len(y_obs),",".join([str(x) for x in removed[0]]),",".join([str(x) for x in removed[1]])))

        x_minus=x_obs[:i]+x_obs[i+1:]
        y_minus=y_obs[:i]+y_obs[i+1:]

        n_minus=n_obs-1

        model_minus=fitting_model.copy([1 for _ in range(len(priors))])
        model_minus.x_obs=x_minus
        model_minus.n_obs=n_minus

        fitted_model,_=ABC.abc_rejcection(n_minus,y_minus,model_minus,priors,sampling_details,summary_stats=summary_stats,show_plots=False)

        error=__record_results(fitting_model,fitted_model,removed,error)

    return error

def LOO_CV_abc_mcmc(n_obs:int,x_obs:[[float]],y_obs:[[float]],fitting_model:Models.Model,priors:["stats.Distribution"],
        perturbance_kernels:["func"],acceptance_kernel:"func",scaling_factor:int,
        chain_length=10000,summary_stats=None) -> float:

    error=0
    for i in range(len(y_obs)):
        removed=(x_obs[i],y_obs[i])
        print("{}/{} ([{}],[{}]) ".format(i,len(y_obs),",".join([str(x) for x in removed[0]]),",".join([str(x) for x in removed[1]])))

        x_minus=x_obs[:i]+x_obs[i+1:]
        y_minus=y_obs[:i]+y_obs[i+1:]

        n_minus=n_obs-1

        model_minus=fitting_model.copy([1 for _ in range(len(priors))])
        model_minus.x_obs=x_minus
        model_minus.n_obs=n_minus

        fitted_model,_=ABC.abc_mcmc(n_obs=n_minus,y_obs=y_minus,fitting_model=model_minus,priors=priors,chain_length=chain_length,perturbance_kernels=perturbance_kernels,acceptance_kernel=acceptance_kernel,scaling_factor=scaling_factor,summary_stats=summary_stats,show_plots=False)

        error=__record_results(fitting_model,fitted_model,removed,error)

    return error

def LOO_CV_abc_smc(n_obs:int,x_obs:[[float]],y_obs:[[float]],fitting_model:Models.Model,priors:["stats.Distribution"],
        perturbance_kernels:["func"],perturbance_kernel_probability:"[function]",acceptance_kernel:"func",scaling_factors:[float],
        num_steps=10,sample_size=100,summary_stats=None) -> float:

    error=0
    for i in range(len(y_obs)):
        removed=(x_obs[i],y_obs[i])
        print("{}/{} ([{}],[{}]) ".format(i,len(y_obs),",".join([str(x) for x in removed[0]]),",".join([str(x) for x in removed[1]])))

        x_minus=x_obs[:i]+x_obs[i+1:]
        y_minus=y_obs[:i]+y_obs[i+1:]

        n_minus=n_obs-1

        model_minus=fitting_model.copy([1 for _ in range(len(priors))])
        model_minus.x_obs=x_minus
        model_minus.n_obs=n_minus

        fitted_model,_=ABC.abc_smc(n_obs=n_minus,y_obs=y_minus,fitting_model=model_minus,priors=priors,
            num_steps=num_steps,sample_size=sample_size,scaling_factors=scaling_factors,
            perturbance_kernels=perturbance_kernels,perturbance_kernel_probability=perturbance_kernel_probability,
            acceptance_kernel=ABC.gaussian_kernel,show_plots=False)

        error=__record_results(fitting_model,fitted_model,removed,error)

    return error

def __record_results(fitting_model,fitted_model,removed,error) -> float:
    if type(fitting_model) is Models.SIRModel:
        # SIRModel._calc requires a list rather than a single value
        fitted_val=fitted_model._calc([removed[0]])[0]
        print("Actual - ",fitted_val,"\nPredicted - ",removed[1])
        add_error=ABC.l2_norm(fitted_val,removed[1])
        error+=ABC.l2_norm(fitted_val,removed[1])
        print("New Error - {:.3f}\nTotal so far - {:,.0f}\n".format(add_error,error))
    else:
        fitted_val=fitted_model._calc(removed[0])
        print("Actual - ",fitted_val,"\nPredicted - ",removed[1])
        add_error=ABC.l2_norm(fitted_val,removed[1])
        error+=ABC.l2_norm(fitted_val,removed[1])
        print("New Error - {:.3f}\nTotal so far - {:,.0f}\n".format(add_error,error))

    return error
