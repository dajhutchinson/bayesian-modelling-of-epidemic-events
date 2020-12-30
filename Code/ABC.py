from Models import Model
from scipy import stats
import matplotlib.pyplot as plt

import numpy as np
import Plotting

"""
    KERNELS
"""
def abstract_kernel(x:float,epsilon:float) -> bool:
    """
    DESCRIPTION
    determine whether to accept an observation based on how far it is from other observations.

    PARAMETERS
    x (float) - value for comparision (typically distance between two observations)
    epsilon (float) - scaling factor

    RETURNS
    bool - whether to accept
    """
    return True

def uniform_kernel(x:float,epsilon:float) -> bool:
    return abs(x)<=epsilon

def epanechnikov_kernel(x:float,epsilon:float) -> bool:
    if (abs(x)>epsilon): return False
    ep_val=(1-(x/epsilon)**2)#*(1/epsilon)*(3/4) # prob to accept
    x=stats.uniform(0,1).rvs(1)[0] # sample from U[0,1]
    return (x<=ep_val)

def gaussian_kernel(x:float,epsilon:float) -> bool:
    gaus_val=np.exp(-(1/2)*((x/epsilon)**2)) #*(1/np.sqrt(2*np.pi*(epsilon**2))) # prob to accept
    x=stats.uniform(0,1).rvs(1)[0] # sample from U[0,1]
    return (x<=gaus_val)

"""
    DISTANCE MEASURES
"""
def l2_norm(s_t:(float),s_obs:(float)) -> float:
    return sum([(x-y)**2 for (x,y) in zip(s_t,s_obs)])**.5

"""
    REJECTION SAMPLING METHODS
"""

def __sampling_stage_fixed_number(DESIRED_SAMPLE_SIZE:int,EPSILON:float,KERNEL:"func",PRIORS:["stats.Distribution"],
    s_obs:[float],model_t:Model,summary_stats:["function"],pct_matches=.8) -> ([[float]],[[float]],[[float]]):
    """
    DESCRIPTION
    keep generating parameter values and observing the equiv models until a sufficient number of `good` parameter values have been found.

    PARAMETERS
    DESIRED_SAMPLE_SIZE (int) - number of `good` samples to wait until
    EPSILON (int) - scale parameter for `KERNEL`
    KERNEL (func) - one of the kernels defined above. determine which parameters are good or not.
    PRIORS ([stats.Distribution]) - prior distribution for parameter values (one per parameter)
    s_obs ([float]) - summary statistic values for observations from true model.
    summary_stat ([func]) - functions used to determine each summary statistic.

    OPTIONAL PARAMETERS
    pct_matches (float) - percentage of sumamry statistics which need to be accepted by the kernels for the parameters to be accepted.

    RETURNS
    [[float]] - accepted parameter values
    [[float]] - observations made when using accepted parameter values
    [[float]] - summary statistic values for accepted parameter observations
    """
    if (pct_matches<0 or pct_matches>1): raise ValueError("`pct_matches` must be in [0,1].")

    ACCEPTED_PARAMS=[]
    ACCEPTED_OBS=[]
    ACCEPTED_SUMMARY_VALS=[]
    i=0
    while (len(ACCEPTED_OBS)<DESIRED_SAMPLE_SIZE):
        # sample parameters
        theta_t=[pi_i.rvs(1)[0] for pi_i in PRIORS]

        # observe theorised model
        model_t.update_params(theta_t)
        y_t=model_t.observe()
        s_t=[s(y_t) for s in summary_stats]

        # accept-reject
        norm_vals=[l2_norm(s_t_i,s_obs_i) for (s_t_i,s_obs_i) in zip(s_t,s_obs)]
        if (np.mean([KERNEL(v,EPSILON) for v in norm_vals])>=pct_matches): # if at least `pct_matches`% of observations satisfy kernel
            ACCEPTED_PARAMS.append(theta_t)
            ACCEPTED_OBS.append(y_t)
            ACCEPTED_SUMMARY_VALS.append(s_t)

        i+=1
        print("({:,}) {:,}/{:,}".format(i,len(ACCEPTED_PARAMS),DESIRED_SAMPLE_SIZE),end="\r") # update user on sampling process
    print("\n")

    return ACCEPTED_PARAMS,ACCEPTED_OBS,ACCEPTED_SUMMARY_VALS

def __sampling_stage_best_samples(NUM_RUNS:int,SAMPLE_SIZE:int,PRIORS:["stats.Distribution"],
    s_obs:[float],model_t:Model,summary_stats:["function"]) -> ([[float]],[[float]],[[float]]):
    """
    DESCRIPTION
    perform `NUM_RUNS` samples and return the parameters values associated to the best `SAMPLE_SIZE`.

    PARAMETERS
    NUM_RUNS (int) - number of samples to make
    SAMPLE_SIZE (int) - The best n set of parameters to return
    PRIORS ([stats.Distribution]) - prior distribution for parameter values (one per parameter)
    s_obs ([float]) - summary statistic values for observations from true model.
    summary_stat ([func]) - functions used to determine each summary statistic.

    RETURNS
    [[float]] - accepted parameter values
    [[float]] - observations made when using accepted parameter values
    [[float]] - summary statistic values for accepted parameter observations
    """

    SAMPLES=[None for _ in range(SAMPLE_SIZE)]

    for i in range(NUM_RUNS):
        # sample parameters
        theta_t=[pi_i.rvs(1)[0] for pi_i in PRIORS]

        # observe theorised model
        model_t.update_params(theta_t)
        y_t=model_t.observe()
        s_t=[s(y_t) for s in summary_stats]

        norm_vals=[l2_norm(s_t_i,s_obs_i) for (s_t_i,s_obs_i) in zip(s_t,s_obs)]
        summarised_norm_val=np.mean(norm_vals) # used to order samples by quality (TODO)

        for j in range(max(0,SAMPLE_SIZE-i-1,SAMPLE_SIZE)):

            if (SAMPLES[j]==None) or (SAMPLES[j][0]>summarised_norm_val):
                if (j>0): SAMPLES[j-1]=SAMPLES[j]
                SAMPLES[j]=(summarised_norm_val,(theta_t,y_t,s_t))

        print("({:,}/{:,})".format(i,NUM_RUNS),end="\r") # update user on sampling process

    print("\n")
    SAMPLES=[x[1] for x in SAMPLES] # remove norm value
    ACCEPTED_PARAMS=[x[0] for x in SAMPLES]
    ACCEPTED_OBS=[x[1] for x in SAMPLES]
    ACCEPTED_SUMMARY_VALS=[x[2] for x in SAMPLES]

    return ACCEPTED_PARAMS,ACCEPTED_OBS,ACCEPTED_SUMMARY_VALS

"""
    ABC
"""

def abc_rejcection(n_obs:int,y_obs:[[float]],fitting_model:Model,priors:["stats.Distribution"],sampling_details:dict,summary_stats=None) -> Model:
    """
    DESCRIPTION
    Rejction Sampling version of Approximate Bayesian Computation for the generative models defined in `Models.py`.

    PARAMETERS
    n_obs (int) - Number of observations available.
    y_obs ([[float]]) - Observations from true model.
    fitting_model (Model) - Model the algorithm will aim to fit to observations.
    priors (["stats.Distribution"]) - Priors for the value of parameters of `fitting_model`.
    sampling_details - specification of how sampling should be done (see README.md)

    OPTIONAL PARAMETERS
    summary_stats ([function]) - functions which summarise `y_obs` and the observations of `fitting_model` in some way. (default=group by dimension)

    RETURNS
    Model - fitted model with best parameters
    """

    if (type(y_obs)!=list): raise TypeError("`y_obs` must be a list (not {})".format(type(y_obs)))
    if (len(y_obs)!=n_obs): raise ValueError("Wrong number of observations supplied (len(y_obs)!=n_obs) ({}!={})".format(len(y_obs),n_obs))
    if (len(priors)!=fitting_model.n_params): raise ValueError("Wrong number of priors given (exp fitting_model.n_params={})".format(fitting_model.n_params))

    group_dim = lambda ys,i: [y[i] for y in ys]
    summary_stats=summary_stats if (summary_stats) else ([(lambda ys:group_dim(ys,i)) for i in range(len(y_obs[0]))])
    s_obs=[s(y_obs) for s in summary_stats]

    # sampling step
    if ("sampling_method" not in sampling_details): raise Exception("Insufficient details provided in `sampling_details` - missing `sampling_method`")

    elif (sampling_details["sampling_method"]=="fixed_number"):
        if any([x not in sampling_details for x in ["sample_size","scaling_factor","kernel_func"]]): raise Exception("`sampling_details` missing key(s) - expecting `sample_size`,`scaling_factor` and `kernel_func`")
        sample_size=sampling_details["sample_size"]
        epsilon=sampling_details["scaling_factor"]
        kernel=sampling_details["kernel_func"]
        ACCEPTED_PARAMS,ACCEPTED_OBS,ACCEPTED_SUMMARY_VALS=__sampling_stage_fixed_number(sample_size,epsilon,kernel,PRIORS=priors,s_obs=s_obs,model_t=fitting_model,summary_stats=summary_stats)

    elif (sampling_details["sampling_method"]=="best"):
        if any([x not in sampling_details for x in ["sample_size","num_runs"]]): raise Exception("`sampling_details` missing key(s) - expecting `num_runs` and `sample_size`")
        num_runs=sampling_details["num_runs"]
        sample_size=sampling_details["sample_size"]
        ACCEPTED_PARAMS,ACCEPTED_OBS,ACCEPTED_SUMMARY_VALS=__sampling_stage_best_samples(num_runs,sample_size,PRIORS=priors,s_obs=s_obs,model_t=fitting_model,summary_stats=summary_stats)

    # best estimate of model
    theta_hat=[np.mean([p[i] for p in ACCEPTED_PARAMS]) for i in range(fitting_model.n_params)]
    model_hat=fitting_model.copy(theta_hat)
    s_hat=[s(model_hat.observe()) for s in summary_stats]

    # plot results
    n_simple_ss=sum(len(s)==1 for s in ACCEPTED_SUMMARY_VALS[0]) # number of summary stats which map to a single dimension
    n_rows=max([1,np.lcm(fitting_model.n_params,n_simple_ss)])

    fig=plt.figure(constrained_layout=True)
    gs=fig.add_gridspec(n_rows,3) # 3 columns
    ax=fig.add_subplot(gs[:,-1])
    Plotting.plot_accepted_observations(ax,fitting_model.n_obs,y_obs,ACCEPTED_OBS,model_hat)

    row_step=n_rows//fitting_model.n_params
    for i in range(fitting_model.n_params):
        name="Theta_{}".format(i)
        ax=fig.add_subplot(gs[i*row_step:(i+1)*row_step,0])
        accepted_vals=[x[i] for x in ACCEPTED_PARAMS]
        Plotting.plot_parameter_posterior(ax,name,accepted_parameter=accepted_vals,predicted_val=theta_hat[i],prior=priors[i])

    row=0
    row_step=n_rows//n_simple_ss
    for i in range(len(summary_stats)):
        if (len(ACCEPTED_SUMMARY_VALS[0][i])==1):
            name="s_{}".format(i)
            ax=fig.add_subplot(gs[row*row_step:(row+1)*row_step,1])
            row+=1
            accepted_vals=[s[i][0] for s in ACCEPTED_SUMMARY_VALS]
            Plotting.plot_summary_stats(ax,name,accepted_s=accepted_vals,s_obs=s_obs[i],s_hat=s_hat[i])

    plt.get_current_fig_manager().window.state("zoomed")
    plt.show()

    return model_hat

def abc_mcmc(n_obs:int,y_obs:[[float]],
    fitting_model:Model,priors:["stats.Distribution"],
    chain_length:int,perturbance_kernels:"[function]",acceptance_kernel:"function",scaling_factor:float,
    summary_stats=None) -> Model:
    """
    DESCRIPTION
    Markov Chain Monte-Carlo Sampling version of Approximate Bayesian Computation for the generative models defined in `Models.py`.

    PARAMETERS
    n_obs (int) - Number of observations available.
    y_obs ([[float]]) - Observations from true model.
    fitting_model (Model) - Model the algorithm will aim to fit to observations.
    priors (["stats.Distribution"]) - Priors for the value of parameters of `fitting_model`.
    chain_length (int) - Length of markov chain to allow.
    perturbance_kernels ([function]) - Functions for varying parameters each monte-carlo steps.
    acceptance_kernel (function) - Function to determine whether to accept parameters
    scaling_factor (float) - Scaling factor for `acceptance_kernel`.

    OPTIONAL PARAMETERS
    summary_stats ([function]) - functions which summarise `y_obs` and the observations of `fitting_model` in some way. (default=group by dimension)

    RETURNS
    Model - fitted model with best parameters
    """
    group_dim = lambda ys,i: [y[i] for y in ys]
    summary_stats=summary_stats if (summary_stats) else ([(lambda ys:group_dim(ys,i)) for i in range(len(y_obs[0]))])
    s_obs=[s(y_obs) for s in summary_stats]

    # find starting sample
    i=0
    while (True):
        print("Finding Start - ({:,})".format(i),end="\r")
        i+=1
        theta_0=[pi_i.rvs(1)[0] for pi_i in priors]

        # observe theorised model
        fitting_model.update_params(theta_0)
        y_0=fitting_model.observe()
        s_0=[s(y_0) for s in summary_stats]

        # accept-reject
        norm_vals=[l2_norm(s_0_i,s_obs_i) for (s_0_i,s_obs_i) in zip(s_0,s_obs)]
        if all([acceptance_kernel(v,scaling_factor) for v in norm_vals]): break

    THETAS=[theta_0]
    ACCEPTED_SUMMARY_VALS=[s_0]

    print("Found Start - ({:,})".format(i),theta_0)

    # MCMC step
    new=0
    for t in range(1,chain_length+1):
        # perturb last sample
        theta_temp=[k(theta_i) for (k,theta_i) in zip(perturbance_kernels,THETAS[-1])]
        while any([p.pdf(theta)==0.0 for (p,theta) in zip(priors,theta_temp)]): theta_temp=[k(theta_i) for (k,theta_i) in zip(perturbance_kernels,THETAS[-1])]

        # observed theorised model
        fitting_model.update_params(theta_temp)
        y_temp=fitting_model.observe()
        s_temp=[s(y_temp) for s in summary_stats]

        # accept-reject
        norm_vals=[l2_norm(s_temp_i,s_obs_i) for (s_temp_i,s_obs_i) in zip(s_temp,s_obs)]
        if all([acceptance_kernel(v,scaling_factor) for v in norm_vals]): # accept new parameter sample
            new+=1
            THETAS.append(theta_temp)
            ACCEPTED_SUMMARY_VALS.append(s_temp)
            print("({:,}) - NEW".format(t),end="\r")
        else: # stick with last parameter sample
            THETAS.append(THETAS[-1])
            ACCEPTED_SUMMARY_VALS.append(ACCEPTED_SUMMARY_VALS[-1])
            print("({:,}) - OLD".format(t),end="\r")

    theta_hat=list(np.mean(THETAS,axis=0))
    model_hat=fitting_model.copy(theta_hat)
    s_hat=[s(model_hat.observe()) for s in summary_stats]
    print("{:.3f} observations were new.".format(new/chain_length))

    n_simple_ss=sum(len(s)==1 for s in ACCEPTED_SUMMARY_VALS[0]) # number of summary stats which map to a single dimension
    n_rows=max([1,np.lcm(fitting_model.n_params,n_simple_ss)])
    fig=plt.figure(constrained_layout=True)
    gs=fig.add_gridspec(n_rows,4) # 3 columns
    ax=fig.add_subplot(gs[:,-1])
    Plotting.plot_accepted_observations(ax,fitting_model.n_obs,y_obs,[],model_hat)

    # plot traces
    row_step=n_rows//fitting_model.n_params
    for i in range(fitting_model.n_params):
        name="Theta_{}".format(i)
        ax=fig.add_subplot(gs[i*row_step:(i+1)*row_step,0])
        accepted_vals=[x[i] for x in THETAS]
        Plotting.plot_MCMC_trace(ax,name,accepted_parameter=accepted_vals,predicted_val=theta_hat[i])

    # plot posteriors
    for i in range(fitting_model.n_params):
        name="Theta_{}".format(i)
        ax=fig.add_subplot(gs[i*row_step:(i+1)*row_step,1])
        accepted_vals=[x[i] for x in THETAS]
        Plotting.plot_parameter_posterior(ax,name,accepted_parameter=accepted_vals,predicted_val=theta_hat[i],prior=priors[i])

    # plot summary vals
    row=0
    row_step=n_rows//n_simple_ss
    for i in range(len(summary_stats)):
        if (len(ACCEPTED_SUMMARY_VALS[0][i])==1):
            name="s_{}".format(i)
            ax=fig.add_subplot(gs[row*row_step:(row+1)*row_step,2])
            row+=1
            accepted_vals=[s[i][0] for s in ACCEPTED_SUMMARY_VALS]
            Plotting.plot_summary_stats(ax,name,accepted_s=accepted_vals,s_obs=s_obs[i],s_hat=s_hat[i])

    plt.show()
    return model_hat

def abc_smc(n_obs:int,y_obs:[[float]],
    fitting_model:Model,priors:["stats.Distribution"],
    num_steps:int,sample_size:int,
    scaling_factors:[float],perturbance_kernels:"[function]",perturbance_kernel_probability:"[function]",acceptance_kernel:"function",summary_stats=None):
    """
    DESCRIPTION
    Sequential Monte-Carlo Sampling version of Approximate Bayesian Computation for the generative models defined in `Models.py`.

    PARAMETERS
    n_obs (int) - Number of observations available.
    y_obs ([[float]]) - Observations from true model.
    fitting_model (Model) - Model the algorithm will aim to fit to observations.
    priors (["stats.Distribution"]) - Priors for the value of parameters of `fitting_model`.
    num_steps (int) - Number of steps (ie number of scaling factors).
    sample_size (int) - Number of parameters samples to keep per step.
    scaling_factors ([float]) - Scaling factor for `acceptance_kernel`.
    perturbance_kernels ([function]) - Functions for varying parameters each monte-carlo steps.
    perturbance_kernel_probability ([function]) - Probability of x being pertubered to value y
    acceptance_kernel (function) - Function to determine whether to accept parameters

    OPTIONAL PARAMETERS
    summary_stats ([function]) - functions which summarise `y_obs` and the observations of `fitting_model` in some way. (default=group by dimension)

    RETURNS
    Model - fitted model with best parameters
    """
    # initial sampling
    if (num_steps!=len(scaling_factors)): raise ValueError("`num_steps` must equal `len(scaling_factors)`")

    group_dim = lambda ys,i: [y[i] for y in ys]
    summary_stats=summary_stats if (summary_stats) else ([(lambda ys:group_dim(ys,i)) for i in range(len(y_obs[0]))])
    s_obs=[s(y_obs) for s in summary_stats]

    # initial sampling
    THETAS=[] # (weight,params)
    i=0
    while (len(THETAS)<sample_size):
        i+=1
        theta_temp=[pi_i.rvs(1)[0] for pi_i in priors]

        # observed theorised model
        fitting_model.update_params(theta_temp)
        y_temp=fitting_model.observe()
        s_temp=[s(y_temp) for s in summary_stats]

        # accept-reject
        norm_vals=[l2_norm(s_temp_i,s_obs_i) for (s_temp_i,s_obs_i) in zip(s_temp,s_obs)]
        if all([acceptance_kernel(v,scaling_factors[0]) for v in norm_vals]): # accept new parameter sample
            THETAS.append((1/sample_size,theta_temp))
        print("({:,}) - {:,}/{:,}".format(i,len(THETAS),sample_size),end="\r")
    print()

    # resampling & reweighting step
    for t in range(1,num_steps):
        i=0
        NEW_THETAS=[] # (weight,params)
        while (len(NEW_THETAS)<sample_size):
            i+=1
            print("({:,}/{:,} - {:,}) - {:,}/{:,} ({:.3f})".format(t,num_steps,i,len(NEW_THETAS),sample_size,scaling_factors[t]),end="\r")

            # sample from THETA
            u=stats.uniform(0,1).rvs(1)[0]
            theta_t=None
            for (weight,theta_i) in THETAS:
                u-=weight
                if (u<=0): theta_t=theta_i; break

            # perturb sample
            theta_temp=[k(theta_i) for (k,theta_i) in zip(perturbance_kernels,theta_t)]
            while any([p.pdf(theta)==0.0 for (p,theta) in zip(priors,theta_temp)]): theta_temp=[k(theta_i) for (k,theta_i) in zip(perturbance_kernels,theta_t)]

            # observed theorised model
            fitting_model.update_params(theta_temp)
            y_temp=fitting_model.observe()
            s_temp=[s(y_temp) for s in summary_stats]

            # accept-reject
            norm_vals=[l2_norm(s_temp_i,s_obs_i) for (s_temp_i,s_obs_i) in zip(s_temp,s_obs)]
            if all([acceptance_kernel(v,scaling_factors[t]) for v in norm_vals]): # accept new parameter sample
                weight_numerator=sum([p.pdf(theta) for (p,theta) in zip(priors,theta_temp)])
                weight_denominator=0
                for (weight,theta) in THETAS:
                    weight_denominator+=sum([weight*p(theta_i,theta_temp_i) for (p,theta_i,theta_temp_i) in zip(perturbance_kernel_probability,theta,theta_temp)]) # probability theta_temp was sampled
                weight=weight_numerator/weight_denominator
                NEW_THETAS.append((weight,theta_temp))

        weight_sum=sum([w for (w,_) in NEW_THETAS])
        NEW_THETAS=[(w/weight_sum,theta) for (w,theta) in NEW_THETAS]

        THETAS=NEW_THETAS

    print()

    param_values=[theta for (_,theta) in THETAS]
    weights=[w for (w,_) in THETAS]
    theta_hat=list(np.average(param_values,axis=0,weights=weights))
    model_hat=fitting_model.copy(theta_hat)

    fig=plt.figure()
    ax=fig.add_subplot(1,fitting_model.n_params+1,fitting_model.n_params+1)
    Plotting.plot_accepted_observations(ax,fitting_model.n_obs,y_obs,[],model_hat)

    print("theta_hat -",theta_hat)

    for i in range(fitting_model.n_params):
        ax=fig.add_subplot(1,fitting_model.n_params+1,i+1)
        name="theta_{}".format(i)
        parameter_values=[theta[i] for theta in param_values]
        Plotting.plot_smc_posterior(ax,name,parameter_values=parameter_values,weights=weights,predicted_val=theta_hat[i])

    plt.show()

    return model_hat
