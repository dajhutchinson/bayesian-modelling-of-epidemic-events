from Models import Model
from scipy import stats
from scipy.signal import correlate2d
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
def l1_norm(xs:[float],ys=[]) -> float:
    return sum(xs)+sum(ys)

def l2_norm(s_t:(float),s_obs:(float)) -> float:
    return sum([(x-y)**2 for (x,y) in zip(s_t,s_obs)])**.5

def log_l2_norm(s_t:(float),s_obs:(float)) -> float:

    return sum([(np.log(x)-np.log(y))**2 if (x>0 and y>0) else 0 for (x,y) in zip(s_t,s_obs)])**.5

def l_infty_norm(xs:[float]) -> float:
    return max(xs)

"""
    REJECTION SAMPLING METHODS
"""

def __sampling_stage_fixed_number(DESIRED_SAMPLE_SIZE:int,EPSILON:float,KERNEL:"func",PRIORS:["stats.Distribution"],
    s_obs:[float],model_t:Model,summary_stats:["function"],distance_measure=l2_norm) -> ([[float]],[[float]],[[float]]):
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
    distance_measure - (func) - distance function to use (See choices above)

    RETURNS
    [[float]] - accepted parameter values
    [[float]] - observations made when using accepted parameter values
    [[float]] - summary statistic values for accepted parameter observations
    """

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
        norm_vals=[distance_measure(s_t_i,s_obs_i) for (s_t_i,s_obs_i) in zip(s_t,s_obs)]
        if (KERNEL(l1_norm(norm_vals),EPSILON)): # NOTE - l1_norm() can be replaced by anyother other norm
            ACCEPTED_PARAMS.append(theta_t)
            ACCEPTED_OBS.append(y_t)
            ACCEPTED_SUMMARY_VALS.append(s_t)

        i+=1
        print("({:,}) {:,}/{:,}".format(i,len(ACCEPTED_PARAMS),DESIRED_SAMPLE_SIZE),end="\r") # update user on sampling process
    print("\n")

    return ACCEPTED_PARAMS,ACCEPTED_OBS,ACCEPTED_SUMMARY_VALS

def __sampling_stage_best_samples(NUM_RUNS:int,SAMPLE_SIZE:int,PRIORS:["stats.Distribution"],
    s_obs:[float],model_t:Model,summary_stats:["function"],distance_measure=l2_norm) -> ([[float]],[[float]],[[float]]):
    """
    DESCRIPTION
    perform `NUM_RUNS` samples and return the parameters values associated to the best `SAMPLE_SIZE`.

    PARAMETERS
    NUM_RUNS (int) - number of samples to make
    SAMPLE_SIZE (int) - The best n set of parameters to return
    PRIORS ([stats.Distribution]) - prior distribution for parameter values (one per parameter)
    s_obs ([float]) - summary statistic values for observations from true model.
    summary_stat ([func]) - functions used to determine each summary statistic.
    distance_measure - (func) - distance function to use (See choices above)

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

        norm_vals=[distance_measure(s_t_i,s_obs_i) for (s_t_i,s_obs_i) in zip(s_t,s_obs)]
        summarised_norm_val=l1_norm(norm_vals) # l1_norm can be replaced by any other norm

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

def abc_rejcection(n_obs:int,y_obs:[[float]],fitting_model:Model,priors:["stats.Distribution"],sampling_details:dict,summary_stats=None) -> (Model,[[float]]):
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
    [[float]] - set of all accepted parameter values (use for further investigation)
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
        distance_measure=l2_norm if (not "distance_measure" in sampling_details) else sampling_details["distance_measure"]
        ACCEPTED_PARAMS,ACCEPTED_OBS,ACCEPTED_SUMMARY_VALS=__sampling_stage_fixed_number(sample_size,epsilon,kernel,PRIORS=priors,s_obs=s_obs,model_t=fitting_model,summary_stats=summary_stats,distance_measure=distance_measure)

    elif (sampling_details["sampling_method"]=="best"):
        if any([x not in sampling_details for x in ["sample_size","num_runs"]]): raise Exception("`sampling_details` missing key(s) - expecting `num_runs` and `sample_size`")
        num_runs=sampling_details["num_runs"]
        sample_size=sampling_details["sample_size"]
        distance_measure=l2_norm if (not "distance_measure" in sampling_details) else sampling_details["distance_measure"]
        ACCEPTED_PARAMS,ACCEPTED_OBS,ACCEPTED_SUMMARY_VALS=__sampling_stage_best_samples(num_runs,sample_size,PRIORS=priors,s_obs=s_obs,model_t=fitting_model,summary_stats=summary_stats,distance_measure=distance_measure)

    # best estimate of model
    theta_hat=[np.mean([p[i] for p in ACCEPTED_PARAMS]) for i in range(fitting_model.n_params)]
    model_hat=fitting_model.copy(theta_hat)
    s_hat=[s(model_hat.observe()) for s in summary_stats]

    # plot results
    n_simple_ss=sum(len(s)==1 for s in ACCEPTED_SUMMARY_VALS[0]) # number of summary stats which map to a single dimension
    n_cols=2 if (n_simple_ss==0) else 3
    n_rows=max([1,np.lcm.reduce([fitting_model.n_params,max(1,n_simple_ss),fitting_model.dim_obs])])

    # plot accepted obervations for each dimension
    fig=plt.figure(constrained_layout=True)
    gs=fig.add_gridspec(n_rows,n_cols)

    # plot accepted observations (each dimension separate)
    row_step=n_rows//fitting_model.dim_obs
    for i in range(fitting_model.dim_obs):
        ax=fig.add_subplot(gs[i*row_step:(i+1)*row_step,-1])
        y_obs_dim=[y[i] for y in y_obs]
        accepted_obs_dim=[[y[i] for y in obs] for obs in ACCEPTED_OBS]
        Plotting.plot_accepted_observations(ax,fitting_model.n_obs,y_obs_dim,accepted_obs_dim,model_hat,dim=i)

    # plot posterior for each parameter
    row_step=n_rows//fitting_model.n_params
    for i in range(fitting_model.n_params):
        name="Theta_{}".format(i)
        ax=fig.add_subplot(gs[i*row_step:(i+1)*row_step,0])
        accepted_vals=[x[i] for x in ACCEPTED_PARAMS]
        Plotting.plot_parameter_posterior(ax,name,accepted_parameter=accepted_vals,predicted_val=theta_hat[i],prior=priors[i],dim=i)

    # plot histogram of each summary statistic value
    row=0
    if (n_simple_ss!=0):
        row_step=n_rows//n_simple_ss
        for i in range(len(summary_stats)):
            if (len(ACCEPTED_SUMMARY_VALS[0][i])==1):
                name="s_{}".format(i)
                ax=fig.add_subplot(gs[row*row_step:(row+1)*row_step,1])
                row+=1
                accepted_vals=[s[i][0] for s in ACCEPTED_SUMMARY_VALS]
                Plotting.plot_summary_stats(ax,name,accepted_s=accepted_vals,s_obs=s_obs[i],s_hat=s_hat[i],dim=i)

    # plt.get_current_fig_manager().window.state("zoomed")
    plt.show()

    return model_hat,ACCEPTED_PARAMS

def abc_mcmc(n_obs:int,y_obs:[[float]],
    fitting_model:Model,priors:["stats.Distribution"],
    chain_length:int,perturbance_kernels:"[function]",acceptance_kernel:"function",scaling_factor:float,
    summary_stats=None,distance_measure=l2_norm) -> (Model,[[float]]):
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
    distance_measure - (func) - distance function to use (See choices above)

    OPTIONAL PARAMETERS
    summary_stats ([function]) - functions which summarise `y_obs` and the observations of `fitting_model` in some way. (default=group by dimension)

    RETURNS
    Model - fitted model with best parameters
    [[float]] - set of all accepted parameter values (use for further investigation)
    """
    group_dim = lambda ys,i: [y[i] for y in ys]
    summary_stats=summary_stats if (summary_stats) else ([(lambda ys:group_dim(ys,i)) for i in range(len(y_obs[0]))])
    s_obs=[s(y_obs) for s in summary_stats]

    # find starting sample
    min_l1_norm=100000000000
    i=0
    while (True):
        print("Finding Start - ({:,},{:,.3f})                       ".format(i,min_l1_norm),end="\r")
        i+=1
        theta_0=[pi_i.rvs(1)[0] for pi_i in priors]

        # observe theorised model
        fitting_model.update_params(theta_0)
        y_0=fitting_model.observe()
        s_0=[s(y_0) for s in summary_stats]

        # accept-reject
        norm_vals=[l2_norm(s_0_i,s_obs_i) for (s_0_i,s_obs_i) in zip(s_0,s_obs)]
        if (l1_norm(norm_vals)<min_l1_norm): min_l1_norm=l1_norm(norm_vals)
        if (acceptance_kernel(l1_norm(norm_vals),scaling_factor)): break

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
        norm_vals=[distance_measure(s_temp_i,s_obs_i) for (s_temp_i,s_obs_i) in zip(s_temp,s_obs)]
        if (acceptance_kernel(l1_norm(norm_vals),scaling_factor)):
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
    # print("Auto-correlation - ",correlate2d(THETAS,THETAS))

    n_simple_ss=sum(len(s)==1 for s in ACCEPTED_SUMMARY_VALS[0]) # number of summary stats which map to a single dimension
    n_cols=3 if (n_simple_ss==0) else 4
    n_rows=max([1,np.lcm.reduce([fitting_model.n_params,max(1,n_simple_ss),fitting_model.dim_obs])])

    fig=plt.figure(constrained_layout=True)
    gs=fig.add_gridspec(n_rows,n_cols)

    # plot fitted model
    row_step=n_rows//fitting_model.dim_obs
    for i in range(fitting_model.dim_obs):
        ax=fig.add_subplot(gs[i*row_step:(i+1)*row_step,-1])
        y_obs_dim=[y[i] for y in y_obs]
        Plotting.plot_accepted_observations(ax,fitting_model.n_obs,y_obs_dim,[],model_hat,dim=i)

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
        Plotting.plot_parameter_posterior(ax,name,accepted_parameter=accepted_vals,predicted_val=theta_hat[i],prior=priors[i],dim=i)

    # plot summary vals
    row=0
    row_step=n_rows//n_simple_ss
    for i in range(len(summary_stats)):
        if (len(ACCEPTED_SUMMARY_VALS[0][i])==1):
            name="s_{}".format(i)
            ax=fig.add_subplot(gs[row*row_step:(row+1)*row_step,2])
            row+=1
            accepted_vals=[s[i][0] for s in ACCEPTED_SUMMARY_VALS]
            Plotting.plot_summary_stats(ax,name,accepted_s=accepted_vals,s_obs=s_obs[i],s_hat=s_hat[i],dim=i)

    plt.show()
    return model_hat, THETAS

def abc_smc(n_obs:int,y_obs:[[float]],
    fitting_model:Model,priors:["stats.Distribution"],
    num_steps:int,sample_size:int,
    scaling_factors:[float],perturbance_kernels:"[function]",perturbance_kernel_probability:"[function]",
    acceptance_kernel:"function",summary_stats=None,distance_measure=l2_norm) -> (Model,[[float]]):
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
    distance_measure - (func) - distance function to use (See choices above)

    OPTIONAL PARAMETERS
    summary_stats ([function]) - functions which summarise `y_obs` and the observations of `fitting_model` in some way. (default=group by dimension)

    RETURNS
    Model - fitted model with best parameters
    [[float]] - set of all accepted parameter values (use for further investigation)
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
        norm_vals=[distance_measure(s_temp_i,s_obs_i) for (s_temp_i,s_obs_i) in zip(s_temp,s_obs)]
        if (acceptance_kernel(l1_norm(norm_vals),scaling_factors[0])):
            THETAS.append((1/sample_size,theta_temp))
        print("({:,}) - {:,}/{:,}".format(i,len(THETAS),sample_size),end="\r")
    print()

    total_simulations=i

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
            if (acceptance_kernel(l1_norm(norm_vals),scaling_factors[t])):
                weight_numerator=sum([p.pdf(theta) for (p,theta) in zip(priors,theta_temp)])
                weight_denominator=0
                for (weight,theta) in THETAS:
                    weight_denominator+=sum([weight*p(theta_i,theta_temp_i) for (p,theta_i,theta_temp_i) in zip(perturbance_kernel_probability,theta,theta_temp)]) # probability theta_temp was sampled
                weight=weight_numerator/weight_denominator
                NEW_THETAS.append((weight,theta_temp))

        total_simulations+=i
        weight_sum=sum([w for (w,_) in NEW_THETAS])
        NEW_THETAS=[(w/weight_sum,theta) for (w,theta) in NEW_THETAS]

        THETAS=NEW_THETAS

    print()

    param_values=[theta for (_,theta) in THETAS]
    weights=[w for (w,_) in THETAS]
    theta_hat=list(np.average(param_values,axis=0,weights=weights))
    model_hat=fitting_model.copy(theta_hat)

    n_rows=max([1,np.lcm(fitting_model.n_params,fitting_model.dim_obs)])

    fig=plt.figure(constrained_layout=True)
    gs=fig.add_gridspec(n_rows,2)

    # plot fitted model
    row_step=n_rows//fitting_model.dim_obs
    for i in range(fitting_model.dim_obs):
        ax=fig.add_subplot(gs[i*row_step:(i+1)*row_step,-1])
        y_obs_dim=[y[i] for y in y_obs]
        Plotting.plot_accepted_observations(ax,fitting_model.n_obs,y_obs_dim,[],model_hat,dim=i)

    print("Total Simulations - {:,}".format(total_simulations))
    print("theta_hat -",theta_hat)

    row_step=n_rows//fitting_model.n_params
    for i in range(fitting_model.n_params):
        ax=fig.add_subplot(gs[i*row_step:(i+1)*row_step,0])
        name="theta_{}".format(i)
        parameter_values=[theta[i] for theta in param_values]
        Plotting.plot_smc_posterior(ax,name,parameter_values=parameter_values,weights=weights,predicted_val=theta_hat[i],prior=priors[i],dim=i)

    plt.show()

    return model_hat,param_values

"""
    SUMMARY STATISTIC SELECTION
"""

def __compare_summary_stats(accepted_params_curr:[[float]],accepted_params_prop:[[float]],param_bounds:[(float,float)],n_params:int,n_bins=10) -> bool:
    """
    DESCRIPTION
    The algorithm proposed by joyce-marjoram for estimating the odds-ratio for two sets of summary statistics S_{K-1} and S_K
    where S_K is a super-set of S_{K-1}

    PARAMETERS
    accepted_params_curr ([[float]]) - Sets of parameters which were accepted when using current set of summary stats S_{K-1}.
    accepted_params_prop ([[float]]) - Sets of parameters which were accepted when using propsed set of summary stats S_K.
    param_bounds ([(float,float)]) - The bounds of the priors used to generate parameter sets.
    n_params (int) - Number of parameters being fitted.
    n_bins (int) - Number of bins to discretise each dimension of posterior into (default=10)

    RETURNS
    bool - Whether S_K produces a notably different posterior to S_{K-1} (ie whether to accept new summary stat or not)
    """
    n_curr=len(accepted_params_curr)
    if (n_curr==0): return True # if nothing is currently being used then accept
    n_prop=len(accepted_params_prop)

    # count occurences of accepted params
    bins_curr=[[0 for _ in range(n_bins)] for _ in range(n_params)]
    for params in accepted_params_curr:
        for (dim,param) in enumerate(params):
            step=(param_bounds[dim][1]-param_bounds[dim][0])/(n_bins-1)
            if (step==0): step=1 # avoid division my zero

            i=int(np.floor((param-param_bounds[dim][0])/step))
            bins_curr[dim][i]+=1

    bins_prop=[[0 for _ in range(n_bins)] for _ in range(n_params)]
    for params in accepted_params_prop:
        for (dim,param) in enumerate(params):
            step=(param_bounds[dim][1]-param_bounds[dim][0])/(n_bins-1)
            if (step==0): step=1 # avoid division my zero

            i=int(np.floor((param-param_bounds[dim][0])/step))
            bins_prop[dim][i]+=1

    # calculated expected number of occurences for each bin
    expected=[[(x*n_prop)/n_curr for x in bins] for bins in bins_curr]
    sd=[[np.sqrt(expected[i][j]*((n_curr-x)/n_curr)) for (j,x) in enumerate(bins)] for (i,bins) in enumerate(bins_curr)]

    upper_thresh=[[expected[i][j]+4*sd[i][j] for j in range(n_bins)] for i in range(n_params)]
    lower_thresh=[[expected[i][j]-4*sd[i][j] for j in range(n_bins)] for i in range(n_params)]

    # count number of extreme values
    # value is extreme if it is more than 4sd away from expected
    extreme=0
    for i in range(n_params):
        for j in range(n_bins):
            if (bins_prop[i][j]>upper_thresh[i][j]) or (bins_prop[i][j]<lower_thresh[i][j]): extreme+=1

    return (extreme>0)


def joyce_marjoram(summary_stats:["function"],n_obs:int,y_obs:[[float]],fitting_model:Model,priors:["stats.Distribution"],param_bounds:[(float,float)],
    KERNEL=uniform_kernel,BANDWIDTH=1,n_samples=10000,n_bins=10,printing=True) -> [int]:
    """
    DESCRIPTION
    Use the algorithm in Paul Joyce, Paul Marjoram 2008 to find an approxiamtely sufficient set of summary statistics (from set `summary_stats`)

    PARAMETERS
    summary_stats ([function]) - functions which summarise `y_obs` and the observations of `fitting_model` in some way. These are what will be evaluated
    n_obs (int) - Number of observations available.
    y_obs ([[float]]) - Observations from true model.
    fitting_model (Model) - Model the algorithm will aim to fit to observations.
    priors (["stats.Distribution"]) - Priors for the value of parameters of `fitting_model`.
    param_bounds ([(float,float)]) - The bounds of the priors used to generate parameter sets.
    KERNEL (func) - one of the kernels defined above. determine which parameters are good or not.
    BANDWIDTH (float) - scale parameter for `KERNEL`
    n_samples (int) - number of samples to make
    n_bins (int) - Number of bins to discretise each dimension of posterior into (default=10)

    RETURNS
    [int] - indexes of selected summary stats in `summary_stats`
    """

    if (type(y_obs)!=list): raise TypeError("`y_obs` must be a list (not {})".format(type(y_obs)))
    if (len(y_obs)!=n_obs): raise ValueError("Wrong number of observations supplied (len(y_obs)!=n_obs) ({}!={})".format(len(y_obs),n_obs))
    if (len(priors)!=fitting_model.n_params): raise ValueError("Wrong number of priors given (exp fitting_model.n_params={})".format(fitting_model.n_params))

    group_dim = lambda ys,i: [y[i] for y in ys]
    summary_stats=summary_stats if (summary_stats) else ([(lambda ys:group_dim(ys,i)) for i in range(len(y_obs[0]))])
    s_obs=[s(y_obs) for s in summary_stats]

    # generate samples
    SAMPLES=[] # (theta,s_vals)
    for i in range(n_samples):
        if (printing): print("{:,}/{:,}".format(i+1,n_samples),end="\r")

        # sample parameters
        theta_t=[pi_i.rvs(1)[0] for pi_i in priors]

        # observe theorised model
        fitting_model.update_params(theta_t)
        y_t=fitting_model.observe()
        s_t=[s(y_t) for s in summary_stats]

        SAMPLES.append((theta_t,s_t))

    if (printing): print()

    # consider adding each summary stat in turn
    ACCEPTED_SUMMARY_STATS_ID=[] # index of accepted summary stats
    prev_prob=1

    # prep for comparing summary stats
    SAMPLES_curr=[]
    s_obs_curr=[]
    ACCEPTED_PARAMS_curr=[]

    """TODO-improve this so best stat is chosen first or do some sort of leave-one-out testing"""

    added=True
    while added: # keep running until no new summary stat is accepted
        added=False
        if (printing): print("ACCEPTED_SUMMARY_STATS_ID - ",ACCEPTED_SUMMARY_STATS_ID)
        best_ss=(0,-1) # prob,index

        # identify which params are accepted under proposed
        for i in range(len(summary_stats)):
            if i in ACCEPTED_SUMMARY_STATS_ID: continue # ignore stats already in set
            if (printing): print("Considering adding {} to [{}]. ".format(i,",".join([str(x) for x in ACCEPTED_SUMMARY_STATS_ID])),end="")

            SAMPLES_prop=[(theta,[s[j] for j in ACCEPTED_SUMMARY_STATS_ID+[i]]) for (theta,s) in SAMPLES] # parameters & summary stat values
            s_obs_prop=[s_obs[j] for j in ACCEPTED_SUMMARY_STATS_ID+[i]] # observed summary stat values

            # accept-reject
            ACCEPTED_PARAMS_prop=[]
            for (theta_t,s_t) in SAMPLES_prop:
                norm_vals=[l2_norm(s_t_i,s_obs_i) for (s_t_i,s_obs_i) in zip(s_t,s_obs_prop)]
                if (KERNEL(l1_norm(norm_vals),BANDWIDTH*len(ACCEPTED_SUMMARY_STATS_ID+[i]))): # NOTE - l1_norm() can be replaced by anyother other norm
                    ACCEPTED_PARAMS_prop.append(theta_t)

            if (printing): print("(n_curr={},n_prop={})".format(len(ACCEPTED_PARAMS_curr),len(ACCEPTED_PARAMS_prop)),end="\n")

            # if summary stat helps
            if __compare_summary_stats(ACCEPTED_PARAMS_curr,ACCEPTED_PARAMS_prop,param_bounds=param_bounds,n_params=len(priors),n_bins=n_bins):
                if (printing): print("Accepted summary stats - {}".format(i))

                # confirm addition by considering removing each other summary stat in turn
                removal=False
                for j in ACCEPTED_SUMMARY_STATS_ID:
                    if (printing): print("Comparing [{}] to [{}]. ".format(",".join([str(x) for x in ACCEPTED_SUMMARY_STATS_ID if x!=j]+[str(i)]),",".join([str(x) for x in ACCEPTED_SUMMARY_STATS_ID]+[str(i)])),end="")
                    accepted_ss_ids_minus_one=[x for x in ACCEPTED_SUMMARY_STATS_ID if x!=j]+[i]
                    SAMPLES_minus_one=[(theta,[s[j] for j in accepted_ss_ids_minus_one]) for (theta,s) in SAMPLES] # parameters & summary stat values
                    s_obs_minus_one=[s_obs[j] for j in accepted_ss_ids_minus_one] # observed summary stat values

                    # accept-reject
                    ACCEPTED_PARAMS_minus_one=[]
                    for (theta_t,s_t) in SAMPLES_minus_one:
                        norm_vals=[l2_norm(s_t_i,s_obs_i) for (s_t_i,s_obs_i) in zip(s_t,s_obs_minus_one)]
                        if (KERNEL(l1_norm(norm_vals),BANDWIDTH*len(accepted_ss_ids_minus_one))): # NOTE - l1_norm() can be replaced by anyother other norm
                            ACCEPTED_PARAMS_minus_one.append(theta_t)
                    if (printing): print("(n_minus_one={},n_prop={})".format(len(ACCEPTED_PARAMS_minus_one),len(ACCEPTED_PARAMS_prop)),end="\n")

                    if __compare_summary_stats(ACCEPTED_PARAMS_minus_one,ACCEPTED_PARAMS_prop,param_bounds=param_bounds,n_params=len(priors),n_bins=n_bins):
                        ACCEPTED_SUMMARY_STATS_ID=accepted_ss_ids_minus_one
                        removal=True
                        break

                if (not removal): ACCEPTED_SUMMARY_STATS_ID+=[i]

                SAMPLES_curr=SAMPLES_prop
                s_obs_curr=s_obs_prop
                ACCEPTED_PARAMS_curr=ACCEPTED_PARAMS_prop

                added=True
                break

        if (printing): print()

    return ACCEPTED_SUMMARY_STATS_ID
