"""
TODO
 - ABC-SMC
 - ABC-Semi-Automatic
 - Summary statistics
"""

from scipy import stats
import matplotlib.pyplot as plt
from Models import Model,LinearModel,ExponentialModel,GeneralLinearModel
import numpy as np

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
    ep_val=(1/epsilon)*(3/4)*(1-(x/epsilon)**2) # prob to accept
    x=stats.uniform(0,1).rvs(1)[0] # sample from U[0,1]
    return (x<=ep_val)

def gaussian_kernel(x:float,epsilon:float) -> bool:
    gaus_val=(1/np.sqrt(2*pi*(x**2)))*np.exp(-(1/2)*((x/epsilon)**2)) # prob to accept
    x=stats.uniform(0,1).rvs(1)[0] # sample from U[0,1]
    return (x<=gaus_val)

"""
    DISTANCE MEASURES
"""
def l2_norm(s:(float),s_obs:(float)) -> float:
    return sum([(x-y)**2 for (x,y) in zip(s,s_obs)])**.5

"""
   PLOTTING
"""

# def __plot_results(fig:plt.Figure,model_hat:Model,model_star:Model,samples:[([float],([float],float))],priors:["stats.distribution"],var_ranges:[(float,float)],observations:[([float],float)],plot_truth=True):
def __plot_results(fig:plt.Figure,model_hat:Model,model_star:Model,THETA_SAMPLES:[[float]],ACCEPTED_OBS:[([float],float)],priors:["stats.distribution"],var_ranges:[(float,float)],observations:[([float],float)],plot_truth=True):
    """
    DESCRIPTION
    Produces plots for parameter posteriors and model fit (if <=2 variables).
    """
    n_plots=model_star.n_params
    if (n_plots)<=3: n_plots+=1
    plt.subplots_adjust(left=.05,right=.95,bottom=.05,top=.95)

    # plot parameter results
    for i in range(model_star.n_params):
        theta_i_samples=[x[i] for x in THETA_SAMPLES]
        ax=fig.add_subplot(1,n_plots,i+1)
        __plot_posterior(ax,"Theta_{}".format(i),model_star.params[i],theta_i_samples,priors[i],plot_truth)

    # 2d plot
    if (model_star.n_vars==1):
        ax=fig.add_subplot(1,n_plots,n_plots)
        __plot_2d_samples(ax,ACCEPTED_OBS,observations,model_hat,plot_truth)

    # 2d plot
    elif (model_star.n_vars==2):
        ax=fig.add_subplot(1,n_plots,n_plots,projection="3d")
        __plot_3d_samples(ax,ACCEPTED_OBS,observations,model_hat,model_star,var_ranges,plot_truth)

    plt.get_current_fig_manager().window.state('zoomed')
    plt.show()
    pass

def __plot_posterior(ax:plt.Axes,name:str,theta_star:float,theta_samples:[float],prior:"stats.Distribution",plot_truth=True) -> plt.Axes:
    """
    DESCRIPTION
    plot the posterior for a given parameter.

    PARAMETERS
    ax (plt.Axes) - Axes to produce plot on.
    name(str) - Name of parameter (theta_0 etc.)
    theta_star(float) - True value of parameter
    theta_samples([float]) - List of all samples of parameter which were accepted.
    prior(stats.Distribution) - Prior used to sample parameter.
    plot_true(bool) - Whether to plot the true parameter value.

    RETURNS
    plt.Axes - The axis which plot was made on.
    """
    print("{}\nTrue {:.5f}\nMean {:.5f}\nMedian {:.5f}\n".format(name,theta_star,np.mean(theta_samples),np.median(theta_samples)))

    ax.set_title("Posterior for {}".format(name))
    ax.hist(theta_samples,density=True)

    x=np.linspace(prior.ppf(0.01),prior.ppf(0.99), 100)
    ax.plot(x,prior.pdf(x), 'k-', lw=2, label='Prior')

    if (plot_truth): ax.vlines(theta_star,ymin=0,ymax=ax.get_ylim()[1],colors="red",label="Truth")
    ax.vlines(np.mean(theta_samples), ymin=0,ymax=ax.get_ylim()[1],colors="orange",label="Prediction")
    ax.set_xlabel("{}".format(name))
    ax.legend()

    return ax

def __plot_2d_samples(ax:plt.Axes,accepted_samples:[([float],float)],observations:[([float],float)],model_hat:Model,plot_truth=True) -> plt.Axes:
    """
    DESCRIPTION
    plot the fitted model, observations and accepted samples for a fitted model with a single predictor variable.

    PARAMETERS
    ax (plt.Axes) - Axes to produce plot on.
    accepted_samples([([float],float)]) - Details of samples which were accepted [([pred_var_values],response_var_value)].
    observations([([float],float)]) - Observations made before sampling [([pred_var_values],response_var_value)].
    model_hat(Model) - Fitted model.
    plot_true(bool) - Whether to plot the true parameter value.

    RETURNS
    plt.Axes - The axis which plot was made on.
    """
    ax.set_title("Accepted Samples")
    ax.set_xlabel("Predictor Variable")
    ax.set_ylabel("Response Variable")

    # plot accepted samples
    sample_xs=[x[0][0] for x in accepted_samples]
    sample_ss=[x[1] for x in accepted_samples]
    ax.scatter(sample_xs,sample_ss,label="Accepted Samples")

    # plot observations
    x_obs=[x[0] for x in observations]
    s_obs=[x[1] for x in observations]

    ax.scatter(x_obs,s_obs,c="red",label="Observations")

    # plot predicted model
    s_hat=[model_hat.calc(x) for x in x_obs]
    ax.plot(x_obs,s_hat,c="orange",label="Predicted Model")
    ax.legend()

    return ax

def __plot_3d_samples(ax:plt.Axes,accepted_samples:[([float],float)],observations:[([float],float)],model_hat:Model,model_star:Model,var_ranges:[(float,float)],plot_truth=True) -> plt.Axes:
    """
    DESCRIPTION
    plot the fitted model, observations and accepted samples for a fitted model with two predictor variable.

    PARAMETERS
    ax (plt.Axes) - Axes to produce plot on.
    accepted_samples([([float],float)]) - Details of samples which were accepted [([pred_var_values],response_var_value)].
    observations([([float],float)]) - Observations made before sampling [([pred_var_values],response_var_value)].
    model_hat(Model) - Fitted model.
    model_star(Model) - True model.
    plot_true(bool) - Whether to plot the true parameter value.

    RETURNS
    plt.Axes - The axis which plot was made on.
    """
    ax.set_title("Accepted Samples")
    ax.set_xlabel("Predictor Var. 1")
    ax.set_ylabel("Predictor Var. 2")
    ax.set_zlabel("Response Var.")

    # plot accepted samples
    x1_sam=[x[0][0] for x in accepted_samples]
    x2_sam=[x[0][1] for x in accepted_samples]
    s_sam =[x[1] for x in accepted_samples]
    ax.scatter(x1_sam,x2_sam,s_sam,c="orange",label="Accepted Samples")

    # plot observations
    x1_obs=[x[0][0] for x in observations]
    x2_obs=[x[0][1] for x in observations]
    s_obs =[x[1] for x in observations]
    ax.scatter(x1_obs,x2_obs,s_obs,c="red",label="Observations")

    # plot models
    x1s=np.linspace(var_ranges[0][0],var_ranges[0][1],100)
    x2s=np.linspace(var_ranges[1][0],var_ranges[1][1],100)
    X1,X2=np.meshgrid(x1s,x2s)

    # plot predicted model
    S_hat=[[model_hat.calc([x1,x2]) for x2 in x2s] for x1 in x1s]
    S_hat=np.array(S_hat)
    c1=ax.plot_surface(X1,X2,S_hat,color="orange",label="Predicted Model")

    # plot best model
    if (plot_truth):
        S_star=[[model_star.calc([x1,x2]) for x2 in x2s] for x1 in x1s]
        S_star=np.array(S_star)
        c2=ax.plot_surface(X1,X2,S_star,color="red",label="True Model")

    # prevents error when plotting legend
    c1._facecolors2d=c1._facecolors3d
    c1._edgecolors2d=c1._edgecolors3d
    c2._facecolors2d=c2._facecolors3d
    c2._edgecolors2d=c2._edgecolors3d

    ax.legend()

    return ax

"""
    SAMPLING STAGE
"""

def __sample_stage_fixed_sample_size(DESIRED_SAMPLE_SIZE:int,EPSILON:float,KERNEL:"func",
    x_obs:[[float]],s_obs:[float],PRIORS:["stats.Distribution"],x_samplers:["stats.Distribution"],model_t:Model) -> ( [[float]] , [([float],float)] ):
    """
    DESCRIPTION
    sample from parameter priors until a sufficient number of samples are close to observed parameters.

    PARAMETERS
    DESIRED_SAMPLE_SIZE (int) - Algorithm terminates after collection this many sample sufficient samples.
    EPSILON (int) - How close a sample must be to observations to be accepted. (Use in `KERNEL()`)
    ...

    RETURNS
    [[float]] - parameter sampels accepted
    [([float]),float] - Observations accepted ([predictor_variables],response_variable)
    """
    SAMPLES=[]

    i=0 # count total number of samples
    # Sample-Rejection Stage
    while (len(SAMPLES)<DESIRED_SAMPLE_SIZE):
        # sample parameters
        theta_t=[pi_i.rvs(1)[0] for pi_i in PRIORS] # sample a single value from each parameter-prior

        # simulate values
        model_t.params=theta_t
        x_t=[sampler.rvs(1)[0] for sampler in x_samplers] # make multiple samples from theorised model
        s_t=model_t.calc(x_t) # add response variable

        # accept-reject
        norm_vals=[l2_norm(x_i+[s_i],x_t+[s_t]) for (x_i,s_i) in zip (x_obs,s_obs)]
        if (KERNEL(min(norm_vals),EPSILON)):
            SAMPLES.append((theta_t,(x_t,s_t)))

        i=i+1
        print("({:,}) {:,}/{:,}".format(i,len(SAMPLES),DESIRED_SAMPLE_SIZE),end="\r") # update user on sampling process

    print("\n")

    theta_samples=[x[0] for x in SAMPLES]
    accepted_obs=[x[1] for x in SAMPLES]

    return theta_samples,accepted_obs

def __sample_stage_best_samples(NUM_RUNS:int,SAMPLE_SIZE:int,
    x_obs:[[float]],s_obs:[float],PRIORS:["stats.Distribution"],x_samplers:["stats.Distribution"],model_t:Model) -> ( [[float]] , [([float],float)] ):
    """
    DESCRIPTION
    make a defined number of samples and keep only the best n.

    PARAMETERS
    NUM_RUNS (int) - Number of samples to make
    SAMPLE_SIZE (int) - Number of samples to keep
    ...

    RETURNS
    [[float]] - parameter sampels accepted
    [([float]),float] - Observations accepted ([predictor_variables],response_variable)
    """
    SAMPLES=[None for _ in range(SAMPLE_SIZE)]

    # Sample-Rejection Stage
    for i in range(NUM_RUNS):
        # sample parameters
        theta_t=[pi_i.rvs(1)[0] for pi_i in PRIORS] # sample a single value from each parameter-prior

        # simulate values
        model_t.params=theta_t
        x_t=[sampler.rvs(1)[0] for sampler in x_samplers] # make multiple samples from theorised model
        s_t=model_t.calc(x_t) # add response variable

        # accept-reject
        norm_vals=[l2_norm(x_i+[s_i],x_t+[s_t]) for (x_i,s_i) in zip (x_obs,s_obs)]
        best_norm=min(norm_vals)

        # insert into samples
        for j in range(max(0,SAMPLE_SIZE-i-1,SAMPLE_SIZE)):

            if (SAMPLES[j]==None) or (SAMPLES[j][0]>best_norm):
                if (j>0): SAMPLES[j-1]=SAMPLES[j]
                SAMPLES[j]=(best_norm,(theta_t,(x_t,s_t)))

        print("({:,})".format(i),end="\r") # update user on sampling process

    print("\n")

    # remove norm value from SAMPLES
    SAMPLES=[x[1] for x in SAMPLES]
    theta_samples=[x[0] for x in SAMPLES]
    accepted_obs=[x[1] for x in SAMPLES]

    return theta_samples,accepted_obs

def __sample_stage_multi_compare(NUM_RUNS:int,SAMPLE_SIZE:int,NUM_COMPARISONS:int,
    x_obs:[[float]],s_obs:[float],PRIORS:["stats.Distribution"],x_samplers:["stats.Distribution"],model_t:Model) -> ( [[float]] , [([float],float)] ):
    """
    DESCRIPTION
    make a defined number of parameter samples and keep the best.
    With each set of sampled parameters, multiple observations are made from the theorised model and compared to the given observations from the true model (x_obs,s_obs). The mean kernel value is used to compare the fit of each theorised model.

    PARAMETERS
    NUM_RUNS (int) - Number of samples to make
    SAMPLE_SIZE (int) - Number of samples to keep
    NUM_COMPARISONS (int) - Number of observations to make from theorised model (used to compare fit to true model)
    ...


    RETURNS
    [[float]] - parameter sampels accepted
    [([float]),float] - Observations accepted ([predictor_variables],response_variable)
    """
    SAMPLES=[None for _ in range(SAMPLE_SIZE)]

    # Sample-Rejection Stage
    for i in range(NUM_RUNS):
        # sample parameters
        theta_t=[pi_i.rvs(1)[0] for pi_i in PRIORS] # sample a single value from each parameter-prior

        # simulate values
        model_t.params=theta_t
        x_ts=[[sampler.rvs(1)[0] for sampler in x_samplers] for _ in range(NUM_COMPARISONS)] # make multiple samples from theorised model
        s_ts=[model_t.calc(x_t) for x_t in x_ts] # add response variable

        # accept-reject
        norm_vals=[[l2_norm(x_i+[s_i],x_t+[s_t]) for (x_i,s_i) in zip(x_obs,s_obs)] for (x_t,s_t) in zip(x_ts,s_ts)]
        best_norm=[min(x) for x in norm_vals]
        mean_norm=np.mean(best_norm)

        # insert into samples
        for j in range(max(0,SAMPLE_SIZE-i-1,SAMPLE_SIZE)):

            if (SAMPLES[j]==None) or (SAMPLES[j][0]>mean_norm):
                if (j>0): SAMPLES[j-1]=SAMPLES[j]
                SAMPLES[j]=(mean_norm,(theta_t,list(zip(x_ts,s_ts))))

        print("({:,})".format(i),end="\r") # update user on sampling process

    print("\n")

    # remove norm value from SAMPLES
    SAMPLES=[x[1] for x in SAMPLES]
    theta_samples=[x[0] for x in SAMPLES]

    accepted_obs=[]
    for x in SAMPLES:
        for y in x[1]:
            accepted_obs.append(y)
    # accepted_obs=[y for y in x[1] for x in SAMPLES]

    return theta_samples,accepted_obs

"""
   ABC
"""
def abc_general(true_model=None,fitting_model=None,priors=None,n_obs=100,var_ranges=None,sampling_details={"sampling_method":"best_samples","sample_size":1000,"num_runs":10000}) -> Model:
    """
    DESCRIPTION
    Approximate Bayesian Computation which terminates when a sufficient number of samples have been accepted.
    A LinearModel with two parameters is used.

    PARAMETERS
    true_model (Model) - implicit model to fit for.
    fitting_model (Model) - the model you wish to fit to the true model (parameters are irrelevant).
    priors(stats.distribution,stats.distribution) - Priors to use for model parameters of `fitting_model` (default=+/-3 uniform around true value).
    n_obs (int) - Number of observations from true model used (default=100).
    var_ranges ([(int,int)]) - Range of each predictor variable in `fitting_model` (default=(0,100) for all variables).
    sampling_details (dict) - Specification of sampling method to use (see README.md)

    RETURNS
    Model = Model fitted by the algorithm
    """
    # define target model
    THETA_STAR=true_model.params if (true_model) else (stats.uniform(0,100).rvs(1)[0],stats.uniform(0,10).rvs(1)[0])
    MODEL_STAR=true_model if (true_model) else LinearModel(2,THETA_STAR)

    # define model to fit
    if (fitting_model):
        plot_truth=False
        model_t=fitting_model.blank_copy()
    else:
        plot_truth=True
        model_t=MODEL_STAR.blank_copy()

    # define true model
    print("True Model - {}\n".format(str(MODEL_STAR)))

    # verify inputs
    if (var_ranges) and (len(var_ranges)!=model_t.n_vars): raise Exception("Incorrect number of `var_ranges` provided. (Exp={})".format(model_t.n_vars))
    if (priors) and (len(priors)!=model_t.n_params): raise Exception("Incorrect number of `priors` provided. (Exp={})".format(model_t.n_params))

    # define priors for parameters
    PRIORS=priors if (priors) else [stats.uniform(THETA_STAR[i]-8,25) for i in range(model_t.n_params)]

    # parameter sampling details
    VAR_RANGES=var_ranges if (var_ranges) else [(0,100) for _ in range(model_t.n_vars)]
    N_OBS=n_obs

    # generate observations from target model
    x_samplers=[stats.uniform(r[0],r[1]-r[0]) for r in VAR_RANGES]
    x_obs=[[sampler.rvs(1)[0] for sampler in x_samplers] for _ in range(N_OBS)]
    if (MODEL_STAR.n_vars==1): x_obs=sorted(x_obs,key=(lambda x:x[0]))
    s_obs=[MODEL_STAR.calc(x) for x in x_obs]

    # perform sampling
    if (sampling_details["sampling_method"]=="best_samples"): # keep only best
        NUM_RUNS=   sampling_details["num_runs"]
        SAMPLE_SIZE=sampling_details["sample_size"]

        THETA_SAMPLES,ACCEPTED_OBS=__sample_stage_best_samples(NUM_RUNS,SAMPLE_SIZE,x_obs,s_obs,PRIORS,x_samplers,model_t)

    elif (sampling_details["sampling_method"]=="fixed_number"): # keep all which fulfil criteria
        SAMPLE_SIZE=sampling_details["sample_size"]
        EPSILON=    sampling_details["epsilon"]
        KERNEL=     sampling_details["kernel"]

        THETA_SAMPLES,ACCEPTED_OBS=__sample_stage_fixed_sample_size(SAMPLE_SIZE,EPSILON,KERNEL,x_obs,s_obs,PRIORS,x_samplers,model_t)

    elif (sampling_details["sampling_method"]=="multi_compare"):
        NUM_RUNS       =sampling_details["num_runs"]
        SAMPLE_SIZE    =sampling_details["sample_size"]
        NUM_COMPARISONS=sampling_details["num_comparisons"]

        THETA_SAMPLES,ACCEPTED_OBS=__sample_stage_multi_compare(NUM_RUNS,SAMPLE_SIZE,NUM_COMPARISONS,x_obs,s_obs,PRIORS,x_samplers,model_t)

    else:
        raise Exception("Must specify valid sampling details")

    # best fit model
    theta_hat=[np.mean([s[i] for s in THETA_SAMPLES]) for i in range(model_t.n_params)] # posterior sample mean
    model_hat=model_t.blank_copy()
    model_hat.params=theta_hat

    # plot results
    observations=list(zip(x_obs,s_obs))
    fig=plt.figure()
    __plot_results(fig,model_hat,MODEL_STAR,THETA_SAMPLES,ACCEPTED_OBS,PRIORS,VAR_RANGES,observations,plot_truth)

    print("Best Model - {}".format(str(model_hat)))

    return model_hat
