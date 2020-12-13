"""
PARAMETERS
prior for parameters
Set of observed summary statistic values
"""

from scipy import stats
import matplotlib.pyplot as plt
from Models import Model,LinearModel,ExponentialModel
import numpy as np

def uniform_kernel(x:float,epsilon:float) -> bool:
    return abs(x)<=epsilon

def l2_norm(s:(float),s_obs:(float)) -> float:
    return sum([(x-y)**2 for (x,y) in zip(s,s_obs)])**.5

"""
   PLOTTING
"""

def __plot_results(fig:plt.Figure,model_hat:Model,model_star:Model,samples:[([float],([float],float))],priors:["stats.distribution"],var_ranges:[(float,float)],observations:[([float],float)]):

    n_plots=model_star.n_params
    if (n_plots)<=3: n_plots+=1
    plt.subplots_adjust(left=.05,right=.95,bottom=.05,top=.95)

    # plot parameter results
    for i in range(model_star.n_params):
        theta_i_samples=[x[0][i] for x in samples]
        ax=fig.add_subplot(1,n_plots,i+1)
        __plot_posterior(ax,"Theta_{}".format(i),model_star.params[i],theta_i_samples,priors[i])

    accepted_samples=[x[1] for x in samples]
    # 2d plot
    if (model_star.n_vars==1):
        ax=fig.add_subplot(1,n_plots,n_plots)
        __plot_2d_samples(ax,accepted_samples,observations,model_hat)

    # 2d plot
    elif (model_star.n_vars==2):
        ax=fig.add_subplot(1,n_plots,n_plots,projection="3d")
        __plot_3d_samples(ax,accepted_samples,observations,model_hat,model_star,var_ranges)

    plt.get_current_fig_manager().window.state('zoomed')
    plt.show()
    pass

def __plot_posterior(ax:plt.Axes,name:str,theta_star:float,theta_samples:[float],prior:"stats.Distribution") -> plt.Axes:

    print("{}\nTrue {:.5f}\nMean {:.5f}\nMedian {:.5f}\n".format(name,theta_star,np.mean(theta_samples),np.median(theta_samples)))

    ax.set_title("Posterior for {}".format(name))
    ax.hist(theta_samples,density=True)

    x=np.linspace(prior.ppf(0.01),prior.ppf(0.99), 100)
    ax.plot(x,prior.pdf(x), 'k-', lw=2, label='Prior')

    ax.vlines(theta_star,ymin=0,ymax=ax.get_ylim()[1],colors="red",label="Truth")
    ax.vlines(np.mean(theta_samples), ymin=0,ymax=ax.get_ylim()[1],colors="orange",label="Prediction")
    ax.set_xlabel("{}".format(name))
    ax.legend()

    return ax

# plot of samples accepted
def __plot_2d_samples(ax:plt.Axes,accepted_samples:[([float],float)],observations:[([float],float)],model_hat:Model) -> plt.Axes:
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

    ax.scatter(x_obs,s_obs,label="Observations")

    # plot predicted model
    s_hat=[model_hat.calc(x) for x in x_obs]
    ax.plot(x_obs,s_hat,c="red",label="Predicted Model")
    ax.legend()

    return ax

# plot of samples accepted
def __plot_3d_samples(ax:plt.Axes,accepted_samples:[([float],float)],observations:[([float],float)],model_hat:Model,model_star:Model,var_ranges:[(float,float)]) -> plt.Axes:
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
   ABC
"""
def abc_fixed_sample_size(sample_size=1000,model=None,priors=None,n_obs=100,epsilon=.1,var_ranges=None) -> Model:
    """
    Approximate Bayesian Computation which terminates when a sufficient number of samples have been accepted.
    A LinearModel with two parameters is used.

    PARAMETERS
    sample_size(int) = desired sample size (default=1,000).
    theta_star((int,int)) = parameters of the true model (default=each chosen uniformly at random from [0,10]).
    priors(stats.distribution,stats.distribution) = Priors to use for model parameters (default=+/-3 uniform around true value).
    n_obs (int)= Number of observations from true model used (default=100).
    epsilon (float) = Acceptable values from kernel (default=.1).
    var_ranges ([(int,int)]) = Range of each predictor variable in model (default=(0,100) for all variables).

    RETURNS
    Model = Model fitted by the algorithm
    """

    DESIRED_SAMPLE_SIZE=sample_size
    THETA_STAR=model.params if (model) else (stats.uniform(0,100).rvs(1)[0],stats.uniform(0,10).rvs(1)[0])
    MODEL_STAR=model if (model) else LinearModel(2,THETA_STAR)
    VAR_RANGES=var_ranges if (var_ranges) else [(0,100) for _ in range(model.n_vars)]
    N_OBS=n_obs
    EPSILON=epsilon
    SAMPLES=[]

    # define priors for parameters
    PRIORS=priors if (priors) else [stats.uniform(THETA_STAR[i]-8,25) for i in range(model.n_params)]

    model_t=LinearModel(MODEL_STAR.n_params,[1 for _ in range(MODEL_STAR.n_params)]) if (type(MODEL_STAR)==LinearModel) else ExponentialModel([1,1])
    # define true model
    print("True Model - {}\n".format(str(MODEL_STAR)))

    # generate observations from target model
    x_samplers=[stats.uniform(r[0],r[1]) for r in VAR_RANGES]
    x_obs=[[sampler.rvs(1)[0] for sampler in x_samplers] for _ in range(N_OBS)]
    s_obs=[MODEL_STAR.calc(x) for x in x_obs]

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
        if (uniform_kernel(min(norm_vals),EPSILON)):
            SAMPLES.append((theta_t,(x_t,s_t)))

        i=i+1
        print("({:,}) {:,}/{:,}".format(i,len(SAMPLES),DESIRED_SAMPLE_SIZE),end="\r") # update user on sampling process

    print("\n")

    # # best fit model
    # theta_hat=[np.mean([s[0][i] for s in SAMPLES]) for i in range(MODEL_STAR.n_params)] # posterior sample mean
    # model_hat=LinearModel(MODEL_STAR.n_params,theta_hat) if (type(MODEL_STAR)==LinearModel) else ExponentialModel(theta_hat)
    #
    # n_plots=MODEL_STAR.n_params
    # if (n_plots)<=3: n_plots+=1
    # fig=plt.figure()
    # plt.subplots_adjust(left=.05,right=.95,bottom=.05,top=.95)
    #
    # # plot parameter results
    # for i in range(MODEL_STAR.n_params):
    #     theta_i_samples=[x[0][i] for x in SAMPLES]
    #     ax=fig.add_subplot(1,n_plots,i+1)
    #     __plot_posterior(ax,"Theta_{}".format(i),THETA_STAR[i],theta_i_samples,PRIORS[i])
    #
    # # 2d plot
    # if (MODEL_STAR.n_vars==1):
    #     ax=fig.add_subplot(1,n_plots,n_plots)
    #     accepted_samples=[x[1] for x in SAMPLES]
    #     observations=list(zip(x_obs,s_obs))
    #     __plot_2d_samples(ax,accepted_samples,observations,model_hat)
    #
    # # 2d plot
    # elif (MODEL_STAR.n_vars==2):
    #     ax=fig.add_subplot(1,n_plots,n_plots,projection="3d")
    #     accepted_samples=[x[1] for x in SAMPLES]
    #     observations=list(zip(x_obs,s_obs))
    #     __plot_3d_samples(ax,accepted_samples,observations,model_hat,MODEL_STAR,VAR_RANGES)
    #
    # print("Best Model - {}".format(str(model_hat)))
    #
    # plt.get_current_fig_manager().window.state('zoomed')
    # plt.show()

    # best fit model
    theta_hat=[np.mean([s[0][i] for s in SAMPLES]) for i in range(MODEL_STAR.n_params)] # posterior sample mean
    model_hat=LinearModel(MODEL_STAR.n_params,theta_hat) if (type(MODEL_STAR)==LinearModel) else ExponentialModel(theta_hat)

    # plot results
    observations=list(zip(x_obs,s_obs))
    fig=plt.figure()
    __plot_results(fig,model_hat,MODEL_STAR,SAMPLES,PRIORS,VAR_RANGES,observations)

    print("Best Model - {}".format(str(model_hat)))

    return model_hat
