"""
PARAMETERS
prior for parameters
Set of observed summary statistic values
"""

from scipy import stats
import matplotlib.pyplot as plt
from Models import LinearModel,ExponentialModel
import numpy as np

def uniform_kernel(x:float,epsilon:float) -> bool:
    return abs(x)<=epsilon

def l2_norm(s:(float),s_obs:(float)) -> float:
    return sum([(x-y)**2 for (x,y) in zip(s,s_obs)])**.5

def abc_fixed_sample_size(sample_size=1000,model=None,priors=None,n_obs=100,epsilon=.1,var_ranges=[(0,100)]):
    """
    Approximate Bayesian Computation which terminates when a sufficient number of samples have been accepted.
    A LinearModel with two parameters is used.

    PARAMETERS
    sample_size(int) = desired sample size (default=1,000).
    theta_star((int,int)) = parameters of the true model (default=each chosen uniformly at random from [0,10]).
    priors(stats.distribution,stats.distribution) = Priors to use for model parameters (default=+/-3 uniform around true value).
    n_obs (int)= Number of observations from true model used (default=100).
    epsilon (float) = Acceptable values from kernel (default=.1).
    var_ranges ([(int,int)]) = Range of each predictor variable in model (default=((0,100))).

    RETURNS
    None
    """

    DESIRED_SAMPLE_SIZE=sample_size
    THETA_STAR=model.params if (model) else (stats.uniform(0,100).rvs(1)[0],stats.uniform(0,10).rvs(1)[0])
    MODEL_STAR=model if (model) else LinearModel(2,THETA_STAR)
    N_OBS=n_obs
    EPSILON=epsilon
    SAMPLES=[]

    model_t=LinearModel(2,[0,0]) if (type(MODEL_STAR)==LinearModel) else ExponentialModel([0,0])

    # define true model
    print("True Model - {}\n".format(str(MODEL_STAR)))

    # generate observations from target model
    x_sampler=stats.uniform(var_ranges[0][0],var_ranges[0][1])
    x_obs=x_sampler.rvs(N_OBS)
    x_obs.sort()
    s_obs=[MODEL_STAR.calc([x]) for x in x_obs]

    # define priors for parameters
    pi_0=priors[0] if (priors) else stats.uniform(THETA_STAR[0]-5,10)
    pi_1=priors[1] if (priors) else stats.uniform(THETA_STAR[1]-5,10)

    i=0 # count total number of samples
    # Sample-Rejection Stage
    while (len(SAMPLES)<DESIRED_SAMPLE_SIZE):
        # sample parameters
        theta_t=(pi_0.rvs(1)[0],pi_1.rvs(1)[0]) # sample a single value from each parameter-prior

        # simulate values
        model_t.params=theta_t
        x_t=x_sampler.rvs(1) # make multiple samples from theorised model
        s_t=[model_t.calc([x]) for x in x_t]

        # plt.scatter(x_t,s_t)
        # plt.show()

        # accept-reject
        for (x,s) in zip(x_t,s_t):
            norm_vals=[l2_norm([x_i,s_i],[x,s]) for (x_i,s_i) in zip(x_obs,s_obs)] # calculate l2 norm between each samples and observations
            if (uniform_kernel(min(norm_vals),EPSILON)): # if sample is sufficiently close to an observation, store the sample and its parameters
                SAMPLES.append((theta_t,(x,s)))
            if (len(SAMPLES)>=DESIRED_SAMPLE_SIZE): break

        i=i+1
        print("({:,}) {:,}/{:,}".format(i,len(SAMPLES),DESIRED_SAMPLE_SIZE),end="\r") # update user on sampling process

    print("\n")
    theta_0_samples=[x[0][0] for x in SAMPLES] # accepted samples of theta_0
    theta_1_samples=[x[0][1] for x in SAMPLES] # accepted samples of theta_1

    # plot results
    fig,axs=plt.subplots(nrows=1,ncols=3)
    theta_hat=(np.mean(theta_0_samples),np.mean(theta_1_samples)) # use sample mean as best fit for parameters


    # plot of parameter theta_1
    axs[0].set_title("Sampled values for Theta_0")
    axs[0].hist(theta_0_samples,density=True)

    x=np.linspace(pi_0.ppf(0.01),pi_0.ppf(0.99), 100)
    axs[0].plot(x,pi_0.pdf(x), 'k-', lw=2, label='Prior')

    axs[0].vlines(THETA_STAR[0],ymin=0,ymax=axs[0].get_ylim()[1],colors="red")
    axs[0].vlines(theta_hat[0], ymin=0,ymax=axs[0].get_ylim()[1],colors="orange")
    axs[0].set_xlabel("Theta_0")

    print("Theta_0\nTrue {:.5f}\nMean {:.5f}\nMedian {:.5f}\n".format(THETA_STAR[0],np.mean(theta_0_samples),np.median(theta_0_samples)))

    # plot of parameter theta_2
    axs[1].set_title("Sampled values for Theta_1")
    axs[1].hist(theta_1_samples,density=True)

    x=np.linspace(pi_1.ppf(0.01),pi_1.ppf(0.99), 100)
    axs[1].plot(x,pi_1.pdf(x), 'k-', lw=2, label='Prior')

    axs[1].vlines(THETA_STAR[1],ymin=0,ymax=axs[1].get_ylim()[1],colors="red")
    axs[1].vlines(theta_hat[1], ymin=0,ymax=axs[1].get_ylim()[1],colors="orange")
    axs[1].set_xlabel("Theta_1")

    print("Theta_1\nTrue {:.5f}\nMean {:.5f}\nMedian {:.5f}\n".format(THETA_STAR[1],np.mean(theta_1_samples),np.median(theta_1_samples)))

    # plot of samples accepted
    axs[2].set_title("Accepted Samples")
    axs[2].scatter([x[1][0] for x in SAMPLES],[x[1][1] for x in SAMPLES])
    axs[2].scatter(x_obs,s_obs)
    axs[2].set_xlabel("Predictor Variable")
    axs[2].set_ylabel("Response Variable")

    # plot predicted model
    model_hat=LinearModel(2,theta_hat) if (type(MODEL_STAR)==LinearModel) else ExponentialModel(theta_hat)
    s_hat=[model_hat.calc([x]) for x in x_obs]
    axs[2].plot(x_obs,s_hat,c="red")

    print("Best Model - {}".format(str(model_hat)))

    plt.get_current_fig_manager().window.state('zoomed')
    plt.show()
