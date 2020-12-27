import matplotlib.pyplot as plt
import numpy as np

def plot_accepted_observations(ax:plt.Axes,n_obs:int,y_obs:[float],accepted_observations:[[float]]) -> plt.Axes:
    """
    DESCRIPTION

    PARAMETERS
    ax (plt.Axes) -
    n_obs (int) -
    y_obs ([float]) -
    accepted_observations ([[float]]) -

    RETURNS
    """
    xs=list(range(n_obs))
    for obs in accepted_observations:
        ax.scatter(xs,obs,c="blue",alpha=.05,marker="x")
    ax.scatter(xs,y_obs,c="green",alpha=1,label="From Truth")

    ax.set_title("Accepted Observations")
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.legend()

    return ax

def plot_parameter_posterior(ax:plt.Axes,name:str,accepted_parameter:[float],predicted_val:float,prior:"stats.Distribution") -> plt.Axes:
    ax.hist(accepted_parameter,density=True)
    x=np.linspace(prior.ppf(.01),prior.ppf(.99),100)
    ax.plot(x,prior.pdf(x), 'k-', lw=2, label='Prior')

    ymax=ax.get_ylim()[1]
    ax.vlines(predicted_val,ymin=0,ymax=ymax,colors="orange",label="Prediction")
    ax.set_xlabel(name)
    ax.set_title("Posterior for {}".format(name))
    ax.legend()

    return ax

def plot_summary_stats(ax:plt.Axes,name:str,accepted_s:[float],s_obs:float,s_hat:float) -> plt.Axes:

    ax.hist(accepted_s)
    ymax=ax.get_ylim()[1]
    ax.vlines(s_obs,ymin=0,ymax=ymax,colors="green",label="From Truth")
    ax.vlines(s_hat,ymin=0,ymax=ymax,colors="orange",label="From Fitted")

    ax.set_xlabel(name)
    ax.set_title("Accepted {}".format(name))
    ax.legend()

    return ax
