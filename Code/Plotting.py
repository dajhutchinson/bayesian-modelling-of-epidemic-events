import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from Models import Model

def plot_accepted_observations(ax:plt.Axes,n_obs:int,y_obs:[[float]],accepted_observations:[[float]],predicted_model:Model) -> plt.Axes:
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
    ax.scatter(xs,y_obs,c="green",alpha=1,label="y_obs")

    y_pred=predicted_model.observe(inc_noise=False)
    ax.plot(xs,y_pred,c="orange",label="Pred")

    ax.set_title("Accepted Observations")
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.legend()
    ax.margins(0)

    return ax

def plot_parameter_posterior(ax:plt.Axes,name:str,accepted_parameter:[float],predicted_val:float,prior:"stats.Distribution") -> plt.Axes:
    # plot prior used
    x=np.linspace(min(accepted_parameter+[prior.ppf(.01)-1]),max(accepted_parameter+[prior.ppf(.99)+1]),100)
    # x=np.linspace(prior.ppf(.01),prior.ppf(.99),100)
    ax.plot(x,prior.pdf(x),"k-",lw=2, label='Prior')

    # plot accepted  points
    ax.hist(accepted_parameter,density=True)

    # plot smooth posterior (ie KDE)
    density=stats.kde.gaussian_kde(accepted_parameter)
    ax.plot(x,density(x),"--",lw=2,c="orange",label="Posterior KDE")

    ymax=ax.get_ylim()[1]
    ax.vlines(predicted_val,ymin=0,ymax=ymax,colors="orange",label="Prediction")
    ax.set_xlabel(name)
    ax.set_title("Posterior for {}".format(name))
    ax.legend()
    ax.margins(0)

    return ax

def plot_summary_stats(ax:plt.Axes,name:str,accepted_s:[float],s_obs:float,s_hat:float) -> plt.Axes:

    ax.hist(accepted_s)
    ymax=ax.get_ylim()[1]
    ax.vlines(s_obs,ymin=0,ymax=ymax,colors="green",label="s_obs")
    ax.vlines(s_hat,ymin=0,ymax=ymax,colors="orange",label="From Fitted")

    ax.set_xlabel(name)
    ax.set_title("Accepted {}".format(name))
    ax.legend()
    ax.margins(0)

    return ax

def plot_MCMC_trace(ax:plt.Axes,name:str,accepted_parameter:[float],predicted_val:float) -> plt.Axes:
    x=list(range(1,len(accepted_parameter)+1))
    ax.plot(x,accepted_parameter,c="black")
    ax.hlines(predicted_val,xmin=0,xmax=len(accepted_parameter),colors="orange")

    ax.set_ylabel(name)
    ax.set_xlabel("t")
    ax.set_title("Trace {}".format(name))
    ax.margins(0)

    return ax

def plot_smc_posterior(ax:plt.Axes,name:str,parameter_values:[float],weights:[float],predicted_val:float) -> plt.Axes:
    density=stats.kde.gaussian_kde(parameter_values,weights=weights)
    x=np.linspace(min(parameter_values)*.9,max(parameter_values)*1.1,100)
    ax.plot(x,density(x),c="blue",label="Posterior")

    ymax=ax.get_ylim()[1]
    ax.vlines(predicted_val,ymin=0,ymax=ymax,colors="orange",label="Predicted Value")
    ax.set_xlabel(name)
    ax.set_ylabel("P")
    ax.set_title("Posterior for {}".format(name))
    ax.margins(0)
    ax.legend()

    return ax
