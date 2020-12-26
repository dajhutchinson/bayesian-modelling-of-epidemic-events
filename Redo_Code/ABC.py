from Models import Model
from scipy import stats

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
    SAMPLING METHODS
"""

"""
    ABC
"""

def abc_general(n_obs:int,y_obs:[[float]],fitting_model:Model,priors:["stats.distribution"],summary_stats=None) -> Model:
    """
    DESCRIPTION
    Approximate Bayesian Computation for the generative models defined in `Models.py`.

    PARAMETERS
    n_obs (int) - Number of observations available.
    y_obs ([[float]]) - Observations from true model.
    fitting_model (Model) - Model the algorithm will aim to fit to observations.
    priors (["stats.distribution"]) - Priors for the value of parameters of `fitting_model`.
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
    print(s_obs)
