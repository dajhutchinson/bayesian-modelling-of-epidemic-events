import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

class Model():

    def __init__(self):
        """
        DESCRIPTION
        generative models which I shall perform Approximate Bayesian Computation on.
        """
        raise NotImplementedError

        # all models have the following
        self.n_params=int # number of parameters
        self.params=list(float) # parameter values

        self.n_obs=int # number of observations made by `observe`
        self.dim_obs=int # dimension of each observation

        self.noise=float # variance of additive gaussian noise (default=0)
        self.param_labels=None # names for each parameter (used for plotting so optional)

    def update_params(self,new_params:[float]):
        """
        DESCRIPTION
        update the parameters of the model. the observations for `observe` need to be recalculated

        PARAMETERS
        new_paramas ([float]) - new parameter values
        """
        raise NotImplementedError

    def observe(self,inc_noise=True) -> [[float]]:
        """
        DESCRIPTION
        generate a sequence of `n_obs` observations from the model, each of dimension `dim_obs`.
        The same sequence is returned each time this function is called.
        sequence is ordered by `x_obs` so is best for `x_obs` to provide a useful ordering.

        PARAMETERS
        None

        RETURNS
        [[float]] - sequence of observations
        """
        raise NotImplementedError

    def copy(self,new_params:[float]) -> "Model":
        """
        DESCRIPTION
        create a copy of the model with new parameter values

        PARAMETERS
        new_params ([float]) - new parameter values

        RETURNS
        Model - New copy, with stated parameter values
        """
        raise NotImplementedError

    def plot_obs(self,constant_scale=False):
        """
        DESCRIPTION
        generate plots of observations returned by `observe`. There is a different plot for each dimension.

        PARAMETERS
        constant_scale (bool) - Whether the y-axis should have the same range or not (default=False)

        RETURNS
        None
        """

        param_labels = self.param_labels if (self.param_labels) else ["Dim {}".format(i+1) for i in range(self.dim_obs)]

        fig=plt.figure()
        plt.subplots_adjust(left=.05,right=.95,bottom=.05,top=.95)

        x=self.x_obs
        y_obs=self.observe()

        if (constant_scale):
            y_min=min([min(y) for y in y_obs])
            y_max=max([max(y) for y in y_obs])

            # Add padding
            y_range=y_max-y_min
            y_min-=y_range/20
            y_max+=y_range/20

        for i in range(self.dim_obs):
            y_obs_dim=[y[i] for y in y_obs]
            ax=fig.add_subplot(1,self.dim_obs,i+1)
            ax.set_title(param_labels[i])
            ax.scatter(x,y_obs_dim)
            if (constant_scale): ax.set_ylim(y_min,y_max)

        plt.show()

    def plot_obs_dim(self,dim:int,ax=None) -> plt.Axes:
        """
        DESCRIPTION
        Plot results of specific dimension of the observations from `observe`.

        PARAMETERS
        dim (int) - dimension you wish to plot.
        ax (plt.Axes) - Axes to make plot on (optional).

        RETURNS
        plt.Axes - axes containing new plot
        """
        if (type(dim)!=int): raise TypeError("`dim` must be an interger (not {})".format(type(dim)))
        if (dim<0 or dim>=self.dim_obs): raise ValueError("`dim` is out of range (exp in [0,{}])".format(self.dim_obs-1))

        show=False
        if (not ax):
            ax=plt.axes()
            show=True

        x=list(range(self.n_obs))
        y_obs=self.observe()

        y_obs_dim=[y[dim] for y in y_obs]
        ax.set_title("Dim {}".format(dim+1))
        ax.scatter(x,y_obs_dim)

        if (show): plt.show()

        return ax

class LinearModel(Model): # a+bx+cy+...

    def __init__(self,n_params:int,params:[float],n_vars:int,n_obs:int,x_obs:[[float]],noise=None):
        """
        DESCRIPTION

        PARAMETERS
        n_params (int) - number of parameters in model.
        params ([float]) - parameters of the model
        n_vars (int) - number of varaibles in model.
        n_obs (int) - number of observations generated by `observe()`
        x_obs ([[float]]) - variable values used in observations (dim = n_obs*n_vars)

        OPTIONAL PARAMETERS
        noise - variance of additive gaussian noise (default=None)
        """

        # valid inputs
        if not all([type(x)==int for x in [n_params,n_vars,n_obs]]): raise TypeError("n_params,n_vars,n_obs should all be integers.")
        if not all([x>=0 for x in [n_params,n_vars,n_obs]]): raise ValueError("n_params,n_vars,n_obs should all be >=0.")
        if (len(params)!=n_params): raise ValueError("Incorrect number of parameters passed. len(params)!=n_params.")
        if (n_vars+1!=n_params): raise ValueError("Expected one more parameter than variable.")
        if (len(x_obs)!=n_obs): raise ValueError("Inccorect number of observation points given. len(x_obs)!=n_obs.")
        if not all([len(x_obs_i)==n_vars for x_obs_i in x_obs]): raise TypeError("All elements of x_obs should have dimension n_vars ({})".formt(n_vars))
        if (noise) and (noise<0): raise ValueError("`noise` must be non-negative.")

        # specify model
        self.n_params=n_params
        self.params=params
        self.n_vars=n_vars
        self.n_obs=n_obs
        self.x_obs=x_obs
        self.dim_obs=1

        self.noise_var=noise if (noise) else 0
        self.add_noise=(lambda : stats.norm(0,np.sqrt(self.noise_var)).rvs(1)[0])
        self.param_labels=None

        # observations
        self.observations=[self._calc(x) for x in self.x_obs]

    def update_params(self,new_params):
        """
        DESCRIPTION
        update the parameters of the model. the observations for `observe` need to be recalculated

        PARAMETERS
        new_paramas ([float]) - new parameter values
        """
        if (len(new_params)!=self.n_params): raise ValueError("Incorrect number of parameters passed. len(params)!=n_params.")

        self.params=new_params
        # update observations observations
        self.observations=[self._calc(x) for x in self.x_obs]

    def observe(self,inc_noise=True) -> [[float]]:
        """
        DESCRIPTION
        generate a sequence of `n_obs` observations from the model, each of dimension `dim_obs`.
        The same sequence is returned each time this function is called/

        PARAMETERS
        inc_noise (bool) - whether to include noise in returned value (default=True)

        Returns
        [[float]] - sequence of observations (For LinearModel each observation is a 1d list)
        """
        if (inc_noise): return self.observations
        return [self._calc(x,False) for x in self.x_obs]

    def _calc(self,x:[float],inc_noise=True) -> [float]:
        """
        DESCRIPTION
        calculate the value of an observation at a specific point `x` in the varaible space.

        PARAMETERS
        x ([float]) - point to observe model at (len(x)=n_vars)

        OPTIONAL PARAMETERS
        inc_noise (bool) - whether to include noise in calculation (default=True)

        Returns
        [[float]] - value of model at observed point (for LinearModel this is a 1d list)
        """
        # valid x is in the model's variable space
        if (type(x)!=list): raise TypeError("`x` must be a `list` (not `{}`)".format(type(x)))
        if (len(x)!=self.n_vars): raise TypeError("`x` is of wrong dimension (Exp={})".format(len(x)))

        # calculate value
        y=self.params[0]
        for (x,param) in zip(x,self.params[1:]):
            y+=param*x
        if (inc_noise): y+=self.add_noise()
        return [y]

    def copy(self,new_params:[float]) -> "LinearModel":
        """
        DESCRIPTION
        create a copy of the model with new parameter values.
        does NOT copy noise over

        PARAMETERS
        new_params ([float]) - new parameter values

        RETURNS
        LinearModel - New copy, with stated parameter values
        """

        if (type(new_params)!=list): raise TypeError("`new_params` shoud be a list (not {})".format(type(new_params)))
        if (len(new_params)!=self.n_params): raise TypeError("`new_params` shoud of length `n_params` ({})".format(self.n_params))

        new_model=LinearModel(self.n_params,new_params,self.n_vars,self.n_obs,self.x_obs)
        return new_model

    def __str__(self):
        print_str="{:.3f}".format(self.params[0])
        for (i,p) in enumerate(self.params[1:]):
            if (p<0): print_str+="{:.3f}*x{:.3f}".format(p,i)
            elif (p>0): print_str+="+{:.3f}*x{}".format(p,i)

        if (self.noise_var!=0): print_str+="+N(0,{:.3f})".format(self.noise_var)
        return print_str

class ExponentialModel(Model): # ae^{xb}

    def __init__(self,params:[float],n_obs:int,x_obs:[[float]],noise=None):
        """
        DESCRIPTION

        PARAMETERS
        n_params (int) - number of parameters in model.
        params ([float]) - parameters of the model
        n_obs (int) - number of observations generated by `observe()`
        x_obs ([[float]]) - variable values used in observations (dim = n_obs*n_vars)

        OPTIONAL PARAMETERS
        noise - variance of additive gaussian noise (default=None)
        """

        # valid inputs
        if not all([type(x)==int for x in [n_obs]]): raise TypeError("n_obs should all be integers.")
        if not all([x>=0 for x in [n_obs]]): raise ValueError("n_obs should all be >=0.")
        if (len(params)!=2): raise ValueError("Incorrect number of parameters passed. len(params)!=2.")
        if (len(x_obs)!=n_obs): raise ValueError("Inccorect number of observation points given. len(x_obs)!=n_obs.")
        if not all([len(x_obs_i)==1 for x_obs_i in x_obs]): raise TypeError("All elements of x_obs should have dimension n_vars ({})".formt(1))

        # specify model
        self.n_params=2
        self.params=params
        self.n_vars=1
        self.n_obs=n_obs
        self.x_obs=x_obs
        self.dim_obs=1

        self.param_labels=None

        self.noise_var=noise if (noise) else 0
        self.add_noise=(lambda : stats.norm(0,np.sqrt(self.noise_var)).rvs(1)[0])

        # observations (ensure same noise each time `observe is called`)
        self.observations=[self._calc(x) for x in x_obs]

    def update_params(self,new_params):
        """
        DESCRIPTION
        update the parameters of the model. the observations for `observe` need to be recalculated

        PARAMETERS
        new_paramas ([float]) - new parameter values
        """
        if (len(new_params)!=self.n_params): raise ValueError("Incorrect number of parameters passed. len(params)!=n_params.")

        self.params=new_params
        # update observations observations
        self.observations=[self._calc(x) for x in self.x_obs]

    def observe(self,inc_noise=True) -> [[float]]:
        """
        DESCRIPTION
        generate a sequence of `n_obs` observations from the model, each of dimension `dim_obs`.
        The same sequence is returned each time this function is called/

        PARAMETERS
        None

        Returns
        [[float]] - sequence of observations (For LinearModel each observation is a 1d list)
        """
        if (inc_noise): return self.observations
        return [self._calc(x,False) for x in self.x_obs]

    def _calc(self,x:[float],inc_noise=True) -> [float]:
        """
        DESCRIPTION
        calculate the value of an observation at a specific point `x` in the varaible space.

        PARAMETERS
        inc_noise (bool) - whether to include noise in calculation (default=True)

        OPTIONAL PARAMETERS
        inc_noise (bool) -

        Returns
        [[float]] - value of model at observed point (for ExponentialModel this is a 1d list)
        """
        # valid x is in the model's variable space
        if (type(x)!=list): raise TypeError("`x` must be a `list` (not `{}`)".format(type(x)))
        if (len(x)!=self.n_vars): raise TypeError("`x` is of wrong dimension (Exp={})".format(len(x)))

        # calculate value
        y=[self.params[0]+np.exp(x[0]*self.params[1])]
        if (inc_noise): y+=self.add_noise()
        return y

    def copy(self,new_params:[float]) -> "ExponentialModel":
        """
        DESCRIPTION
        create a copy of the model with new parameter values

        PARAMETERS
        new_params ([float]) - new parameter values

        RETURNS
        ExponentialModel - New copy, with stated parameter values
        """

        if (type(new_params)!=list): raise TypeError("`new_params` shoud be a list (not {})".format(type(new_params)))
        if (len(new_params)!=self.n_params): raise TypeError("`new_params` shoud of length `n_params` ({})".format(self.n_params))

        new_model=ExponentialModel(new_params,self.n_obs,self.x_obs)
        return new_model

    def __str__(self):
        printing_str="{:.3f}*e^({:.3f}*x0)".format(self.params[0],self.params[1])
        if (self.noise_var!=0): printing_str+="+N(0,{:.3f})".format(self.noise_var)
        return printing_str

class SIRModel(Model):

    def __init__(self,params:(int,float,float),n_obs:int,x_obs:[int]):
        """
        DESCRIPTION
        Classical SIR model

        params ((int,int,float,float)) - (population_size,initial_infectied_population_size,beta,gamma)
        n_obs (int) - number of time-periods to run model for
        x_obs ([int]) - days on which to make observations
        """

        if (params[0]<params[1]): raise ValueError("Number of initially infected individuals cannot be greater than the population size.")

        # all models have the following
        self.n_params=4
        self.population_size=params[0]
        self.initially_infected=params[1]
        self.beta=params[2]
        self.gamma=params[3]
        self.params=params # parameter values

        self.param_labels=["Susceptible","Infectious","Removed"]

        self.n_obs=n_obs # number of observations made by `observe`
        self.dim_obs=3 # dimension of each observation

        self.noise=0 # variance of additive gaussian noise (default=0)

        self.x_obs=x_obs
        self.observations=self._calc(x_obs)

    def update_params(self,new_params:[float]):
        """
        DESCRIPTION
        update the parameters of the model. the observations for `observe` need to be recalculated

        PARAMETERS
        new_paramas ([float]) - new parameter values
        """
        if (len(new_params)!=self.n_params): raise ValueError("Incorrect number of parameters passed. len(params)!=n_params.")

        self.population_size=new_params[0]
        self.initially_infected=new_params[1]
        self.beta=new_params[2]
        self.gamma=new_params[3]

        self.observations=self._calc(self.x_obs)

    def observe(self,inc_noise=True) -> [[float]]:
        """
        DESCRIPTION
        generate a sequence of `n_obs` observations from the model, each of dimension `dim_obs`.
        The same sequence is returned each time this function is called.
        sequence is ordered by `x_obs` so is best for `x_obs` to provide a useful ordering.

        PARAMETERS
        None

        RETURNS
        [[float]] - sequence of observations
        """
        return self.observations

    def _calc(self,x_obs:[int]) -> [(int,int,int)]:
        """
        DESCRIPTION
        calculate the time-series of observations from the specified SIR model (using ODEs)

        PARAMS
        x_obs ([int]) - Days on which to make observations

        RETURNS
        [(int,int,int)] - time-series with each data-point being (S,I,R)
        """
        last_obs=(self.population_size-self.initially_infected,self.initially_infected,0)
        x_flat=[x[0] for x in x_obs]

        if 0 in x_flat: observations=[last_obs] # [(S,I,R)]
        else: observations=[]

        for t in range(1,max(x_flat)+1):
            new_infections=int(self.beta*((last_obs[0]*last_obs[1])/self.population_size))
            new_removed   =int(self.gamma*last_obs[1])

            d_S=-new_infections
            d_I=new_infections-new_removed
            d_R=new_removed

            new_obs=(last_obs[0]+d_S,last_obs[1]+d_I,last_obs[2]+d_R)
            if (t in x_flat): observations.append(new_obs)

            last_obs=new_obs

        return observations

    def copy(self,new_params:[float]) -> "Model":
        """
        DESCRIPTION
        create a copy of the model with new parameter values

        PARAMETERS
        new_params ([float]) - new parameter values

        RETURNS
        Model - New copy, with stated parameter values
        """
        if (type(new_params)!=list): raise TypeError("`new_params` shoud be a list (not {})".format(type(new_params)))
        if (len(new_params)!=self.n_params): raise TypeError("`new_params` shoud of length `n_params` ({})".format(self.n_params))

        new_model=SIRModel(new_params,self.n_obs,self.x_obs)
        return new_model

    def __str__(self) -> str:
        printing_str="Population Size={:,.1f}\n".format(self.population_size)
        printing_str+="Initially Infected={:,.1f}\n".format(self.initially_infected)
        printing_str+="Beta={:.3f}\n".format(self.beta)
        printing_str+="Gamma={:.3f}\n".format(self.gamma)
        printing_str+="R_0={:.3f}".format(self.beta/self.gamma)

        return printing_str

class GaussianMixtureModel_two(Model):

    def __init__(self,params:(int,float,float),n_obs:int,sd=(1,1)):
        """
        DESCRIPTION
        Gaussian Mixture Model with 2 gaussians with known variance (example from https://www.tandfonline.com/doi/pdf/10.1080/00949655.2020.1843169).
        Produces n observations, nx from one distribution and n(1-x) from another.
        Parameters to learn are the mean for each distribution and the relative weighting

        params ((float,float,float)) - (mean_1,mean_2,x) where x is the weight given to the first distribution
        n_obs (int) - number of observations to make from model
        sd ((float,float)) - standard deviation from each gaussian (default=(1,1))
        """

        if (params[2]>1) or (params[2]<0): raise ValueError("weight (params[2]) must be in [0,1].")

        # all models have the following
        self.n_params=3
        self.params=params # parameter values
        self.mu_1=params[0]
        self.mu_2=params[1]
        self.sigma_1=sd[0]
        self.sigma_2=sd[1]
        self.weight_1=params[2]
        self.weight_2=1-self.weight_1

        self.param_labels=None

        self.n_obs=n_obs # number of observations made by `observe`
        self.dim_obs=1 # dimension of each observation

        self.noise=0 # variance of additive gaussian noise (default=0)

        self.observations=self._calc()

    def update_params(self,new_params:[float]):
        """
        DESCRIPTION
        update the parameters of the model. the observations for `observe` need to be recalculated

        PARAMETERS
        new_paramas ([float]) - new parameter values
        """
        if (len(new_params)!=self.n_params): raise ValueError("Incorrect number of parameters passed. len(params)!=n_params.")

        self.params=new_params # parameter values
        self.mu_1=new_params[0]
        self.mu_2=new_params[1]
        self.weight_1=new_params[2]
        self.weight_2=1-self.weight_1

        self.observations=self._calc()

    def observe(self,inc_noise=True) -> [[float]]:
        """
        DESCRIPTION
        generate a sequence of `n_obs` observations from the model, each of dimension `dim_obs`.
        The same sequence is returned each time this function is called.
        sequence is ordered by `x_obs` so is best for `x_obs` to provide a useful ordering.

        PARAMETERS
        None

        RETURNS
        [[float]] - sequence of observations
        """
        if (inc_noise): return self.observations
        return self._calc(inc_noise=False)

    def _calc(self,inc_noise=True) -> [(int,int,int)]:
        """
        DESCRIPTION
        calculate the time-series of observations from the specified gaussian mixtures model

        RETURNS
        [(int,int,int)] - time-series with each data-point being (S,I,R)
        """

        n_obs_model_1=int(self.weight_1*self.n_obs)
        n_obs_model_2=self.n_obs-n_obs_model_1

        if (inc_noise):
            dist_1=stats.norm(loc=self.mu_1,scale=self.sigma_1)
            dist_2=stats.norm(loc=self.mu_2,scale=self.sigma_2)
        else:
            dist_1=stats.norm(loc=self.mu_1,scale=0)
            dist_2=stats.norm(loc=self.mu_2,scale=0)

        observations=list(dist_1.rvs(size=n_obs_model_1))+list(dist_2.rvs(size=n_obs_model_2))
        observations=[[x] for x in observations]

        return observations

    def copy(self,new_params:[float]) -> "Model":
        """
        DESCRIPTION
        create a copy of the model with new parameter values

        PARAMETERS
        new_params ([float]) - new parameter values

        RETURNS
        Model - New copy, with stated parameter values
        """
        if (type(new_params)!=list): raise TypeError("`new_params` shoud be a list (not {})".format(type(new_params)))
        if (len(new_params)!=self.n_params): raise TypeError("`new_params` shoud of length `n_params` ({})".format(self.n_params))

        new_model=GaussianMixtureModel_two(new_params,self.n_obs,sd=(self.sigma_1,self.sigma_2))
        return new_model

    def __str__(self) -> str:
        printing_str="X_1~Normal({:.3f},{:.3f})\n".format(self.mu_1,self.sigma_1**2)
        printing_str+="X_2~Normal({:.3f},{:.3f})\n".format(self.mu_2,self.sigma_2**2)
        printing_str+="X={:.3f}*X_1+{:.3f}*X_2".format(self.weight_1,self.weight_2)

        return printing_str
