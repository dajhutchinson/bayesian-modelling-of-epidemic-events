"""
TODO
- add noise
- Assess fit of two models
"""
from math import exp
from matplotlib import pyplot as plt
from scipy import stats
import numpy as np

class Model():

    def __init__(self):
        self.n_params=None # number of parameters
        self.n_vars= None # number of variables
        self.params=None # [float] of parameter values
        self.var_sample_vals=None # variable values to use for sampling

    def blank_copy(self) -> "Model":
        """
        Create a copy of the model but with "blank" parameter values
        """
        raise Exception("`blank_copy` not implemented")

    def calc(self,x:[float],noise=True) -> float:
        """
        Calculate value of response variable, given values of the predictor variables x

        PARAMS
        x ([float]) - Values of predictor variables
        noise (bool) - Whether to include gaussian noise in plot (default=False)

        RETURNS
        float: value of response variable
        """
        raise Exception("`calc` method not implemented.")

    def plot(self,ax=None,var_ranges=None,noise=False) -> plt.Axes:
        """
        Plot model

        PARAMS
        ax (plt.Axes) - Axis to plot on (default=None) (if None then will perform plt.show())
        var_ranges ([(float,float)]) - Range of values for variables, to plot within (default=None)
        noise (bool) - Whether to include gaussian noise in plot (default=False)

        RETURNS
        plt.Axes - axis which plot was made on
        """
        show=False
        if (self.n_vars<1 or self.n_vars>2): raise Exception("Only plot models with 1 or 2 variables.")
        if (not ax):
            ax=plt.axes() if (self.n_vars==1) else plt.axes(projection="3d")
            show=True

        # prepare variable ranges
        if not (var_ranges):
            var_ranges=[]
            for i in range(self.n_vars):
                var_ranges.append((0,1))
        elif (len(var_ranges)!=self.n_vars): raise Exception("Must define ranges for all variables.")
        else:
            for r in var_ranges:
                if (r[1]<r[0]): raise Exception("Invalid range in var_ranges.")

        # plots
        if (self.n_vars==1):
            xs=np.linspace(var_ranges[0][0],var_ranges[0][1],100)
            ys=[self.calc([x],noise) for x in xs]

            if (noise==False): ax.plot(xs,ys)
            else: ax.scatter(xs,ys)

            ax.set_title(self.__str__())
            ax.set_xlabel(self.var_names[0])
            ax.set_ylabel("Response")

        elif (self.n_vars==2):
            x1s=np.linspace(var_ranges[0][0],var_ranges[0][1],100)
            x2s=np.linspace(var_ranges[1][0],var_ranges[1][1],100)
            X1,X2=np.meshgrid(x1s,x2s)

            Z=[[self.calc([x1,x2],noise) for x2 in x2s] for x1 in x1s]
            Z=np.array(Z)

            if (noise==False): ax.plot_surface(X1,X2,Z,cmap="viridis", edgecolor="none")
            else: ax.scatter(X1,X2,Z,cmap="viridis",edgecolor="none")

            ax.set_title(self.__str__())
            ax.set_xlabel(self.var_names[0])
            ax.set_ylabel(self.var_names[1])

        if (show): plt.show()

        return ax

    def sample(self) -> [float]:
        """
        DESCRIPTION
        return a sequence of observations from the model. Same result returned everytime this is called

        RETURNS
        [float] - sequence of observations
        """
        return [self.calc(x) for x in self.var_sample_vals]

class LinearModel(Model):

    def __init__(self,n:int,params:[float],var_sample_vals:[[float]],var_names=None,noise=0):
        """
        REQUIRED
        n (int):          number of model parameters
        params ([float]): value for model parameters
        var_sample_vals ([[float]]): variable to use in `sample`

        OPTIONAL
        var_names ([str]): names of variables, excluding bias (for printing only)
        noise (float):    variance of additive gaussian noise ~ N(0,noise) (default=0)
        """
        if (n!=len(params)):
            raise Exception("Wrong number of parameters provided. (n!=len(params))")

        self.n_params=n
        self.n_vars=n-1
        self.params=[float(p) for p in params]

        for x in var_sample_vals:
            if (len(x)!=self.n_vars): raise Exception("Invalid `var_sample_vals`. Must be list of lists of floats, each sublist of length n_params({})".format(self.n_params))
        self.var_sample_vals=var_sample_vals

        self.var_names=var_names
        if (var_names):
            if not (type(var_names)==list and len(var_names)==n-1 and type(var_names[0])==str):
                raise Exception("Invalid `var_names`. Must be list of strings of length n-1({}).".format(n-1))
            self.var_names=var_names
        else:
            self.var_names=["X{}".format(i) for i in range(n-1)]

        self.noise=noise
        if (noise==0): self.add_noise=(lambda : 0)
        else: self.add_noise=(lambda:stats.norm(0,np.sqrt(noise)).rvs(1)[0])

    def calc(self,x:[float],noise=True) -> float:
        """
        Calculate value of response variable, given values of the predictor variables x

        PARAMS
        x ([float]) - Values of predictor variables
        noise (bool) - Whether to add additive gaussian noise to returned value (don't want to when plotting true model)

        RETURNS
        float: value of response variable
        """
        if (len(x)!=self.n_vars): return None
        y=self.params[0]+sum([x[i]*self.params[i+1] for i in range(self.n_vars)])
        if (noise): y+=self.add_noise()

        return y

    def blank_copy(self) -> "LinearModel":
        temp_params=[1 for _ in range(self.n_params)]
        return LinearModel(self.n_params,temp_params,self.var_sample_vals,var_names=self.var_names)

    def __str__(self):
        """
        print model
        """
        print_str=""
        if (self.params[0]!=0):
            print_str="{:.5f}".format(self.params[0])
        for i in range(1,self.n_params):
            if (self.params[i]!=0):
                sign=self.params[i]/abs(self.params[i])
                print_str+="-" if (sign==-1) else "+"
                print_str+="{:.5f}*{}".format(abs(self.params[i]),self.var_names[i-1])

        if (self.noise!=0):
            print_str+=" + N(0,{})".format(self.noise)
        return print_str

class ExponentialModel(Model):

    def __init__(self,params:[float],var_sample_vals:[[float]],var_names=None,noise=0):
        """
        REQUIRED
        params ([float]): value for model parameters
        var_sample_vals ([[float]]): variable to use in `sample`

        OPTIONAL
        var_names ([str]): names of parameters, excluding bias (for printing only)
        noise (float):    variance of additive gaussian noise ~ N(0,noise) (default=0)
        """
        self.n_params=2
        self.n_vars=1

        if (len(params)!=2):
            raise Exception("Wrong number of parameters provided. (len(params)!=2)")

        self.params=[float(p) for p in params]

        for x in var_sample_vals:
            if (len(x)!=self.n_vars): raise Exception("Invalid `var_sample_vals`. Must be list of lists of floats, each sublist of length n_params({})".format(self.n_params))
        self.var_sample_vals=var_sample_vals

        if (var_names):
            if not (type(var_names)==list and len(var_names)==1 and type(var_names[0])==str):
                raise Exception("Invalid `var_names`. Must be list of strings of length 1.")
            self.var_names=var_names
        else:
            self.var_names=["X{}".format(i) for i in range(1)]

        self.noise=noise
        if (noise==0): self.add_noise=(lambda:0)
        else: self.add_noise=(lambda:stats.norm(0,np.sqrt(noise)).rvs(1)[0])

    def calc(self,x:[float],noise=True) -> float:
        """
        Calculate value of response variable, given values of the predictor variables x

        PARAMS
        x ([float]) - Values of predictor variables
        noise (bool) - Whether to add additive gaussian noise to returned value (don't want to when plotting true model)

        RETURNS
        float: value of response variable
        """
        y=self.params[0]*exp(x[0]*self.params[1])
        if (noise): y+=self.add_noise()

        return y

    def blank_copy(self) -> "ExponentialModel":
        temp_params=[1 for _ in range(self.n_params)]
        return ExponentialModel(temp_params,self.var_sample_vals,var_names=self.var_names)

    def __str__(self):
        """
        print model
        """
        print_str="{:.5f}*exp({:.5f}*{})".format(self.params[0],self.params[1],self.var_names[0])
        if (self.noise!=0):
            print_str+=" + N(0,{})".format(self.noise)
        return print_str

class GeneralLinearModel(Model):

    def __init__(self,n_params:int,n_vars:int,func:"function",theta_star:[float],var_sample_vals:[[float]],var_names=None,noise=0):
        """
        REQUIRED
        n_params (int) - number of parameters in model
        n_vars (int) - number of vars in model
        func (function) - model function
        theta_star ([float]): true model parameters
        var_sample_vals ([[float]]): variable to use in `sample`

        OPTIONAL
        var_names ([string]): readable name for each variable (used for plots)
        noise (float):    variance of additive gaussian noise ~ N(0,noise) (default=0)
        """

        if (n_params!=len(theta_star)): raise Exception("Incorrect number of parameters provided `(n_params!=len(theta_star))`")
        if (var_names):
            if not (type(var_names)==list and len(var_names)==1 and type(var_names[0])==str):
                raise Exception("Invalid `var_names`. Must be list of strings of length n_vars({}).".format(n_vars))
            self.var_names=var_names
        else:
            self.var_names=["X{}".format(i) for i in range(1)]

        self.n_params=n_params
        self.n_vars  =n_vars
        self.func= func
        self.params=theta_star

        for x in var_sample_vals:
            if (len(x)!=self.n_vars): raise Exception("Invalid `var_sample_vals`. Must be list of lists of floats, each sublist of length n_params({})".format(self.n_params))
        self.var_sample_vals=var_sample_vals

        self.noise=noise
        if (noise==0): self.add_noise = (lambda :0)
        else: self.add_noise=(lambda:stats.norm(0,np.sqrt(noise)).rvs(1)[0])

        self.calc=(lambda x,noise=True: self.func(x,self.params) + noise*self.add_noise()) # apply `self.func` to passed variable values `x` with true params `self.params`

    def blank_copy(self) -> "GeneralLinearModel":
        temp_params=[1 for _ in range(self.n_params)]
        return GeneralLinearModel(self.n_params,self.n_vars,self.func,temp_params,self.var_sample_vals,var_names=self.var_names)

class ManyModels():

    def __init__(self,n_vars:int,n_models:int,models:["Models"],var_sample_vals:[[float]]):
        """
        REQUIRED
        n_vars (int) - Number of variables each model has (all models must have the same).
        n_models (int) - Number of models
        models ([Model]) - Models in group
        var_sample_vals ([[float]]): variable to use in `sample`

        OPTIONAL
        """
        if (len(models)!=n_models): raise Exception("Incorect number of models passed (len(models)!=n_models)!")
        for (i,model) in enumerate(models):
            if (type(model) not in [LinearModel,ExponentialModel,GeneralLinearModel]): raise Exception("Not all objects in `models` are Models.")
            if (model.n_vars!=n_vars): raise Exception("Model {} does not have {} variable(s) (act={})".format(i,n_vars,model.n_vars))

        self.n_vars=n_vars
        self.n_models=n_models
        self.models=models

        for x in var_sample_vals:
            if (len(x)!=self.n_vars): raise Exception("Invalid `var_sample_vals`. Must be list of lists of floats, each sublist of length n_params({})".format(self.n_params))
        self.var_sample_vals=var_sample_vals

    def blank_copy(self) -> "ManyModels":
        blank_models=[model.blank_copy() for model in self.models]
        return ManyModels(self.n_vars,self.n_models,blank_models)

    def plot(self,var_ranges=None,noise=False):

        if (self.n_vars<1 or self.n_vars>2): raise Exception("Only plot models with 1 or 2 variables.")

        fig=plt.figure()
        plt.subplots_adjust(left=.05,right=.95,bottom=.05,top=.95)
        projection=None if (self.n_vars==1) else "3d"

        for (i,model) in enumerate(self.models):
            ax=fig.add_subplot(1,self.n_models,i+1,projection=projection)
            ax.set_title("Model {}".format(i))
            model.plot(ax=ax,var_ranges=var_ranges,noise=noise)

        plt.get_current_fig_manager().window.state("zoomed")
        plt.show()

    def calc(self,x:[float],noise=True) -> [float]:
        return [model.calc(x,noise) for model in self.models]

    def __str__(self):
        print_str="{} models each with {} variables.\nTrue Models\n".format(self.n_vars,self.n_models)
        for (i,model) in enumerate(self.models):
            print_str+="({})\t{}".format(i,str(model))
            if (i<self.n_models):print_str+="\n"
        return print_str
