from math import exp
from matplotlib import pyplot as plt
import numpy as np

class Model():

    def __init__(self):
        self.n_params=None # number of parameters
        self.n_vars= None # number of variables
        self.params=None # [float] of parameter values

    def calc(self,x:[float]) -> float:
        """
        Calculate value of response variable, given values of the predictor variables x

        PARAMS
        x ([float]) - Values of predictor variables

        RETURNS
        float: value of response variable
        """
        raise Exception("`calc` method not implemented.")

    def plot(self,var_ranges=None):
        """
        Plot model
        """
        if (self.n_vars<1 or self.n_vars>2): raise Exception("Only plot models with 1 or 2 variables.")

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
            ys=[self.calc([x]) for x in xs]

            plt.plot(xs,ys)
            plt.title(self.__str__())
        elif (self.n_vars==2):
            x1s=np.linspace(var_ranges[0][0],var_ranges[0][1],100)
            x2s=np.linspace(var_ranges[1][0],var_ranges[1][1],100)
            X1,X2=np.meshgrid(x1s,x2s)

            Z=[[self.calc([x1,x2]) for x2 in x2s] for x1 in x1s]
            Z=np.array(Z)

            fig=plt.figure()
            ax =plt.axes(projection='3d')

            ax.plot_surface(X1,X2,Z,cmap='viridis', edgecolor='none')
            ax.set_title(self.__str__())
        plt.show()

class LinearModel(Model):

    def __init__(self,n:int,params:[float],var_names=None):
        """
        REQUIRED
        n (int):          number of model parameters
        params ([float]): value for model parameters

        OPTIONAL
        var_names ([str]): names of variables, excluding bias (for printing only)
        """
        if (n!=len(params)):
            raise Exception("Wrong number of parameters provided. (n!=len(params))")

        self.n_params=n
        self.n_vars=n-1
        self.params=params

        self.var_names=var_names
        if (var_names):
            if not (type(var_names)==list and len(var_names)==n-1 and type(var_names[0])==str):
                raise Exception("Invalid `var_names`. Must be list of strings of length n-1.")
            self.var_names=var_names
        else:
            self.var_names=["X{}".format(i) for i in range(n-1)]

    def calc(self,x:[float]) -> float:
        """
        Calculate value of response variable, given values of the predictor variables x

        PARAMS
        x ([float]) - Values of predictor variables

        RETURNS
        float: value of response variable
        """
        if (len(x)!=self.n_params-1): return None # -1 due to bias term

        return self.params[0]+sum([x[i]+self.params[i+1] for i in range(self.n_params-1)])

    def __str__(self):
        """
        print model
        """
        print_str=""
        if (self.params[0]!=0):
            print_str=str(self.params[0])
        for i in range(1,self.n_params):
            if (self.params[i]!=0):
                sign=self.params[i]/abs(self.params[i])
                print_str+="-" if (sign==-1) else "+"
                print_str+=str(abs(self.params[i]))+"*"+self.var_names[i-1]
        return print_str

class ExponentialModel(Model):

    def __init__(self,params:[float],var_names=None):
        """
        REQUIRED
        params ([float]): value for model parameters

        OPTIONAL
        var_names ([str]): names of parameters, excluding bias (for printing only)
        """
        self.n_params=2
        self.n_vars=1

        if (len(params)!=2):
            raise Exception("Wrong number of parameters provided. (len(params)!=2)")

        self.params=params

        if (var_names):
            if not (type(var_names)==list and len(var_names)==1 and type(var_names[0])==str):
                raise Exception("Invalid `var_names`. Must be list of strings of length 1.")
            self.var_names=var_names
        else:
            self.var_names=["X{}".format(i) for i in range(1)]

    def calc(self,x:[float]) -> float:
        """
        Calculate value of response variable, given values of the predictor variables x

        PARAMS
        x ([float]) - Values of predictor variables

        RETURNS
        float: value of response variable
        """
        # if (type(x)!=list or len(x)!=1): return None

        return self.params[0]*exp(x[0]*self.params[1])

    def __str__(self):
        """
        print model
        """
        return "{}*exp({}*{})".format(self.params[0],self.params[1],self.var_names[0])
