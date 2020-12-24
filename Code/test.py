from scipy import stats
from matplotlib import pyplot as plt
import numpy as np

from Models import Model, LinearModel, ExponentialModel, GeneralLinearModel, ManyModels
import ABC

var_vals=np.linspace(-5,5,100)
var_vals_1d=[[x]*1 for x in var_vals]
var_vals_2d=[[x]*2 for x in var_vals]
var_vals_3d=[[x]*3 for x in var_vals]

lm_2=LinearModel(2,[1,2],var_vals_1d,var_names=["age"])
lm_3=LinearModel(3,[1,2,3],var_vals_2d)
lm_4=LinearModel(4,[10,20,30,40],var_vals_3d)

lm_2_noise=LinearModel(2,[1,2],var_vals_1d,var_names=["age"],noise=.1)
lm_3_noise=LinearModel(3,[1,2,3],var_vals_2d,noise=4)

em=ExponentialModel([1,2],var_vals_1d,var_names=["age"])

em_noise=ExponentialModel([1,2],var_vals_1d,var_names=["age"],noise=1)

sampling_details={"sampling_method":"fixed_number","sample_size":1000,"epsilon":4,"kernel":ABC.uniform_kernel}
theta_star=[10,20,-1]; f=(lambda x,theta: theta[0]+theta[1]*x[0]**2+theta[2]*(x[0]**3))
glm=GeneralLinearModel(3,1,f,theta_star,var_vals_1d)

glm_noise=GeneralLinearModel(3,1,f,theta_star,var_vals_1d,noise=10000)

"""
    FIXED NUMBER OF SAMPLES
# fixed number of samples - EM with a GLM
sampling_details={"sampling_method":"fixed_number","sample_size":1000,"epsilon":1,"kernel":ABC.epanechnikov_kernel}
theta_star=[1,1,1]; f=(lambda x,theta: theta[0]+theta[1]*x[0]+theta[2]*(x[0]**2))
fitting_glm=GeneralLinearModel(3,1,f,theta_star,var_vals_1d)
priors=[stats.uniform(0,10),stats.uniform(0,5),stats.uniform(0,5)]

ABC.abc_general(true_model=em,fitting_model=fitting_glm,priors=priors,var_ranges=[(-5,5)],sampling_details=sampling_details)

# fixed number of samples - LM
sampling_details={"sampling_method":"fixed_number","sample_size":1000,"epsilon":4,"kernel":ABC.uniform_kernel}
ABC.abc_general(true_model=lm_2,sampling_details=sampling_details)

# fixed number of samples - GLM
priors=[stats.uniform(5,10),stats.uniform(15,10),stats.uniform(-2,3)]

ABC.abc_general(true_model=glm,priors=priors,n_obs=500,var_ranges=[(0,20)],sampling_details=sampling_details)
"""

"""
    BEST OF SAMPLES
# best samples for LM
sampling_details={"sampling_method":"best_samples","sample_size":100,"num_runs":10000}
ABC.abc_general(true_model=lm_2,sampling_details=sampling_details)
"""

"""
BEST OF MULTI-COMPARISON SAMPLING
# multi-comaprisons - LM
sampling_details={"sampling_method":"multi_compare","sample_size":10,"num_runs":1000,"num_comparisons":10}
ABC.abc_general(true_model=lm_2,sampling_details=sampling_details)

# multi-comparisons - EM
sampling_details={"sampling_method":"multi_compare","sample_size":10,"num_runs":1000,"num_comparisons":10}
priors=[stats.uniform(0,2),stats.uniform(1,3)]
ABC.abc_general(true_model=em,sampling_details=sampling_details,priors=priors,var_ranges=[(-5,5)])

# multi-comparison - GLM
sampling_details={"sampling_method":"multi_compare","sample_size":10,"num_runs":1000,"num_comparisons":10}
theta_star=[10,20,-1]; f=(lambda x,theta: theta[0]+theta[1]*x[0]**2+theta[2]*(x[0]**3))
glm=GeneralLinearModel(3,1,f,theta_star,var_vals_1d)
priors=[stats.uniform(5,10),stats.uniform(15,10),stats.uniform(-2,3)]

ABC.abc_general(true_model=glm,priors=priors,n_obs=500,var_ranges=[(0,20)],sampling_details=sampling_details)

# multi-comparison - GLM
theta_star=[1,1/4]; f=(lambda x,theta:theta[0]*(x[0]**2)+theta[1]*(x[0]**5)) # x^2+(x^5)/4
glm=GeneralLinearModel(2,1,f,theta_star,var_vals_1d)
priors=[stats.uniform(0,2),stats.uniform(0,1)]
sampling_details={"sampling_method":"multi_compare","sample_size":100,"num_runs":1000,"num_comparisons":10}
ABC.abc_general(true_model=glm,priors=priors,var_ranges=[(-2,2)],sampling_details=sampling_details)

# multi-comaprisons - LM \w noise
sampling_details={"sampling_method":"multi_compare","sample_size":10,"num_runs":1000,"num_comparisons":10}
ABC.abc_general(true_model=lm_2_noise,sampling_details=sampling_details)

# multi-comparisons - EM
sampling_details={"sampling_method":"multi_compare","sample_size":10,"num_runs":10000,"num_comparisons":10}
priors=[stats.uniform(0,2),stats.uniform(1,3)]
ABC.abc_general(true_model=em_noise,sampling_details=sampling_details,priors=priors,var_ranges=[(0,3)])
"""

mm=ManyModels(1,2,[em_noise,lm_2_noise],var_vals_1d)
print(mm)
mm.plot(noise=True,var_ranges=[(-1,3)])
