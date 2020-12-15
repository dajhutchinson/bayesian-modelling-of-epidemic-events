from Models import LinearModel, ExponentialModel, GeneralLinearModel
from scipy import stats
import ABC

lm_2=LinearModel(2,[1,2],["age"])
lm_3=LinearModel(3,[1,2,3])
lm_4=LinearModel(4,[10,20,30,40])

em=ExponentialModel([1,2],["age"])

theta_star=[10,20,-1]
f=(lambda x,theta: theta[0]+theta[1]*x[0]**2+theta[2]*(x[0]**3))
glm=GeneralLinearModel(3,1,f,theta_star)

# ABC.abc_best_samples(num_runs=10000,sample_size=1000)

theta_star=[1,1,1]; f=(lambda x,theta: theta[0]+theta[1]*x[0]+theta[2]*(x[0]**2))
fitting_glm=GeneralLinearModel(3,1,f,theta_star)

priors=[stats.uniform(0,10),stats.uniform(0,5),stats.uniform(0,5)]
sampling_details={"sampling_method":"fixed_number","sample_size":1000,"epsilon":.1}

ABC.abc_general(true_model=em,fitting_model=fitting_glm,priors=priors,var_ranges=[(-5,5)],sampling_details=sampling_details)

sampling_details={"sampling_method":"best_samples","sample_size":1000,"num_runs":10000}
ABC.abc_general(true_model=lm_2,sampling_details=sampling_details)

sampling_details={"sampling_method":"fixed_number","sample_size":1000,"epsilon":4}
ABC.abc_general(true_model=lm_2,sampling_details=sampling_details)
ABC.abc_general(true_model=lm_3,sampling_details=sampling_details)
ABC.abc_general(true_model=lm_4,sampling_details=sampling_details)

priors=[stats.uniform(5,10),stats.uniform(15,10),stats.uniform(-2,3)]
ABC.abc_general(true_model=glm,priors=priors,n_obs=500,var_ranges=[(0,20)],sampling_details=sampling_details)
