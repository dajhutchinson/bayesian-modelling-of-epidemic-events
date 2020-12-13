from Models import LinearModel, ExponentialModel, GeneralLinearModel
from ABC import abc_fixed_sample_size
from scipy import stats

lm_2=LinearModel(2,[1,2],["age"])
lm_3=LinearModel(3,[1,2,3])
lm_4=LinearModel(4,[10,20,30,40])

em=ExponentialModel([1,2],["age"])

theta_star=[10,20,-1]
f=(lambda x,theta: theta[0]+theta[1]*x[0]**2+theta[2]*(x[0]**3))
glm=GeneralLinearModel(3,1,f,theta_star)

theta_star=[1,1,1]; f=(lambda x,theta: theta[0]+theta[1]*x[0]+theta[2]*(x[0]**2))
fitting_glm=GeneralLinearModel(3,1,f,theta_star)
priors=[stats.uniform(0,10),stats.uniform(0,5),stats.uniform(0,5)]
abc_fixed_sample_size(1000,true_model=em,fitting_model=fitting_glm,priors=priors,epsilon=.1,var_ranges=[(-5,5)])

# abc_fixed_sample_size(100,true_model=lm_2,epsilon=5)
# abc_fixed_sample_size(100,model=lm_3,epsilon=10)
# abc_fixed_sample_size(100,model=lm_4,epsilon=40)

# priors=[stats.uniform(5,10),stats.uniform(15,10),stats.uniform(-2,3)]
# abc_fixed_sample_size(100,true_model=glm,priors=priors,epsilon=.25,n_obs=500,var_ranges=[(0,20)])
