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

# priors=[stats.uniform(.5,1),stats.uniform(1.5,1)]
# abc_fixed_sample_size(100,model=em,priors=priors,epsilon=.1,var_ranges=[(0,1)])

# abc_fixed_sample_size(100,model=lm_2,epsilon=5)
# abc_fixed_sample_size(100,model=lm_3,epsilon=10)
# abc_fixed_sample_size(100,model=lm_4,epsilon=40)

priors=[stats.uniform(5,10),stats.uniform(15,10),stats.uniform(-2,3)]
abc_fixed_sample_size(1000,model=glm,priors=priors,epsilon=.25,n_obs=500,var_ranges=[(0,40)])
