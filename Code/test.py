from Models import LinearModel, ExponentialModel
from ABC import abc_fixed_sample_size
from scipy import stats

lm_2=LinearModel(2,[1,2],["age"])
lm_3=LinearModel(3,[1,2,3])
lm_4=LinearModel(4,[10,20,30,40])
em=ExponentialModel([1,2],["age"])

# priors=[stats.uniform(.5,1),stats.uniform(1.5,1)]
# abc_fixed_sample_size(100,model=em,priors=priors,epsilon=.1,var_ranges=[(0,1)])

# abc_fixed_sample_size(100,epsilon=1)

abc_fixed_sample_size(100,model=lm_3,epsilon=20)
abc_fixed_sample_size(1000,model=lm_4,epsilon=10)
