from Models import LinearModel, ExponentialModel
from ABC import abc_fixed_sample_size
from scipy import stats

lm=LinearModel(2,[1,2],["age"])
em=ExponentialModel([1,2],["age"])

# priors=[stats.uniform(.5,1),stats.uniform(1.5,1)]
# abc_fixed_sample_size(100,model=em,priors=priors,epsilon=.1,var_ranges=[(0,1)])

abc_fixed_sample_size(100,epsilon=1)
