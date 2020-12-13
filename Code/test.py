from Models import LinearModel, ExponentialModel
from ABC import abc_fixed_sample_size

lm=LinearModel(3,[1,2,3],["age","weight"])
em=ExponentialModel([2,3],["age"])

abc_fixed_sample_size(1000,epsilon=1)
