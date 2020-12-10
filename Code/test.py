from Models import LinearModel, ExponentialModel

lm=LinearModel(3,[1,2,3],["age","weight"])
em=ExponentialModel([2,3],["age"])

em.plot()
lm.plot()
