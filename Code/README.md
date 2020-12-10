# Models.py
Provides class for common models:
 - `LinearModel` - <img src="https://latex.codecogs.com/gif.latex?y=beta_0+\sum\beta_ix_i" />
 - `ExponentialModel` - <img src="https://latex.codecogs.com/gif.latex?y=beta_0\exp(\beta_1x)" />

Methods
 - `__init__` - Specify the parameters of the model
 - `.calc()` - Calculate the response variable value given predictor variable values
 - `.plot` - Plot the response variable of a model for given ranges of the predictor variable (Only for models with 1 or 2 predictor variables).
 - `__str__` - String for the equation of the response variable.
