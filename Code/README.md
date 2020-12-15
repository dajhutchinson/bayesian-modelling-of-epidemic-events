# `Models.py`
Provides class for common models:
 - `LinearModel` - <img src="https://latex.codecogs.com/gif.latex?y=\beta_0+\sum\beta_ix_i" />
 - `ExponentialModel` - <img src="https://latex.codecogs.com/gif.latex?y=\beta_0\exp(\beta_1x)" />

Methods
 - `__init__()` - Specify the parameters of the model
 - `.calc()` - Calculate the response variable value given predictor variable values
 - `.plot()` - Plot the response variable of a model for given ranges of the predictor variable (Only for models with 1 or 2 predictor variables).
 - `__str__()` - String for the equation of the response variable.

# `ABC.py`
Implementations of Approximate Bayesian Computation

Functions
- `abc_general()` - Perform ABC. See **sampling_details** for details on specifying what sampling method to use.

## `sampling_details`
There are a few options for different sampling methods to use. The details are defined in a `dict` with the following keys
 - `sampling_method` = "best_samples". Use the best `sample_size` samples from a larger set of `num_runs`. Need to define the follow
   - `sample_size` = Number of samples to keep.
   - `num_runs` = Total of samples to make.
 - `sampling_method` = "fixed_number". Sample until a sufficient number of samples are found which are close to observations
   - `sample_size` = Desired number of samples.
   - `epsilon` = How close a sample needs to be to an observation to be accepted. This is used as an argument to the `uniform_kernel` function.
