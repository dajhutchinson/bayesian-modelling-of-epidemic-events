from scipy import stats
import numpy as np

import ABC
from Models import LinearModel,ExponentialModel

lm=LinearModel(  # 1+10x
    n_params=2,
    params=[1,10],
    n_vars=1,
    n_obs=10,
    x_obs=[[x] for x in range(10)],
    noise=30
    )
lm_priors=[stats.uniform(0,6),stats.uniform(8,6)]

em=ExponentialModel( # 2e^{.2x}
    params=[2,.3],
    n_obs=10,
    x_obs=[[x] for x in range(10)],
    noise=1
    )
em_priors=[stats.uniform(0,3),stats.uniform(0,1)]

# Linear Model
# sampling_details={"sampling_method":"fixed_number","sample_size":100,"scaling_factor":5,"kernel_func":ABC.uniform_kernel}
sampling_details={"sampling_method":"best","num_runs":1000,"sample_size":100}
start = (lambda ys:[ys[0][0]])
end = (lambda ys:[ys[-1][0]])
mean_grad = (lambda ys:[np.mean([ys[i+1][0]-ys[i][0] for i in range(len(ys)-1)])])
summary_stats=[start,end,mean_grad]
fitted_model=ABC.abc_general(n_obs=10,y_obs=lm.observe(),fitting_model=lm.copy([1,1]),priors=lm_priors,sampling_details=sampling_details,summary_stats=summary_stats)
print("True Model - {}".format(lm))
print("Fitted Model - {}".format(fitted_model))

# Exponential Model
# sampling_details={"sampling_method":"best","num_runs":1000,"sample_size":70}
start = (lambda ys:[ys[0][0]])
end = (lambda ys:[ys[-1][0]])
mean_log_grad = (lambda ys:[np.mean([np.log(max(1,ys[i+1][0]-ys[i][0])) for i in range(len(ys)-1)])])
summary_stats=[start,end,mean_log_grad]
fitted_model=ABC.abc_general(n_obs=10,y_obs=em.observe(),fitting_model=em.copy([1,1]),priors=em_priors,sampling_details=sampling_details,summary_stats=summary_stats)
print("True Model - {}".format(em))
print("Fitted Model - {}".format(fitted_model))
