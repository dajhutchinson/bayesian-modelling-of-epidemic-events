from scipy import stats
import numpy as np

import ABC
from Models import LinearModel,ExponentialModel

lm=LinearModel(  # 1+10x
    n_params=2,
    params=[1,10],
    n_vars=1,
    n_obs=10,
    x_obs=[[x] for x in np.linspace(0,9,10)],
    noise=10
    )

lm.plot_obs()

em=ExponentialModel( # 2e^{.2x}
    params=[2,.4],
    n_obs=10,
    x_obs=[[x] for x in np.linspace(0,9,10)],
    noise=1
    )

# em.plot_obs()
mean_grad = lambda ys:np.mean([ys[i+1][0]-ys[i][0] for i in range(len(ys)-1)])
intercept = lambda ys:ys[0][0]
summary_stats=[intercept,mean_grad]
fitted_model=ABC.abc_general(n_obs=10,y_obs=lm.observe(),fitting_model=lm.copy([1,1]),priors=[stats.uniform(-5,10),stats.uniform(5,10)],summary_stats=summary_stats)
