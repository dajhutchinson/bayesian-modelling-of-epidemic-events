from scipy import stats
import numpy as np

import ABC
from Models import LinearModel,ExponentialModel,SIRModel,GaussianMixtureModel_two

lm=LinearModel(  # 1+10x
    n_params=2,
    params=[1,10],
    n_vars=1,
    n_obs=10,
    x_obs=[[x] for x in range(10)],
    noise=30
    )
lm_priors=[stats.uniform(0,6),stats.uniform(8,6)]
lm_priors_intersect_known=[stats.uniform(1,0),stats.uniform(8,6)]

em=ExponentialModel( # 2e^{.3x}
    params=[2,.3],
    n_obs=10,
    x_obs=[[x] for x in range(10)],
    noise=1
    )
em_priors=[stats.uniform(0,3),stats.uniform(0,1)]

sir_model=SIRModel(
    params=[100000,100,1,.5],
    n_obs=30)
sir_priors=[stats.uniform(100000,0),stats.uniform(100,0),stats.uniform(0,1.5),stats.uniform(0,2)]
sir_smc_priors=[stats.uniform(100000,1),stats.uniform(100,1),stats.uniform(0,1.5),stats.uniform(0,2)]

gmm=GaussianMixtureModel_two(
    params=(-20,20,.3),
    n_obs=50,
    sd=(1,1))
gmm_priors=[stats.norm(loc=0,scale=10),stats.norm(loc=0,scale=10),stats.beta(1,1)] # from https://www.tandfonline.com/doi/pdf/10.1080/00949655.2020.1843169

print(stats.gaussian_kde([1,2,3]).ppf(.05),stats.gaussian_kde([1,2,3]).ppf(.95))

"""
    GMM
"""

# print(gmm.observe())
# gmm.plot_obs()

# ABC-Rejection Sampling
# sampling_details={"sampling_method":"best","num_runs":10000,"sample_size":100,"distance_measure":ABC.l2_norm}
# fitted_model,_=ABC.abc_rejcection(n_obs=50,y_obs=gmm.observe(),fitting_model=gmm.copy([1,1,1]),priors=gmm_priors,sampling_details=sampling_details)
# print("True Model - {}".format(gmm))
# print("Fitted Model - {}\n".format(fitted_model))

# ABC-MCMC
# perturbance_kernels = [lambda x:x+stats.norm(0,.3).rvs(1)[0]]*2+[lambda x:x+stats.norm(0,.1).rvs(1)[0]]
# fitted_model,_=ABC.abc_mcmc(n_obs=50,y_obs=gmm.observe(),fitting_model=gmm.copy([1,1,1]),priors=gmm_priors,
#     chain_length=10000,perturbance_kernels=perturbance_kernels,acceptance_kernel=ABC.gaussian_kernel,scaling_factor=70)
# print("True Model - {}".format(gmm))
# print("Fitted Model - {}\n".format(fitted_model))

# ABC-SMC
# scaling_factors=list(np.linspace(70,30,10))
#
# perturbance_kernels = [lambda x:x+stats.norm(0,.3).rvs(1)[0]]*2+[lambda x:x+stats.norm(0,.1).rvs(1)[0]]
# perturbance_kernel_probability = [lambda x,y:stats.norm(0,.3).pdf(x-y)]*2+[lambda x,y:stats.norm(0,.1).pdf(x-y)]
#
# fitted_model,_=ABC.abc_smc(n_obs=50,y_obs=gmm.observe(),fitting_model=gmm.copy([1,1,1]),priors=gmm_priors,
#     num_steps=10,sample_size=50,scaling_factors=scaling_factors,perturbance_kernels=perturbance_kernels,perturbance_kernel_probability=perturbance_kernel_probability,acceptance_kernel=ABC.gaussian_kernel)
#
# print("True Model - {}".format(lm))
# print("Fitted Model - {}\n".format(fitted_model))

"""
    SIR Model
"""
# data=sir_model.observe()
# sir_model.plot_obs(constant_scale=True)

# ABC-Rejection Sampling
# sampling_details={"sampling_method":"best","num_runs":10000,"sample_size":100,"distance_measure":ABC.log_l2_norm}
# fitted_model,_=ABC.abc_rejcection(n_obs=30,y_obs=sir_model.observe(),fitting_model=sir_model.copy([1,1,1,1]),priors=sir_priors,sampling_details=sampling_details)
# print("True Model - {}".format(sir_model))
# print("Fitted Model - {}\n".format(fitted_model))

# ABC-MCMC
# perturbance_kernels = [lambda x:x]*2 + [lambda x:x+stats.norm(0,.1).rvs(1)[0]]*2
# fitted_model,_=ABC.abc_mcmc(n_obs=30,y_obs=sir_model.observe(),fitting_model=sir_model.copy([1,1,1,1]),priors=sir_priors,
#     chain_length=10000,perturbance_kernels=perturbance_kernels,acceptance_kernel=ABC.gaussian_kernel,scaling_factor=15000)
# print("True Model - {}".format(sir_model))
# print("Fitted Model - {}\n".format(fitted_model))

# ABC-SMC
# scaling_factors=list(np.linspace(50000,10000,10))
# perturbance_variance=.1
#
# perturbance_kernels = [lambda x:x]*2 + [lambda x:x+stats.norm(0,perturbance_variance).rvs(1)[0]]*2
# perturbance_kernel_probability = [lambda x,y:1]*2 + [lambda x,y:stats.norm(0,perturbance_variance).pdf(x-y)]*2
#
# fitted_model,_=ABC.abc_smc(n_obs=30,y_obs=sir_model.observe(),fitting_model=sir_model.copy([1,1,1,1]),priors=sir_smc_priors,
#     num_steps=10,sample_size=100,scaling_factors=scaling_factors,perturbance_kernels=perturbance_kernels,perturbance_kernel_probability=perturbance_kernel_probability,acceptance_kernel=ABC.gaussian_kernel)
#
# print("True Model - {}".format(lm))
# print("Fitted Model - {}\n".format(fitted_model))

"""
    CHOOSE SUMMARY STATS
"""
# Linear Model
# mean_grad = (lambda ys:[np.mean([ys[i+1][0]-ys[i][0] for i in range(len(ys)-1)])])
# rand=(lambda ys:[stats.uniform(0,10).rvs(1)[0]])
# rand_2=(lambda ys:[mean_grad(ys)[0]*stats.uniform(0.5,1).rvs(1)[0]])
# summary_stats=[mean_grad,rand,rand_2]
# best_stats=ABC.joyce_marjoram(summary_stats,n_obs=10,y_obs=lm.observe(),fitting_model=lm.copy([1,1]),priors=lm_priors_intersect_known,n_samples=1000)
#
# print(best_stats)
"""
    REJECTION SAMPLING
"""
# Linear Model
# sampling_details={"sampling_method":"fixed_number","sample_size":100,"scaling_factor":5,"kernel_func":ABC.uniform_kernel}
#
# sampling_details={"sampling_method":"best","num_runs":1000,"sample_size":100}
# start = (lambda ys:[ys[0][0]])
# end = (lambda ys:[ys[-1][0]])
# mean_grad = (lambda ys:[np.mean([ys[i+1][0]-ys[i][0] for i in range(len(ys)-1)])])
# summary_stats=[start,end,mean_grad]
# fitted_model,_=ABC.abc_rejcection(n_obs=10,y_obs=lm.observe(),fitting_model=lm.copy([1,1]),priors=lm_priors,sampling_details=sampling_details,summary_stats=summary_stats)
# print("True Model - {}".format(lm))
# print("Fitted Model - {}\n".format(fitted_model))

# Exponential Model
# sampling_details={"sampling_method":"best","num_runs":1000,"sample_size":70}
#
# start = (lambda ys:[ys[0][0]])
# end = (lambda ys:[ys[-1][0]])
# mean_log_grad = (lambda ys:[np.mean([np.log(max(1,ys[i+1][0]-ys[i][0])) for i in range(len(ys)-1)])])
# summary_stats=[start,end,mean_log_grad]
# fitted_model,_=ABC.abc_rejcection(n_obs=10,y_obs=em.observe(),fitting_model=em.copy([1,1]),priors=em_priors,sampling_details=sampling_details,summary_stats=summary_stats)
# print("True Model - {}".format(em))
# print("Fitted Model - {}\n".format(fitted_model))

"""
    MCMC
"""
# Linear Model
# start = (lambda ys:[ys[0][0]])
# end = (lambda ys:[ys[-1][0]])
# mean_grad = (lambda ys:[np.mean([ys[i+1][0]-ys[i][0] for i in range(len(ys)-1)])])
# summary_stats=[start,end,mean_grad]
# perturbance_kernels = [lambda x:x+stats.norm(0,.1).rvs(1)[0]]*2
# fitted_model,_=ABC.abc_mcmc(n_obs=10,y_obs=lm.observe(),fitting_model=lm.copy([1,1]),priors=lm_priors,
#     chain_length=10000,perturbance_kernels=perturbance_kernels,acceptance_kernel=ABC.gaussian_kernel,scaling_factor=1,
#     summary_stats=summary_stats)
# print("True Model - {}".format(lm))
# print("Fitted Model - {}\n".format(fitted_model))

# Exponential Model
# start = (lambda ys:[ys[0][0]])
# end = (lambda ys:[ys[-1][0]])
# mean_log_grad = (lambda ys:[np.mean([np.log(max(1,ys[i+1][0]-ys[i][0])) for i in range(len(ys)-1)])])
# summary_stats=[start,end,mean_log_grad]
# perturbance_kernels = [lambda x:x+stats.norm(0,.1).rvs(1)[0]]*2
# fitted_model,_=ABC.abc_mcmc(n_obs=10,y_obs=em.observe(),fitting_model=em.copy([1,1]),priors=em_priors,
#     chain_length=10000,perturbance_kernels=perturbance_kernels,acceptance_kernel=ABC.gaussian_kernel,scaling_factor=1,
#     summary_stats=summary_stats)
# print("True Model - {}".format(em))
# print("Fitted Model - {}\n".format(fitted_model))

"""
    SMC
"""
# Linear Model
# start = (lambda ys:[ys[0][0]])
# end = (lambda ys:[ys[-1][0]])
# mean_grad = (lambda ys:[np.mean([ys[i+1][0]-ys[i][0] for i in range(len(ys)-1)])])
# summary_stats=[start,end,mean_grad]
# scaling_factors=list(np.linspace(3,.5,10))
#
# perturbance_variance=.1
#
# perturbance_kernels = [lambda x:x+stats.norm(0,perturbance_variance).rvs(1)[0]]*2
# perturbance_kernel_probability = [lambda x,y:stats.norm(0,perturbance_variance).pdf(x-y)]
#
# fitted_model,_=ABC.abc_smc(n_obs=10,y_obs=lm.observe(),fitting_model=lm.copy([1,1]),priors=lm_priors,
#     num_steps=10,sample_size=100,scaling_factors=scaling_factors,perturbance_kernels=perturbance_kernels,perturbance_kernel_probability=perturbance_kernel_probability,acceptance_kernel=ABC.gaussian_kernel,summary_stats=summary_stats)
#
# print("True Model - {}".format(lm))
# print("Fitted Model - {}\n".format(fitted_model))

# Exponential Model
# start = (lambda ys:[ys[0][0]])
# end = (lambda ys:[ys[-1][0]])
# mean_log_grad = (lambda ys:[np.mean([np.log(max(1,ys[i+1][0]-ys[i][0])) for i in range(len(ys)-1)])])
# summary_stats=[start,end,mean_log_grad]
# scaling_factors=list(np.linspace(3,.5,10))
#
# perturbance_variance=.1
# perturbance_kernels = [lambda x:x+stats.norm(0,perturbance_variance).rvs(1)[0]]*2
# perturbance_kernel_probability = [lambda x,y:stats.norm(0,perturbance_variance).pdf(x-y)]
#
# fitted_model,_=ABC.abc_smc(n_obs=10,y_obs=em.observe(),fitting_model=em.copy([1,1]),priors=em_priors,
#     num_steps=10,sample_size=100,scaling_factors=scaling_factors,perturbance_kernels=perturbance_kernels,perturbance_kernel_probability=perturbance_kernel_probability,acceptance_kernel=ABC.gaussian_kernel,summary_stats=summary_stats)
#
# print("True Model - {}".format(em))
# print("Fitted Model - {}\n".format(fitted_model))
