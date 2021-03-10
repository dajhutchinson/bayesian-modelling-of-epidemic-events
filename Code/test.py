from scipy import stats
import numpy as np

import ABC
import ABC_Cross_Validation
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
    n_obs=30,
    x_obs=[[x] for x in range(30)],)
sir_priors=[stats.uniform(100000,0),stats.uniform(100,0),stats.uniform(0,1.5),stats.uniform(0,2)]
sir_smc_priors=[stats.uniform(100000,1),stats.uniform(100,1),stats.uniform(0,1.5),stats.uniform(0,2)]

gmm=GaussianMixtureModel_two(
    params=(-20,20,.3),
    n_obs=50,
    sd=(1,1))
gmm_priors=[stats.norm(loc=0,scale=10),stats.norm(loc=0,scale=10),stats.beta(1,1)] # from https://www.tandfonline.com/doi/pdf/10.1080/00949655.2020.1843169

"""
    CROSS-VALIDATION
"""
# Exponential Model
# start = (lambda ys:[ys[0][0]])
# mean_log_grad = (lambda ys:[10*np.mean([np.log(max(1,ys[i+1][0]-ys[i][0])) for i in range(len(ys)-1)])])
# summary_stats=[start,mean_log_grad]
#
# scaling_factors=list(np.linspace(7,4,10))
#
# perturbance_variance=.1
#
# perturbance_kernels = [lambda x:x+stats.norm(0,perturbance_variance).rvs(1)[0]]*2
# perturbance_kernel_probability = [lambda x,y:stats.norm(0,perturbance_variance).pdf(x-y)]
#
# error=ABC_Cross_Validation.LOO_CV_abc_smc(n_obs=10,x_obs=em.x_obs,y_obs=em.observe(),fitting_model=em.copy([1,1]),priors=em_priors,
#         perturbance_kernels=perturbance_kernels,perturbance_kernel_probability=perturbance_kernel_probability,acceptance_kernel=ABC.gaussian_kernel,scaling_factors=scaling_factors,
#         num_steps=10,sample_size=50,summary_stats=summary_stats)
# print("Total error : {:,.3f}".format(error))

# Linear Model
# sampling_details={"sampling_method":"best","num_runs":10000,"sample_size":1000}
# start = (lambda ys:[ys[0][0]])
# mean_grad = (lambda ys:[np.mean([ys[i+1][0]-ys[i][0] for i in range(len(ys)-1)])])
# summary_stats=[start,mean_grad]
# error=ABC_Cross_Validation.LOO_CV_abc_rejection(n_obs=10,x_obs=lm.x_obs,y_obs=lm.observe(),fitting_model=lm.copy([1,1]),priors=lm_priors,sampling_details=sampling_details,summary_stats=summary_stats)
# print("Total error : {:,.3f}".format(error))

# SIR Model
# peak_infections_date_ss=(lambda ys:[1000*ys.index(max(ys,key=lambda y:y[1]))])
# peak_infections_value_ss=(lambda ys:[max(ys,key=lambda y:y[1])[1]])
# summary_stats=[peak_infections_date_ss,peak_infections_value_ss]
#
# # SIR-ABC_rejection
# sampling_details={"sampling_method":"best","num_runs":10000,"sample_size":1000,"distance_measure":ABC.log_l2_norm}
# error=ABC_Cross_Validation.LOO_CV_abc_rejection(n_obs=30,x_obs=sir_model.x_obs,y_obs=sir_model.observe(),fitting_model=sir_model.copy([1,1,1,1]),priors=sir_priors,sampling_details=sampling_details,summary_stats=summary_stats)
# print("Total error : {:,.3f}".format(error))

# SIR-ABC_MCMC
# perturbance_kernels = [lambda x:x]*2 + [lambda x:x+stats.norm(0,.1).rvs(1)[0]]*2
#
# error=ABC_Cross_Validation.LOO_CV_abc_mcmc(n_obs=30,x_obs=sir_model.x_obs,y_obs=sir_model.observe(),
#     fitting_model=sir_model.copy([1,1,1,1]),priors=sir_priors,perturbance_kernels=perturbance_kernels,
#     acceptance_kernel=ABC.gaussian_kernel,scaling_factor=5000,chain_length=10000,summary_stats=summary_stats)
# print("Total error : {:,.3f}".format(error))

# SIR-ABC_SMC
# perturbance_variance=.1
# perturbance_kernels = [lambda x:x]*2 + [lambda x:x+stats.norm(0,perturbance_variance).rvs(1)[0]]*2
# perturbance_kernel_probability = [lambda x,y:1]*2 + [lambda x,y:stats.norm(0,perturbance_variance).pdf(x-y)]*2
#
# scaling_factors=list(np.linspace(50000,25000,6))
#
# error=ABC_Cross_Validation.LOO_CV_abc_smc(n_obs=30,x_obs=sir_model.x_obs,y_obs=sir_model.observe(),fitting_model=sir_model.copy([1,1,1,1]),priors=sir_smc_priors,
#         perturbance_kernels=perturbance_kernels,perturbance_kernel_probability=perturbance_kernel_probability,acceptance_kernel=ABC.gaussian_kernel,scaling_factors=scaling_factors,
#         num_steps=6,sample_size=50,summary_stats=summary_stats)
# print("Total error : {:,.3f}".format(error))
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

# Linear Model with known start point
# mean_grad = (lambda ys:[np.mean([ys[i+1][0]-ys[i][0] for i in range(len(ys)-1)])])
# rand=(lambda ys:[stats.uniform(0,6).rvs(1)[0]]) # variance set st similar number of samples accepted as mean_grad
# rand_grad = (lambda ys:[mean_grad(ys)[0]*stats.uniform(0,2).rvs(1)[0]])
# summary_stats=[mean_grad,rand,rand_grad]
#
# param_bounds=[(1,1),(8,14)]
# best_stats=ABC.joyce_marjoram(summary_stats,n_obs=10,y_obs=lm.observe(),fitting_model=lm.copy([1,1]),priors=lm_priors_intersect_known,param_bounds=param_bounds,n_samples=10000,n_bins=10,printing=True)
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
