import sys

import numpy as np
import argparse
import time

import jax
import jax.numpy as jnp
import jax.random as random

import numpyro
from numpyro.diagnostics import summary
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from contextlib import redirect_stdout

import syntheticData
import commons
import run_experiments

import jax.random as random

import target_distributions


# see 
# https://forum.pyro.ai/t/creating-a-custom-distribution-in-numpyro/3332/2

RANGE = 10.0

# checked
class MultivariateNormalMixture(numpyro.distributions.Distribution):
    support = numpyro.distributions.constraints.real

    def __init__(self, dim, validate_args=None):
        batch_shape = jax.lax.broadcast_shapes(jnp.shape(jnp.ones(dim)))
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

        cov = jnp.eye(dim)
        self.mvn1 = dist.MultivariateNormal(loc = target_distributions.MultivariateNormalMixture.getAdjustedMeanFac(dim) * jnp.ones(dim), covariance_matrix = cov)
        self.mvn2 = dist.MultivariateNormal(loc = -target_distributions.MultivariateNormalMixture.getAdjustedMeanFac(dim) * jnp.ones(dim), covariance_matrix = cov)
        self.mvn3 = dist.MultivariateNormal(loc = jnp.zeros(dim), covariance_matrix = cov)
        return
    
    # only used for initialization
    def sample(self, key, sample_shape=()):
        init_sample = jax.random.uniform(key, shape=sample_shape + self.batch_shape + self.event_shape, minval=-RANGE, maxval=RANGE)
        return init_sample

    def log_prob(self, z):
        log_probs_each_component = jax.numpy.stack((self.mvn1.log_prob(z), self.mvn2.log_prob(z), self.mvn3.log_prob(z)))
        log_prob_each_sample = jax.scipy.special.logsumexp(log_probs_each_component, 0) - jnp.log(3.0)
        return log_prob_each_sample


# checked
class Funnel(numpyro.distributions.Distribution):
    support = numpyro.distributions.constraints.real

    def __init__(self, dim, validate_args=None):
        batch_shape = jax.lax.broadcast_shapes(jnp.shape(jnp.ones(dim)))
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

        self.scale = 3.0
        self.other_dim = dim - 1
        self.vdist = dist.Normal(loc = 0.0, scale = self.scale)
        return
    
    # only used for initialization
    def sample(self, key, sample_shape=()):
        init_sample = jax.random.uniform(key, shape=sample_shape + self.batch_shape + self.event_shape, minval=-RANGE, maxval=RANGE)
        return init_sample

    def log_prob(self, z):
        
        v = z[0]
        log_prob_v_each_sample = self.vdist.log_prob(v)

        EPSILON = 10 ** -6 # to ensure strict positiviness

        std_other = jax.numpy.sqrt(jax.numpy.exp(v)) + EPSILON
       
        log_prob_other = dist.Normal(0.0, std_other).log_prob((z[1:]))
        each_sample_cond_log_prob = jax.numpy.sum(log_prob_other)

        log_prob_each_sample = log_prob_v_each_sample + each_sample_cond_log_prob
        
        return log_prob_each_sample
    

# checked
class MultivariateStudentT(numpyro.distributions.Distribution):
    support = numpyro.distributions.constraints.real

    def __init__(self, dim, validate_args=None):
        batch_shape = jax.lax.broadcast_shapes(jnp.shape(jnp.ones(dim)))
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

        self.df = 1.0
        self.dim = dim
        self.loc = 2.0 * jnp.ones(dim)
        self.cov = jnp.ones((dim,dim))*0.8 + 0.2*jnp.eye(dim)

        self.L = jnp.linalg.cholesky(jnp.linalg.inv(self.cov))

        _, logdet_cov = jnp.linalg.slogdet(self.cov)
        
        self.fixedTerms = jax.lax.lgamma((self.df + self.dim)/2.0) \
                            - jax.lax.lgamma(self.df/2.0) - (self.dim/2.0) * jnp.log(self.df) \
                            - (self.dim/2.0) * jnp.log(jnp.pi) - 0.5 * logdet_cov 
        return
    
    # only used for initialization
    def sample(self, key, sample_shape=()):
        init_sample = jax.random.uniform(key, shape=sample_shape + self.batch_shape + self.event_shape, minval=-RANGE, maxval=RANGE)
        return init_sample

    def log_prob(self, z):
        assert(z.shape == self.loc.shape)

        zL = (z - self.loc) @ self.L

        log_prob = self.fixedTerms - ((self.df + self.dim)/2.0) * jnp.log(1.0 + (1.0/self.df)*jnp.sum(jnp.square(zL)))
        
        return log_prob
    
       

# horeseshoe regression model with binary-valued responses
def horsehoe_model_bernoulli_likelihood(X, y = None):
    d = X.shape[1]

    # sample from horseshoe prior
    lambdas = numpyro.sample("lambdas", dist.HalfCauchy(scale = jnp.ones(d)))
    tau = numpyro.sample("tau", dist.HalfCauchy(scale = jnp.ones(1)))

    # note that this reparameterization (i.e. coordinate transformation) improves
    # posterior geometry and makes NUTS sampling more efficient
    unscaled_betas = numpyro.sample("unscaled_betas", dist.Normal(0.0, jnp.ones(d)))
    scaled_betas = numpyro.deterministic("betas", tau * lambdas * unscaled_betas)

    # add intercept
    intercept = numpyro.sample("intercept", dist.Cauchy(loc = jnp.zeros(1), scale = 10.0 * jnp.ones(1)))

    # compute mean function using linear coefficients
    mean_function = jnp.dot(X, scaled_betas) + intercept
    
    # observe data
    numpyro.sample("y", dist.Bernoulli(logits=mean_function), obs=y)


# conjugate bayesian linear regression model
def conjugate_bayesian_linear_regression_model(X, y = None):
    d = X.shape[1]

    sigmaSquared = numpyro.sample("sigmaSquared", dist.InverseGamma(0.5, 0.5))
    sigma = numpyro.deterministic("sigma", jnp.sqrt(sigmaSquared))

    betas = numpyro.sample("betas", dist.Normal(0.0, jnp.ones(d) * sigma))

    # compute mean function using linear coefficients
    mean_function = jnp.dot(X, betas)
    
    # observe data
    numpyro.sample("y", dist.Normal(mean_function, sigma), obs=y)






# helper function for HMC inference
def run_inference(model, args, rng_key, X, y):
    
    kernel = NUTS(model)
    mcmc = MCMC(
        kernel,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        thinning = args.thinning,
        progress_bar=True
    )

    mcmc.run(rng_key, X, y)
    mcmc.print_summary(exclude_deterministic=False)

    samples = mcmc.get_samples()
    
    return samples, mcmc


def show_sampling_summary(summary_dict, name):
    print(f"********** {name} samples ********** ")
    print("highest r_hat = ", np.max(summary_dict[name]["r_hat"]))
    print("average n_eff = ", np.mean(summary_dict[name]["n_eff"]))
    return 


def main(args):
    
    TOTAL_NR_MCMC_SAMPLES = ((args.num_samples * args.num_chains) // args.thinning) 
    print("TOTAL_NR_MCMC_SAMPLES = ", TOTAL_NR_MCMC_SAMPLES)
    print("target = ", args.target)
    print("D = ", args.D)
    

    if args.target == "HorseshoePriorLogisticRegression" or args.target == "ConjugateLinearRegression":

        X, y, true_beta, true_intercept = run_experiments.load_data(args)
        n = X.shape[0]
        d = X.shape[1]

    else:
        X = None
        y = None
    

    rng_key, rng_key_predict = random.split(random.PRNGKey(0))

    if args.target == "HorseshoePriorLogisticRegression":
        model = horsehoe_model_bernoulli_likelihood
    elif args.target == "ConjugateLinearRegression":
        model = conjugate_bayesian_linear_regression_model
        assert(true_intercept == 0.0)
    else:
        
        UnnormalizedDist = globals()[args.target]

        def self_defined_log_prob_model(X = None, y = None):
            numpyro.sample("theta", UnnormalizedDist(args.D))

        model = self_defined_log_prob_model
    
    startTime_sampling = time.time()
    
    posterior_samples, mcmc = run_inference(model, args, rng_key, X, y)

    sampling_time = (time.time() - startTime_sampling) / 60.0
    
    print("**** RESULTS MCMC *************")

    if args.target == "HorseshoePriorLogisticRegression" or args.target == "ConjugateLinearRegression":
        
        beta_samples = posterior_samples["betas"]

        if args.target == "HorseshoePriorLogisticRegression":
            intercept_samples = posterior_samples["intercept"]
        elif args.target == "ConjugateLinearRegression":
            intercept_samples = None
        else:
            assert(False)
            
        assert(beta_samples.shape[0] == (args.num_samples * args.num_chains) // args.thinning) # Monte Carlo Samples
        assert(beta_samples.shape[1] == d)

        syntheticData.show_regression_result_statistics(beta_samples, intercept_samples, true_beta, true_intercept)
        
        # print("total number of samples = ", beta_samples.shape[0])
        # show_sampling_summary(summary_dict, "betas")
        # show_sampling_summary(summary_dict, "sigmaSquared")
        
        print("n = ", n)
        print("d = ", d)
        print("true_beta = ", true_beta)
        print("true_intercept = ", true_intercept)

    else:

        theta_samples = posterior_samples["theta"]
        print("theta_samples.shape = ", theta_samples.shape)

        # import seaborn as sns
        # import matplotlib.pyplot as plt
        # data = theta_samples[:, 0]
        # sns.set_style('whitegrid')
        # sns.kdeplot(np.array(data), bw=0.5)
        # plt.show()
        # assert(False)


    DATA_DESCRIPTOR_STR = run_experiments.getDataDescriptorStr(args)
    infoStrMCMC = commons.getMCMC_spec_str(args.target, DATA_DESCRIPTOR_STR, args.num_samples, args.num_warmup, args.num_chains, args.thinning, device = "cpu")

    summary_dict = summary(posterior_samples, group_by_chain=False)
    np.save("all_results/" + infoStrMCMC + "_summary_dict", summary_dict)
 
    with open("all_results/" + infoStrMCMC + "_summary.txt", "wt") as f:
        with redirect_stdout(f):
            mcmc.print_summary(exclude_deterministic=False)
    
    np.save("all_results/" + infoStrMCMC + "_" + "posterior_samples", posterior_samples)
    print("*** Successfully saved all posterior samples ***")


    recorded_results = {}
    recorded_results["sampling_time_in_minutes"] = sampling_time
    np.save("all_results/" + infoStrMCMC + "_" + "timing_results", recorded_results)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Horseshoe regression")
    
    
    # *************** data parameters *************

    parser.add_argument("--n", default=None, type=int)  
    parser.add_argument("--d", default=None, type=int)
    parser.add_argument("--intercept", default=None, type=float)
    parser.add_argument("--rho", default=None, type=float)

    parser.add_argument("--data", default="no_data", type=str)
    parser.add_argument("--D", default=None, type=int)

    parser.add_argument("--target", default=None, type=str) 

    parser.add_argument("--foldId", default=None, type=int)

    # *************** MCMC parameters *************
    
    parser.add_argument("--num-samples", default=commons.NUM_SAMPLES_MCMC, type=int)  
    parser.add_argument("--num-warmup", default=commons.NUM_WARMUP_MCMC, type=int) 
    parser.add_argument("--num-chains", default=commons.NUM_CHAINS_MCMC, type=int)
    parser.add_argument("--thinning", default=commons.THINNING_MCMC, type=int)
    
    parser.add_argument("--device", default="cpu", type=str, help='use "cpu" or "gpu".')
    args = parser.parse_args()

    assert(args.num_samples >= 5 and args.num_warmup >= 2)
    assert(args.device == "cpu" or args.device == "gpu")

    commons.get_most_freemem_gpu()
    
    numpyro.enable_x64()
    numpyro.set_platform(args.device)
    
    main(args)