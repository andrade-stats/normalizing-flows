
import commons
import torch
import run_experiments
import evaluation

import normalizing_flows_core as core
import estimators
import show_plot_analysis_loss
import normflows as nf

import analysis
import numpy
import pyro
import scipy
import show_table

# filename = "all_results/HorseshoePriorLogisticRegression_synthetic_logistic_100n_1000d_1.0intercept_0.1rho_0foldId_RealNVP_small_64nrFLows_30000_256_reverse_kld_without_score_LOFTCUSHION_TYPE_FalseBATCH_NORM_noANNEALING_zerosINITIALIZATION_100.0_64"
# filename = "all_results/ConjugateLinearRegression_synthetic_logistic_100n_1000d_1.0intercept_0.1rho_0foldId_RealNVP_small_64nrFLows_30000_256_reverse_kld_without_score_LOFTCUSHION_TYPE_FalseBATCH_NORM_zerosINITIALIZATION_100.0_64"
# filename = "all_results/HPLR_synthetic_logistic_100n_1000d_1.0intercept_0.1rho_0foldId_RealNVP_small_64nrFLows_30000_256_reverse_kld_without_score_LOFTCUSHION_TYPE_FalseBATCH_NORM_noANNEALING_zerosINITIALIZATION_100.0_64"

# print("len(filename) = ", len(filename))
# print("len(_nr_optimization_steps_stats.npy) = ", len("_nr_optimization_steps_stats.npy"))
# assert(False)

# alpha = 2.0
# beta = 5.0
# ig = pyro.distributions.inverse_gamma.InverseGamma(alpha, beta)
# print("log_prob = ", ig.log_prob(torch.tensor([4.5])))

# invGamma = scipy.stats.invgamma(a = alpha, scale = beta)

# testVal = invGamma.logpdf(4.5)
# print("testVal = ", testVal)



# *************** data parameters *************

setting = {}

setting["n"] = 100
setting["d"] = 1000
setting["rho"] = 0.1
setting["intercept"] = 1.0
setting["data"] = "synthetic_logistic"
setting["target"] = "HorseshoePriorLogisticRegression"
setting["foldId"] = 1

# setting["n"] = 10
# setting["d"] = 1000
# setting["rho"] = 0.5
# setting["intercept"] = 0.0
# setting["data"] = "synthetic_regression"
# setting["target"] = "ConjugateLinearRegression"

# *************** Model/Target *************
# setting["target"] = "MultivariateNormalMixture"
# setting["targetK"] = 3
# setting["targetMeanFac"] = 2.0

# setting["target"] = "Funnel"
# setting["target"] = "MultivariateStudentT"
# setting["D"] = 1000


# *************** NFM parameters *************
# setting["cushion_type"] = "none"
setting["cushion_type"] = "LOFT"

NR_FLOWS = 64
setting["nr_flows"] = NR_FLOWS

# setting["flow_type"] = "GaussianOnly"
setting["flow_type"] = "RealNVP_small"
# setting["flow_type"] = "RealNVP"
setting["nr_cushions"] = 1 # NR_FLOWS
setting["loft_t"] = 100.0

setting["nr_mixture_components"] = 1
setting["init"] = "zeros"
setting["lr_exp"] = 4

setting["divergence"] = "reverse_kld_ws_debug"
# setting["divergence"] = "reverse_kld_without_score"
# setting["divergence"] = "reverse_kld"

setting["l2_strength"] = 0.0
setting["annealing"] = "no"

setting["no_act_norm"] = "yes"

setting["data_type"] = "double"

# if setting["divergence"] == "reverse_kld":
#     setting["max_iterations"] = 60000
# else:
#     setting["max_iterations"] = 30000

setting["max_iterations"] = 100000

# setting["realNVP_variation"] = "var1"
setting["realNVP_threshold"] = 0.1
setting["realNVP_spec"] = "MaskedAffineFlowSoftClamp" 

setting["scaleShiftLayer"] = "ssL"

args = run_experiments.getArgumentParser(**setting)

commons.DATA_TYPE = args.data_type
commons.setGPU()
torch.manual_seed(432432)

target, flows_mixture = run_experiments.initialize_target_and_flow(args, initialize = True)
# flows_mixture.load_state_dict(torch.load(commons.get_model_filename_best(), map_location = commons.DEVICE))

RECORD_TYPES = ["loc", "log_scale", "weight", "bias"]
all_time_param_stats = {}
for rt in RECORD_TYPES:
    all_time_param_stats[rt] = numpy.zeros(1)
analysis.recordMaxAbsValues(all_time_param_stats, 0, flows_mixture)

print("all_time_param_stats = ")
print(all_time_param_stats)
assert(False)

flows_mixture.eval()
with torch.no_grad():
    all_ELBO_bestModel = estimators.getRepeatedEstimates(flows_mixture, "ELBO")
    # all_IS_bestModel = estimators.getRepeatedEstimates(flows_mixture, "IS")

print("******************* ")
print(f"ELBO (best model) = {evaluation.showAvgAndStd_str(all_ELBO_bestModel)}")
# print(f"ELBO (best model) = {evaluation.showAvgAndStd_str(all_ELBO_bestModel)}, IS (best model) = {evaluation.showAvgAndStd_str(all_IS_bestModel)}")
print("******************* ")

# flows_mixture.all_flows[0].log_scale

log_scale = flows_mixture.all_flows[0].q0.log_scale
loc = flows_mixture.all_flows[0].q0.loc
print("log_scale = ", torch.max(log_scale))
print("loc = ", torch.max(torch.abs(loc)))
assert(False)

count = 0
for flow in flows_mixture.all_flows[0].flows:
    if type(flow) is nf.flows.MaskedAffineFlow:
        count += 1
        print("GOT IT")

print("count = ", count)
assert(False)

max_value_s = 0.0
max_value_t = 0.0

for name, param_group in flows_mixture.named_parameters():
    if name.endswith(".t"):
        for param in param_group:
            max_value_t = max(max_value_t, torch.max(torch.abs(param)))
    elif name.endswith(".s"):
        for param in param_group:
            max_value_s = max(max_value_s, torch.max(torch.abs(param)))
        
print("*************")
print("max_value_s = ", max_value_s.item())

print("*************")
print("max_value_t = ", max_value_t.item())

