import commons
import torch
import numpy as np
import run_experiments
import evaluation
import argparse

def insert_result(all_collected_results, results_one_fold, foldId):

    for key in results_one_fold:
        if key not in all_collected_results:
            assert(foldId == 0)
            all_collected_results[key] = np.zeros(commons.REPETITIONS_FOR_MC_ERROR) * np.nan
        
        all_collected_results[key][foldId] = results_one_fold[key]

    return

def show_all_results(all_collected_results):
    for key in all_collected_results:
        print(str(key) + " = " + evaluation.showAvgAndStd_str(all_collected_results[key]))
    print("***********************************")

# python compare_samples.py --target=HorseshoePriorLogisticRegression --d=100 --foldId=-1 --flow-type=RealNVP_small --method=no_loft_proposed

# python compare_samples.py --target=HorseshoePriorLogisticRegression --d=10 --foldId=-1 --flow-type=RealNVP_small --method=proposed_withStudentT
    
# python gather_all_results.py --target=HorseshoePriorLogisticRegression --d=1000 --foldId=-1 --flow-type=GaussianOnly
# python gather_all_results.py --target=HorseshoePriorLogisticRegression --d=1000 --foldId=-1 --flow-type=RealNVP_small --method=proposed_withStudentT
# python gather_all_results.py --target=HorseshoePriorLogisticRegression --d=1000 --foldId=-1 --flow-type=RealNVP_small --method=ATAF
# python gather_all_results.py --target=HorseshoePriorLogisticRegression --d=1000 --foldId=-1 --flow-type=RealNVP_small --method=standard
# python gather_all_results.py --target=HorseshoePriorLogisticRegression --d=1000 --foldId=-1 --flow-type=RealNVP_small --method=SymClip
# python gather_all_results.py --target=HorseshoePriorLogisticRegression --d=1000 --foldId=-1 --flow-type=RealNVP_small --method=proposed

# python gather_all_results.py --target=HorseshoePriorLogisticRegression --d=1000 --foldId=-1 --flow-type=RealNVP_small --method=no_loft_proposed
# python gather_all_results.py --target=HorseshoePriorLogisticRegression --d=1000 --foldId=-1 --flow-type=RealNVP_small --method=no_loft_proposed_withStudentT

# 
# no_loft_proposed_withStudentT
# python compare_samples.py --target=HorseshoePriorLogisticRegression --d=1000 --foldId=-1 --flow-type=RealNVP_small --method=SymClip
# python gather_all_results.py --target=HorseshoePriorLogisticRegression --d=1000 --foldId=-1 --flow-type=RealNVP_small --method=proposed

parser = argparse.ArgumentParser(description="Show Table")

parser.add_argument("--foldId", default=1, type=int)
parser.add_argument("--D", default=-1, type=int)
parser.add_argument("--d", default=-1, type=int)
parser.add_argument("--target", default=None, type=str)
parser.add_argument("--method", default=None, type=str)
parser.add_argument("--flow-type", default=None, type=str)
parser.add_argument("--nr-flows", default=64, type=int)
real_args = parser.parse_args()

_, __, args = run_experiments.simple_init(real_args.target, max(real_args.D, real_args.d), real_args.flow_type, real_args.method, real_args.nr_flows, real_args.foldId, initialize = False)

if args.target == "HorseshoePriorLogisticRegression":
    if args.foldId == -1:
        all_foldIds = np.arange(commons.REPETITIONS_FOR_MC_ERROR)
    else:
        assert(False)
        all_foldIds = [args.foldId]
else:
    all_foldIds = [1]


stat_name = ""

collected_all_results_flow_uncorrected = {}
collected_all_results_flow_adjusted = {}
collected_stats_flow_to_mcmc = {}
collected_all_results_MCMC = {}

all_ELBO = np.zeros(commons.REPETITIONS_FOR_MC_ERROR)
all_IS = np.zeros(commons.REPETITIONS_FOR_MC_ERROR)
all_ELBO_bestModel = np.zeros(commons.REPETITIONS_FOR_MC_ERROR)
all_IS_bestModel = np.zeros(commons.REPETITIONS_FOR_MC_ERROR)

all_WDs = None

for foldId in all_foldIds:
    args.foldId = foldId

    commons.DATA_TYPE = args.data_type
    commons.setGPU()
    torch.manual_seed(432432)

    print("args.target = ", args.target)
    print(f"************** {args.analyzing} ***************")
    target, flows_mixture = run_experiments.initialize_target_and_flow(args, initialize = False)

    if args.target == "HorseshoePriorLogisticRegression":
        uncorrected_beta_stats = commons.loadStatistics("uncorrected_beta_stats")
        insert_result(collected_all_results_flow_uncorrected, uncorrected_beta_stats, foldId)

        adjusted_beta_stats = commons.loadStatistics("adjusted_beta_stats")
        insert_result(collected_all_results_flow_adjusted, adjusted_beta_stats, foldId)

        stats_flow_to_mcmc = commons.loadStatistics("WD_to_MCMC")
        insert_result(collected_stats_flow_to_mcmc, stats_flow_to_mcmc, foldId)

        infoStrMCMC = commons.getMCMC_spec_str(args.target, run_experiments.getDataDescriptorStr(args), commons.NUM_SAMPLES_MCMC, commons.NUM_WARMUP_MCMC, commons.NUM_CHAINS_MCMC, commons.THINNING_MCMC, device = "cpu")
        all_results_MCMC = commons.loadStatistics("MCMC_beta_stats", infoStrMCMC)
        insert_result(collected_all_results_MCMC, all_results_MCMC, foldId)
    else:
        all_WDs = commons.loadArray("WD_to_Truth")

    # get ELBO / IS results
    result = commons.loadStatistics("log_marginal_eval_result")
    all_ELBO[foldId] = np.nanmean(result["ELBO"])
    all_IS[foldId] = np.nanmean(result["IS"])
    
    result_best = commons.loadStatistics("log_marginal_eval_result_best")
    all_ELBO_bestModel[foldId] = np.nanmean(result_best["ELBO"])
    all_IS_bestModel[foldId] = np.nanmean(result_best["IS"])

print("***********************************")

if args.target == "HorseshoePriorLogisticRegression":
    print("collected_all_results_flow_uncorrected")
    show_all_results(collected_all_results_flow_uncorrected)

    print("collected_all_results_flow_adjusted")
    show_all_results(collected_all_results_flow_adjusted)

    print("collected_stats_flow_to_mcmc")
    show_all_results(collected_stats_flow_to_mcmc)

    print("collected_all_results_MCMC")
    show_all_results(collected_all_results_MCMC)
else:
    print(f"WD to truth = {evaluation.showAvgAndStd_str(all_WDs)}")

print("******************* ")
print(f"ELBO (best model) = {evaluation.showAvgAndStd_str(all_ELBO_bestModel)}, IS (best model) = {evaluation.showAvgAndStd_str(all_IS_bestModel)}")
print("******************* ")
print(f"ELBO (last model)= {evaluation.showAvgAndStd_str(all_ELBO)}, IS (last model) = {evaluation.showAvgAndStd_str(all_IS)}")
print("******************* ")
print("************************************")
