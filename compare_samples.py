import commons
import time
import torch
import numpy as np
import ot
import run_experiments
import syntheticData
import core_adjusted
import scipy.special
import argparse

def getExpectedBetaStatistics(beta_samples_approx, intercept_samples_approx, target):
    
    print("******************")
    
    beta_mean_approx, intercept_mean_approx, iqr_approx, avg_mse_to_truth_all_approx, avg_mse_to_truth_beta_approx = syntheticData.show_regression_result_statistics(beta_samples_approx, intercept_samples_approx, target.true_beta.cpu().numpy(), target.true_intercept.cpu().numpy())
    r_precision = syntheticData.eval_variable_selection(beta_mean_approx, target.true_beta.cpu().numpy())
    
    all_results = {}
    all_results["iqr_approx"] = iqr_approx
    all_results["avg_mse_to_truth_all_approx"] = avg_mse_to_truth_all_approx
    all_results["avg_mse_to_truth_beta_approx"] = avg_mse_to_truth_beta_approx
    all_results["r_precision"] = r_precision
    
    print(all_results)
    
    return all_results, beta_mean_approx



TOTAL_NR_SAMPLES_FOR_EVALUATION = ((commons.NUM_SAMPLES_MCMC * commons.NUM_CHAINS_MCMC) // commons.THINNING_MCMC) 

# helper to easily use most common parameter settings
# e.g.:
# python compare_samples.py --target=Funnel --D=100 --flow-type=GaussianOnly
# python compare_samples.py --target=Funnel --D=100 --flow-type=RealNVP_small --method=standard
# python compare_samples.py --target=HorseshoePriorLogisticRegression --d=100 --flow-type=RealNVP_small --method=standard
# python compare_samples.py --target=HorseshoePriorLogisticRegression --d=100 --flow-type=RealNVP_small --method=proposed
# python compare_samples.py --target=Funnel --D=1000 --flow-type=HMC
# python compare_samples.py --target=MultivariateNormalMixture --D=1000 --flow-type=HMC
# python compare_samples.py --target=MultivariateStudentT --D=1000 --flow-type=HMC
# python compare_samples.py --target=ConjugateLinearRegression --D=1000 --flow-type=HMC
# python compare_samples.py --target=HorseshoePriorLogisticRegression --d=1000 --flow-type=HMC

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
        assert(args.foldId >= 0 and args.foldId <= 19)
        assert(False)
        all_foldIds = [args.foldId]
else:
    all_foldIds = [1]


for foldId in all_foldIds:
    args.foldId = foldId

    commons.DATA_TYPE = args.data_type
    commons.setGPU()
    torch.manual_seed(432432)
    startTime = time.time()

    samples_approx = None

    print("args.target = ", args.target)
    print(f"************** {args.flow_type} ***************")

    infoStr_for_WD = None

    if args.flow_type != "HMC":

        target, flows_mixture = run_experiments.initialize_target_and_flow(args, initialize = True)

        flows_mixture.load_state_dict(torch.load(commons.get_model_filename_best(), map_location = commons.DEVICE))

        if args.target == "ConjugateLinearRegression":
            sigmaSquared_samples_approx,  samples_approx = target.get_samples_from_approx(flows_mixture, num_samples = TOTAL_NR_SAMPLES_FOR_EVALUATION)
        elif args.target == "HorseshoePriorLogisticRegression":
            beta_samples_approx, intercept_samples_approx, samples_log_q, samples_log_p = target.get_samples_from_approx(flows_mixture, num_samples = TOTAL_NR_SAMPLES_FOR_EVALUATION)

            assert(samples_log_q.shape[0] >= 100) # can be smaller than TOTAL_NR_SAMPLES_FOR_EVALUATION if illegal values are encountered
            assert(samples_log_q.shape[0] == samples_log_p.shape[0])
            assert(samples_log_q.shape[0] == beta_samples_approx.shape[0])
            actual_nr_samples = samples_log_q.shape[0]

            # log(p(D)) estimate
            log_mC = scipy.special.logsumexp(samples_log_p - samples_log_q, axis = 0)
            log_is_estimate = log_mC - np.log([actual_nr_samples])
            assert(samples_log_q.shape[0] == actual_nr_samples)
            print("log_is_estimate = ", log_is_estimate)

            # get asymptotically unbiased estimate of beta
            log_posterior_est = samples_log_p - log_is_estimate
            scale_corrections = (np.exp(log_posterior_est - samples_log_q)).reshape(-1, 1)
            beta_samples_approx_adjusted_for_expectation = beta_samples_approx * scale_corrections
            
            # print("scale_corrections[0:10] = ", scale_corrections[0:10])
            
            # marginal_likelihood_est = np.exp(log_is_estimate)
            # print("log_is_estimate = ", log_is_estimate)
            # print("marginal_likelihood_est = ", marginal_likelihood_est)

        else:
            with torch.no_grad():
                z,log_q = flows_mixture.sample(num_samples = TOTAL_NR_SAMPLES_FOR_EVALUATION)
                z,log_q,_, _ = core_adjusted.filter_illegal_values_from_samples(z, log_q)
                samples_approx = z.cpu().numpy()

                # print("samples_approx = ", samples_approx)
                # assert(False)

        infoStr_for_WD = "all_results/" + commons.INFO_STR
        
    elif args.flow_type == "HMC":

        target, flows_mixture = run_experiments.initialize_target_and_flow(args, initialize = False)

        infoStrMCMC = commons.getMCMC_spec_str(args.target, run_experiments.getDataDescriptorStr(args), commons.NUM_SAMPLES_MCMC, commons.NUM_WARMUP_MCMC, commons.NUM_CHAINS_MCMC, commons.THINNING_MCMC, device = "cpu")
        posterior_samples_MCMC = np.load("all_results/" + infoStrMCMC + "_" + "posterior_samples" + ".npy", allow_pickle = True).item()

        if args.target == "ConjugateLinearRegression":
            samples_approx = np.asarray(posterior_samples_MCMC["betas"])
            sigmaSquared_samples_approx = np.asarray(posterior_samples_MCMC["sigmaSquared"])
            assert(TOTAL_NR_SAMPLES_FOR_EVALUATION == samples_approx.shape[0])
        elif args.target == "HorseshoePriorLogisticRegression":
            beta_samples_approx = np.asarray(posterior_samples_MCMC["betas"])
            intercept_samples_approx = np.asarray(posterior_samples_MCMC["intercept"])
            assert(TOTAL_NR_SAMPLES_FOR_EVALUATION == beta_samples_approx.shape[0])
        else:
            samples_approx = np.asarray(posterior_samples_MCMC["theta"])

        infoStr_for_WD = "all_results/" + infoStrMCMC


        summary_dict = np.load("all_results/" + infoStrMCMC + "_summary_dict.npy", allow_pickle = True).item()

        print("summary_dict = ")
        print(summary_dict.keys())
        for key in summary_dict.keys():
            print(key)
            print("max r_hat = ", np.max(summary_dict[key]["r_hat"]))
            print("min n_eff = ", np.min(summary_dict[key]["n_eff"]))

    else:
        assert(False)



    if args.target == "HorseshoePriorLogisticRegression":
        
        print("**** VANILLA ***")
        all_results_flow_uncorrected, beta_mean_uncorrected = getExpectedBetaStatistics(beta_samples_approx, intercept_samples_approx, target)
        np.save(infoStr_for_WD + "_uncorrected_beta_stats", all_results_flow_uncorrected)
        
        if args.flow_type != "HMC":
            print("**** ADJUSTED ***")
            all_results_flow_adjusted, beta_mean_adjusted = getExpectedBetaStatistics(beta_samples_approx_adjusted_for_expectation, intercept_samples_approx, target)
            commons.saveStatistics(all_results_flow_adjusted, "adjusted_beta_stats")

            print("**** MCMC ***")
            infoStrMCMC = commons.getMCMC_spec_str(args.target, run_experiments.getDataDescriptorStr(args), commons.NUM_SAMPLES_MCMC, commons.NUM_WARMUP_MCMC, commons.NUM_CHAINS_MCMC, commons.THINNING_MCMC, device = "cpu")
            posterior_samples_MCMC = np.load("all_results/" + infoStrMCMC + "_" + "posterior_samples" + ".npy", allow_pickle = True).item()

            beta_samples_approx_MCMC = np.asarray(posterior_samples_MCMC["betas"])
            intercept_samples_approx_MCMC = np.asarray(posterior_samples_MCMC["intercept"])
            all_results_MCMC, beta_mean_MCMC = getExpectedBetaStatistics(beta_samples_approx_MCMC, intercept_samples_approx_MCMC, target)
            np.save("all_results/" + infoStrMCMC + "_MCMC_beta_stats", all_results_MCMC)

            stats_flow_to_mcmc = {}
            stats_flow_to_mcmc["beta_mean_uncorrected_MSE_to_truth"] = syntheticData.getMSE(beta_mean_uncorrected, target.true_beta.cpu().numpy())
            stats_flow_to_mcmc["beta_mean_adjusted_MSE_to_truth"] = syntheticData.getMSE(beta_mean_adjusted, target.true_beta.cpu().numpy())
            stats_flow_to_mcmc["beta_mean_uncorrected_MSE_to_MCMC"] = syntheticData.getMSE(beta_mean_uncorrected, beta_mean_MCMC)
            stats_flow_to_mcmc["beta_mean_adjusted_MSE_to_MCMC"] = syntheticData.getMSE(beta_mean_adjusted, beta_mean_MCMC)

            print("stats_flow_to_mcmc = ", stats_flow_to_mcmc)
            wasserstein_distance_theta = ot.sliced_wasserstein_distance(beta_samples_approx_MCMC, beta_samples_approx, n_projections = 1000, p=2, seed=4322)
            print(f"theta - wasserstein distance (MCMC,{args.flow_type}) = {wasserstein_distance_theta}")

            stats_flow_to_mcmc["wasserstein_distance_theta"] = wasserstein_distance_theta
            commons.saveStatistics(stats_flow_to_mcmc, "WD_to_HMC")

    else:
        print("samples_approx = ", samples_approx.shape)

        all_WDs = np.zeros(commons.REPETITIONS_FOR_MC_ERROR)

        for i in range(commons.REPETITIONS_FOR_MC_ERROR):
            print("start rep ", i)
            if args.target == "ConjugateLinearRegression":
                _,  samples_true = target.getSamplesFromTruePosterior(num_samples = TOTAL_NR_SAMPLES_FOR_EVALUATION)

            else:
                samples_true = target.getSamplesFromTruePosterior(num_samples = TOTAL_NR_SAMPLES_FOR_EVALUATION)

            wasserstein_distance_theta = ot.sliced_wasserstein_distance(samples_true, samples_approx, n_projections = 1000, p=2, seed=4322)
            print(f"theta - wasserstein distance (TRUE,{args.flow_type}) = {wasserstein_distance_theta}")
            all_WDs[i] = wasserstein_distance_theta
        
        np.save(infoStr_for_WD + "_WD_to_Truth", all_WDs)
        print("saved to ", (infoStr_for_WD + "_WD_to_Truth"))
        print("+ !! ++")

    print("TOTAL_NR_SAMPLES_FOR_EVALUATION = ", TOTAL_NR_SAMPLES_FOR_EVALUATION)

    print("FINISHED WD CALCULATION AND SAVED RESULTS")