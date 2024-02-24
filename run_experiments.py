import evaluation
import commons
import torch
import time

import normalizing_flows_core as core

import estimators
import target_distributions
import matplotlib.pyplot as plt
import normflows
import argparse

import syntheticData
import numpy as np

def getAnnealingInterations(max_iterations):
    MAX_ANNEALING_ITERATIONS = 10000
    annealing_iters = int(max_iterations / 3)

    if annealing_iters >= MAX_ANNEALING_ITERATIONS:
        return MAX_ANNEALING_ITERATIONS
    else:
        return annealing_iters


# for num_samples might try up to 2 ** 15
def getArgumentParser(data = "no_data", n = None, d = None, foldId = None, D = None, target = None, flow_type = None, cushion_type = None, nr_cushions = None, 
                      nr_mixture_components = 1, divergence = "reverse_kld_without_score", num_samples = 2 ** 8,
                      nr_flows = None, l2_strength = 0.0, l2_clip_norm = None, loft_t = 100.0, intercept = None, rho = None, max_iterations = None, no_act_norm = "no", init = "zeros", annealing="yes", lr_exp=5, learn_mixture_weights = "no", targetK = -1, targetMeanFac = -1.0, analyzing = None, data_type = "double", realNVP_threshold = None, realNVP_variation = None, realNVP_spec = None, redGradVarEst = None, opt = "Adam", scaleShiftLayer = None, trainable_base = "yes", use_student_base = "no"):
    
    parser = argparse.ArgumentParser(description="NFM Experiments")
    
    parser.add_argument("--targetK", default=targetK, type=int)  
    parser.add_argument("--targetMeanFac", default=targetMeanFac, type=float)

    parser.add_argument("--data", default=data, type=str)
    parser.add_argument("--n", default=n, type=int)
    parser.add_argument("--d", default=d, type=int)
    parser.add_argument("--intercept", default=intercept, type=float)
    parser.add_argument("--foldId", default=foldId, type=int)
    parser.add_argument("--D", default=D, type=int)
    parser.add_argument("--target", default=target, type=str)
    parser.add_argument("--flow_type", default=flow_type, type=str)
    parser.add_argument("--cushion", default=cushion_type, type=str)
    parser.add_argument("--nr-cushions", default=nr_cushions, type=int)
    parser.add_argument("--initialization", default=init, type=str)
    
    parser.add_argument("--nr-mixture-components", default=nr_mixture_components, type=int) 
    parser.add_argument("--divergence", default=divergence, type=str)
    parser.add_argument("--num-samples", default=num_samples, type=int) 
    parser.add_argument("--nr-flows", default=nr_flows, type=int)
    parser.add_argument("--l2-strength", default=l2_strength, type=float)
    parser.add_argument("--l2-clip-norm", default=l2_clip_norm, type=float)
    parser.add_argument("--loft-t", default=loft_t, type=float) 
    parser.add_argument("--rho", default=rho, type=float)
    parser.add_argument("--max-iterations", default=max_iterations, type=int)
    parser.add_argument("--annealing", default=annealing, type=str)
    parser.add_argument("--lr-exp", default=lr_exp, type=int)
    
    parser.add_argument("--learn-mixture-weights", default=learn_mixture_weights, type=str)    

    parser.add_argument("--realNVP-threshold", default=realNVP_threshold, type=float)
    parser.add_argument("--realNVP-variation", default=realNVP_variation, type=str)
    parser.add_argument("--redGradVarEst", default=redGradVarEst, type=str)
    
    # baselines
    parser.add_argument("--realNVP-spec", default=realNVP_spec, type=str)
    
    # optional:
    parser.add_argument('--no-act-norm', default=no_act_norm, type=str)
    parser.add_argument("--analyzing", default=analyzing, type=str)
    parser.add_argument("--data-type", default=data_type, type=str)

    parser.add_argument("--opt", default=opt, type=str)
    parser.add_argument("--scaleShiftLayer", default=scaleShiftLayer, type=str)

    parser.add_argument('--trainable-base', default=trainable_base, type=str)
    parser.add_argument('--use-student-base', default=use_student_base, type=str)

    # only used for helper access
    parser.add_argument("--method", default=None, type=str)
    parser.add_argument("--flow-type", default=None, type=str)
    
    args = parser.parse_args()

    if args.nr_mixture_components == 1:
        assert(args.learn_mixture_weights == "no")
    else:
        assert(args.learn_mixture_weights == "no" or args.learn_mixture_weights == "yes")
    
    assert(args.no_act_norm == "no" or args.no_act_norm == "yes")
    assert(args.trainable_base == "no" or args.trainable_base == "yes")
    assert(args.use_student_base == "no" or args.use_student_base == "yes")

    return args


def getDataDescriptorStr(args):
    DATA_DESCRIPTOR_STR = args.data
    if args.data.startswith("synthetic"):
        DATA_DESCRIPTOR_STR += "_" + str(args.n) + "n"
        DATA_DESCRIPTOR_STR += "_" + str(args.d) + "d"
        DATA_DESCRIPTOR_STR += "_" + str(args.intercept) + "intercept"
        DATA_DESCRIPTOR_STR += "_" + str(args.rho) + "rho"
        DATA_DESCRIPTOR_STR += "_" + str(args.foldId) + "foldId"
    elif args.data == "no_data":
        DATA_DESCRIPTOR_STR += "_" + str(args.D) + "D"
    
    return DATA_DESCRIPTOR_STR

def load_data(args):
    assert(args.data.startswith("synthetic") or args.data == "COPD" or args.data == "colon")
            
    if args.data.startswith("synthetic"):
        X, y, true_beta, true_bias = syntheticData.load_data(args.data, n = args.n, d = args.d, rho = args.rho, bias = args.intercept, foldNr = args.foldId)
    else:
        whole_data = np.load("microarray_data_prepared/" + args.data + "_whole_data" + ".npy", allow_pickle=True).item()       
        X = whole_data["X"]
        y = whole_data["y"]
        true_beta = None
        true_bias = None
        assert(X.shape[1] == 2000)
        assert(X.shape[0] == 62)
    
    return X, y, true_beta, true_bias

def getMemoryUsage(model):
    mem_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
    mem = mem_params + mem_bufs
    return mem / (1024 * 1024 * 1024)  # get GB

def evaluate_marginal_likelihood(nfm):

    nfm.eval()
    with torch.no_grad():
        
        all_elbos = estimators.getRepeatedEstimates(nfm, type = "ELBO")
        # print("all_elbos = ", all_elbos)
        print("elbo = " + evaluation.showAvgAndStd_str(all_elbos))

        print("Chi2 = " + evaluation.showAvgAndStd_str( estimators.getRepeatedEstimates(nfm, type = "chi2")))
        print("forward-KLD = " + evaluation.showAvgAndStd_str(estimators.getRepeatedEstimates(nfm, type = "forward_kld")))

        print("IS = " + evaluation.showAvgAndStd_str(estimators.getRepeatedEstimates(nfm, type = "IS")))
    
    return


def initialize_target_and_flow(args, initialize = True):
    target_constructor = getattr(target_distributions, args.target)

    if args.target.startswith("Conjugate") or args.target.startswith("Horseshoe"):
        assert(args.D is None)

        if args.foldId == -1:
            target = None
            assert(not initialize)
        else:
            X, y, true_beta, true_bias = load_data(args)
            
            # print("use data with X.shape = ", X.shape)
            target = target_constructor(X, y, true_beta, true_bias)
    
    elif args.target == "MultivariateNormalMixture" or args.target == "StudentTMixture":
        target = target_constructor(dim = args.D, K = args.targetK, meanFac = args.targetMeanFac)
    else:
        assert(args.target == "Funnel" or args.target == "MultivariateStudentT")
        target = target_constructor(dim = args.D)
                

    if args.flow_type != "HMC" and args.flow_type != "smc":
        commons.INFO_STR = commons.getInfoStr_new(args, args.target, getDataDescriptorStr(args), args.flow_type, args.nr_flows, args.max_iterations, args.num_samples, args.divergence,  args.cushion, args.loft_t, args.l2_strength, args.l2_clip_norm, args.nr_cushions, args.initialization, args.annealing, args.lr_exp, args.nr_mixture_components, args.learn_mixture_weights, args.realNVP_threshold, args.realNVP_variation, args.realNVP_spec, args.redGradVarEst, args.opt, args.scaleShiftLayer, args.trainable_base, args.use_student_base)
    else:
        assert(not initialize)
        commons.INFO_STR = None
    
    if initialize:
        flows_mixture = core.FlowsMixture(target, K = args.nr_mixture_components, flow_type = args.flow_type, number_of_flows = args.nr_flows, nr_samples_for_act_norm = args.num_samples, initial_loc_spec = args.initialization, learn_mixture_weights = (args.learn_mixture_weights == "yes"), cushion_type = args.cushion, cushion_t = args.loft_t, nr_cushions = args.nr_cushions, no_act_norm = (args.no_act_norm == "yes"), realNVP_threshold = args.realNVP_threshold, realNVP_variation = args.realNVP_variation, realNVP_spec = args.realNVP_spec, redGradVarEst = args.redGradVarEst, scaleShiftLayer = args.scaleShiftLayer, trainable_base = args.trainable_base, use_student_base = args.use_student_base)
        
        print("*******")
        print("required memory (GB) by Model = ", getMemoryUsage(flows_mixture))
        print("*******")
    else:
        flows_mixture = None
    
    return target, flows_mixture


def visualize(prob_target, D, q1 = None, q2 = None, savefigInfo = None):

    x_values = torch.arange(-10, 10, step = 0.1, dtype = torch.double)
    x_values = commons.moveToDevice(x_values)

    if D == 1:
        coordinates = x_values.reshape(-1, 1)
    elif D >= 2:
        coordinates = x_values.reshape(-1, 1)
        coordinates = torch.tile(coordinates, (1, D))
    
    assert(coordinates.shape[1] == D)
    
    y_values_p = torch.exp(prob_target.log_prob(coordinates))
    
    y_values_q1 = torch.exp(q1.log_prob(coordinates))
   

    plt.plot(x_values.cpu(), y_values_p.cpu(), color = "blue") # target
    plt.plot(x_values.cpu(), y_values_q1.cpu(), color = "red") # approximation

    if q2 is not None:
        y_values_q2 = torch.exp(q2.log_prob(coordinates))
        plt.plot(x_values.cpu(), y_values_q2.cpu(), color = "green")
    
    
    if savefigInfo is None:
        plt.show()
    else:
        plt.savefig("all_results/" + savefigInfo + ".pdf")
    return


def simple_init(target_name, D, flow_type, method, nr_flows = 64, foldId = 1, annealing = None, divergence = None, initialize = False):
    
    setting = {}

    # *************** Model/Target *************
    if target_name == "HorseshoePriorLogisticRegression" and D <= 1000:
        setting["n"] = 100
        setting["d"] = D
        setting["rho"] = 0.1
        setting["intercept"] = 1.0
        setting["data"] = "synthetic_logistic"
        setting["target"] = "HorseshoePriorLogisticRegression"
        setting["foldId"] = foldId
    elif target_name == "HorseshoePriorLogisticRegression" and D == 2000:
        setting["data"] = "colon"
        setting["target"] = "HorseshoePriorLogisticRegression"
        setting["foldId"] = 0
    elif target_name == "ConjugateLinearRegression":
        setting["n"] = 100
        setting["d"] = D
        setting["rho"] = 0.5
        setting["intercept"] = 0.0
        setting["data"] = "synthetic_regression"
        setting["target"] = "ConjugateLinearRegression"
        setting["foldId"] = foldId
    elif target_name ==  "MultivariateNormalMixture":
        setting["target"] = "MultivariateNormalMixture"
        setting["targetK"] = 3
        setting["targetMeanFac"] = 6.0
        setting["D"] = D
    else:
        assert(target_name == "Funnel" or target_name == "MultivariateStudentT")
        setting["target"] = target_name
        setting["D"] = D


    # *************** NFM parameters *************
    assert(flow_type is not None)
    setting["flow_type"] = flow_type

    setting["nr_mixture_components"] = 1
    setting["init"] = "zeros"
    setting["lr_exp"] = 4

    setting["l2_strength"] = 0.0
    setting["annealing"] = "no"

    if annealing is not None:
        assert(annealing == "yes")
        setting["annealing"] = "yes"

    setting["data_type"] = "double"

    if setting["flow_type"] == "GaussianOnly":
        setting["divergence"] = "reverse_kld_without_score"
    elif setting["flow_type"]  == "smc" or setting["flow_type"]  == "HMC":
        print("flow-type = ", setting["flow_type"])
    else:

        setting["no_act_norm"] = "yes"

        setting["nr_flows"] = nr_flows

        setting["nr_cushions"] = 1
        setting["loft_t"] = 100.0
        setting["divergence"] = "reverse_kld_ws_debug"

        if method.startswith("no_loft_proposed"):
            setting["trainable_base"] = "no"
            setting["cushion_type"] = "none"
            setting["realNVP_threshold"] = 0.1
            setting["realNVP_variation"] = "var19"
            setting["scaleShiftLayer"] = "ssL"
            if method == "no_loft_proposed_withStudentT":
                setting["use_student_base"] = "yes"
            else:
                assert(method == "no_loft_proposed")
        elif method.startswith("proposed"):
            setting["trainable_base"] = "no"
            setting["cushion_type"] = "LOFT"
            setting["realNVP_threshold"] = 0.1
            setting["realNVP_variation"] = "var19"
            setting["scaleShiftLayer"] = "ssL"
            if method == "proposed_withStudentT":
                setting["use_student_base"] = "yes"
            else:
                assert(method == "proposed")
        elif method == "standard":
            setting["trainable_base"] = "yes"
            setting["cushion_type"] = "none"
        elif method == "ATAF":
            setting["trainable_base"] = "yes"
            setting["cushion_type"] = "none"
            setting["use_student_base"] = "yes" 
            setting["realNVP_variation"] = "tanh"
        elif method == "SymClip":
            setting["trainable_base"] = "no"
            setting["cushion_type"] = "none"
            setting["realNVP_threshold"] = 0.1
            setting["realNVP_variation"] = "var17"
            setting["scaleShiftLayer"] = "ssL"
        else:
           assert(False)
    
    if ("data" in setting) and (setting["data"] == "colon"):
        setting["max_iterations"] = 400000
    else:
        setting["max_iterations"] = 60000
    
    if divergence is not None:
        assert(divergence == "reverse_kld")
        setting["divergence"] = divergence
        setting["max_iterations"] = int(1.5 * setting["max_iterations"])
        print("it = ", setting["max_iterations"])
        # assert(False)

    args = getArgumentParser(**setting)

    commons.DATA_TYPE = args.data_type
    commons.setGPU()
    torch.manual_seed(432432)

    target, flows_mixture = initialize_target_and_flow(args, initialize)

    return target, flows_mixture, args


if __name__ == "__main__":

    MANUALLY_SPECIFY_ALL_ARGUMENTS = False

    print("normflows - Version = ", normflows.__version__)
    print("PyTorch - Version = ", torch.__version__)

    if MANUALLY_SPECIFY_ALL_ARGUMENTS:
        args = getArgumentParser()

        commons.DATA_TYPE = args.data_type
        commons.setGPU()
        torch.manual_seed(432432)
        
        target, flows_mixture = initialize_target_and_flow(args)
    else:
        # helper to easily use most common parameter settings
        # e.g.:
        # python run_experiments.py --target=Funnel --D=100 --flow-type=GaussianOnly
        # python run_experiments.py --target=Funnel --D=100 --flow-type=RealNVP_small --method=standard
        # python run_experiments.py --target=HorseshoePriorLogisticRegression --d=100 --flow-type=RealNVP_small --method=standard

        parser = argparse.ArgumentParser(description="Show Table")
        
        # parser.add_argument("--d", default=, type=int)
        parser.add_argument("--foldId", default=1, type=int)
        parser.add_argument("--D", default=-1, type=int)
        parser.add_argument("--d", default=-1, type=int)
        parser.add_argument("--target", default=None, type=str)
        parser.add_argument("--method", default=None, type=str)
        parser.add_argument("--flow-type", default=None, type=str)
        parser.add_argument("--nr-flows", default=64, type=int)
        parser.add_argument("--annealing", default=None, type=str)
        parser.add_argument("--divergence", default=None, type=str)
        real_args = parser.parse_args()

        target, flows_mixture, args = simple_init(real_args.target, max(real_args.D, real_args.d), real_args.flow_type, real_args.method, real_args.nr_flows, real_args.foldId, real_args.annealing, real_args.divergence, initialize = True)

    anneal_iter = getAnnealingInterations(args.max_iterations)
    if args.max_iterations < 1000:
        show_iter = 50
    else:
        show_iter = 500
    
    print("args.divergence = ", args.divergence)
    startTime = time.time()

    LEARNING_RATE = 10 **(-args.lr_exp)
    
    print("LEARNING_RATE = ", LEARNING_RATE)
    print("max_iterations = ", args.max_iterations)

    assert(args.flow_type == "GaussianOnly" or args.flow_type == "RealNVP_small")
    
    # if os.path.exists(commons.get_model_filename_best()):
    #    print("FOUND EXISTING MODEL - SKIP TRAINING - LOAD MODEL")
    #    flows_mixture.load_state_dict(torch.load(commons.get_model_filename_best(), map_location = commons.DEVICE))
    # else:
    nr_optimization_steps, current_best_true_loss = core.train(flows_mixture, args.max_iterations, anneal_iter, show_iter, learning_rate = LEARNING_RATE, record_stats=True, l2_strength = args.l2_strength, l2_clip_norm = args.l2_clip_norm, num_samples = args.num_samples, divergence = args.divergence, annealing = (args.annealing == "yes"), redGradVarEst = args.redGradVarEst, cushion_t = args.loft_t, opt = args.opt)

    print("*****************************")    
    print("nr_optimization_steps = ", nr_optimization_steps)
    print("*****************************")   

    training_time = (time.time() - startTime) / 60.0
    
    # torch.save(flows_mixture.state_dict(), commons.get_model_filename())
    # print("**** saved model successfully ****")   

    # ************ evaluate using last model *******************
    print("**** run sampling for ELBO and IS (last Model) *****")
    flows_mixture.eval()
    with torch.no_grad():
        startTime_sampling = time.time()
        all_ELBO = estimators.getRepeatedEstimates(flows_mixture, "ELBO")
        sampling_time = (time.time() - startTime_sampling) / 60.0
        all_IS = estimators.getRepeatedEstimates(flows_mixture, "IS")

    result = {}
    result["ELBO"] = all_ELBO
    result["IS"] = all_IS
    result["true_log_marginal"] = target.true_log_marginal
    result["training_time_in_minutes"] = training_time
    result["sampling_time_in_minutes"] = sampling_time
    commons.saveStatistics(result, "log_marginal_eval_result")

    # ************ evaluate using best model *******************
    flows_mixture.load_state_dict(torch.load(commons.get_model_filename_best(), map_location = commons.DEVICE))
    
    if args.target == "HorseshoePriorLogisticRegression" and args.data != "colon":
        print("**** run sampling for Horseshoe Evaluation *****")
        TOTAL_NR_SAMPLES_FOR_EVALUATION = 40000
        beta_samples_approx, intercept_samples_approx, samples_log_q, samples_log_p = target.get_samples_from_approx(flows_mixture, num_samples = TOTAL_NR_SAMPLES_FOR_EVALUATION)
        beta_mean_approx, intercept_mean_approx, iqr_approx, avg_mse_to_truth_all, avg_mse_to_truth_beta = syntheticData.show_regression_result_statistics(beta_samples_approx, intercept_samples_approx, target.true_beta.cpu().numpy(), target.true_intercept.cpu().numpy())
        print("avg_mse_to_truth_all = ", avg_mse_to_truth_all)
        print("avg_mse_to_truth_beta = ", avg_mse_to_truth_beta)
        print("iqr_approx = ", iqr_approx)
        horseshoe_cmp_result = {}
        horseshoe_cmp_result["avg_mse_to_truth_all"] = avg_mse_to_truth_all
        horseshoe_cmp_result["avg_mse_to_truth_beta"] = avg_mse_to_truth_beta
        horseshoe_cmp_result["iqr_approx"] = iqr_approx
        commons.saveStatistics(horseshoe_cmp_result, "horseshoe_cmp_result")

    print("**** run sampling for ELBO and IS (best Model) *****")
    flows_mixture.eval()
    with torch.no_grad():
        startTime_sampling = time.time()
        all_ELBO_bestModel = estimators.getRepeatedEstimates(flows_mixture, "ELBO")
        sampling_time = (time.time() - startTime_sampling) / 60.0
        all_IS_bestModel = estimators.getRepeatedEstimates(flows_mixture, "IS")

    result = {}
    result["ELBO"] = all_ELBO_bestModel
    result["IS"] = all_IS_bestModel
    result["true_log_marginal"] = target.true_log_marginal
    result["training_time_in_minutes"] = training_time
    result["sampling_time_in_minutes"] = sampling_time
    commons.saveStatistics(result, "log_marginal_eval_result_best")

 

    print(f"************************** Target = {target.__class__.__name__}, DATA_DESCRIPTOR_STR = {getDataDescriptorStr(args)}, D = {target.D} ****************** ")
    print("DIVERGENCE_FOR_OPTIMIZATION = ", args.divergence)
    print("NUM_SAMPLES = ", args.num_samples)
    print("CUSHION = ", args.cushion)
    print("cushion t-parameter = ",  args.loft_t)
    print("L2_STRENGTH = ", args.l2_strength)
    print("L2_CLIP_NORM = ", args.l2_clip_norm)
    print("**** results ******")
    print("MAX_ITERATION = ", args.max_iterations)
    print("training time (in minutes) = ", training_time)
    print("**** performance ******")
    print("******************* ")
    print("all_IS = ", all_IS)
    print("******************* ")
    print("TARGET_NAME = ", target.__class__.__name__)
    print("D = ", target.D)
    print("true_log_marginal = ", target.true_log_marginal)
    print("NUMBER_OF_MIXTURE_COMPONENTS = ", args.nr_mixture_components)
    print("FLOW_TYPE = ", args.flow_type)
    print("******************* ")
    print(f"ELBO (best model) = {evaluation.showAvgAndStd_str(all_ELBO_bestModel)}, IS (best model) = {evaluation.showAvgAndStd_str(all_IS_bestModel)}")
    print("******************* ")
    print(f"ELBO (last model)= {evaluation.showAvgAndStd_str(all_ELBO)}, IS (last model) = {evaluation.showAvgAndStd_str(all_IS)}")
    print("******************* ")
    print("commons.INFO_STR = ", commons.INFO_STR)
