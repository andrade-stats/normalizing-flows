import evaluation
import commons

import numpy
import target_distributions
import matplotlib.pyplot as plt

import run_experiments
import copy
import argparse
import pandas as pd

def getTableName(flow_type, method):
    if flow_type == "NeuralSpline":
        return "neural spline"
    elif (flow_type == "GaussianOnly") and (method is None):
        return "mean field gaussian"
    elif (flow_type == "HMC") and (method is None):
        return "HMC"
    elif flow_type == "smc":
        return "SMC"
    elif method == "proposed":
        return "proposed (gaussian, with LOFT)"
    elif method == "proposed_withStudentT":
        return "proposed (student, with LOFT)"
    elif method == "no_loft_proposed":
        return "proposed (gaussian, no LOFT)"
    elif method == "no_loft_proposed_withStudentT":
        return "proposed (student, no LOFT)"
    else:
        return method

def getTableNameForPlot(flow_type, method):
    if flow_type == "NeuralSpline":
        return "neural spline"
    elif (flow_type == "GaussianOnly") and (method is None):
        return "mean field gaussian"
    elif (flow_type == "HMC") and (method is None):
        return "HMC"
    elif flow_type == "smc":
        return "SMC"
    elif method == "proposed":
        return "proposed \n (gaussian, \n with LOFT)"
    elif method == "proposed_withStudentT":
        return "proposed \n (student, \n with LOFT)"
    elif method == "no_loft_proposed":
        return "proposed \n (gaussian, \n no LOFT)"
    elif method == "no_loft_proposed_withStudentT":
        return "proposed \n (student, \n no LOFT)"
    else:
        return method

def getTableNameForTargetModel(target_name):
    if target_name == "MultivariateStudentT":
        return "Multivariate Student-T"
    elif target_name == "MultivariateNormalMixture":
        return "Multivariate Gaussian Mixture"
    elif target_name == "ConjugateLinearRegression":
        return "Conjugate Linear Regression"
    elif target_name == "HorseshoePriorLogisticRegression":
        return "Horseshoe Logistic Regression Model"
    else:
        return target_name

def getNiceDataStr(data):
    if data == "colon":
        return "Colon"
    elif data == "multidrug_ABCB1":
        return "Multidrug"
    else:
        assert(False)
    

def convert_float_to_str(x):
    if "e" in str(x):
        return f"{x:.5f}"
    else:
        return str(x)


def showTable(target_name, show, nr_flows, d, full = True, extra_info_output = "", filterInfValues = True, annealing = None, divergence = None, use_best_result = True, iteration_setting = None, debug = False, lambd = 100.0, data = None, set_method = None):

    # annealing, divergence, and use_best_result are used showing the results of the ablation test -> at most one should be specified
    assert(((annealing is None) and (divergence is None) and use_best_result) or  \
        ((annealing is not None) and (divergence is None) and use_best_result) or \
        ((annealing is None) and (divergence is not None) and use_best_result) or \
        ((annealing is None) and (divergence is None) and (not use_best_result)))

    if set_method is not None:
        var = None
        ALL_METHODS = [set_method] # ("GaussianOnly", None)
    elif show == "gradVariance":
        var = None
        ALL_METHODS = [("RealNVP_small", "standard"), ("RealNVP_small", "ATAF"), ("RealNVP_small", "SymClip"), ("RealNVP_small", "proposed"), ("RealNVP_small", "proposed_withStudentT")]
    elif debug:
        var = None
        # var = 1000.0
        # ALL_METHODS = [("RealNVP_small", "standard"), ("RealNVP_small", "proposed"), ("RealNVP_small", "proposed_withStudentT"), ("RealNVP_small", "ATAF"), ("RealNVP_small", "SymClip")]
        # ALL_METHODS = [("RealNVP_small", "standard"), ("RealNVP_small", "proposed"), ("RealNVP_small", "proposed_withStudentT"), ("RealNVP_small", "SymClip")]
        # ALL_METHODS = [("RealNVP_small", "standard"), ("RealNVP_small", "ATAF"), ("RealNVP_small", "proposed"), ("RealNVP_small", "proposed_withStudentT"), ("RealNVP_small", "AsymClip_withStudentT"), ("RealNVP_small", "AsymClip"), ("RealNVP_small", "SymClip"), ("RealNVP_small", "SymClip_trainable_base")]
        # ALL_METHODS = [("RealNVP_small", "standard"), ("RealNVP_small", "ATAF"), ("RealNVP_small", "SymClip"), ("RealNVP_small", "proposed"), ("RealNVP_small", "proposed_withStudentT")]
        # ALL_METHODS = [("RealNVP_small", "standard"), ("RealNVP_small", "ATAF"), ("RealNVP_small", "SymClip"), ("RealNVP_small", "proposed_withStudentT")]
        # ALL_METHODS = [("RealNVP_small", "SymClip"), ("RealNVP_small", "proposed_withStudentT")]
    
        # ALL_METHODS = [("NeuralSpline", "standard"), ("RealNVP_small", "standard"), ("RealNVP_small", "ATAF"), ("RealNVP_small", "SymClip"), ("RealNVP_small", "proposed"), ("RealNVP_small", "proposed_withStudentT")]
        # ALL_METHODS = [("RealNVP_small", "standard"), ("RealNVP_small", "ATAF"), ("RealNVP_small", "SymClip"), ("RealNVP_small", "proposed"), ("RealNVP_small", "proposed_withStudentT"), ("RealNVP_small", "no_loft_proposed"), ("RealNVP_small", "no_loft_proposed_withStudentT")]
        ALL_METHODS = [("RealNVP_small", "standard"), ("RealNVP_small", "SymClip"), ("RealNVP_small", "proposed"), ("RealNVP_small", "proposed_withStudentT"), ("RealNVP_small", "no_loft_proposed"), ("RealNVP_small", "no_loft_proposed_withStudentT")]
        if show == "IS":
            ALL_METHODS.append(("smc", 100))

        assert(False)
    else:
        var = None
        
        if show == "time":
            ALL_METHODS = [("GaussianOnly", None), ("RealNVP_small", "standard"), ("RealNVP_small", "proposed_withStudentT")]
        elif full:
            ALL_METHODS = [("RealNVP_small", "standard"), ("RealNVP_small", "ATAF"), ("RealNVP_small", "SymClip"), ("RealNVP_small", "proposed"), ("RealNVP_small", "proposed_withStudentT"), ("RealNVP_small", "no_loft_proposed"), ("RealNVP_small", "no_loft_proposed_withStudentT"), ("GaussianOnly", None)]
        else:
            ALL_METHODS = [("GaussianOnly", None), ("RealNVP_small", "standard"), ("RealNVP_small", "ATAF"), ("RealNVP_small", "SymClip"), ("RealNVP_small", "proposed"), ("RealNVP_small", "proposed_withStudentT")]

        ALL_METHODS.append(("NeuralSpline", "standard"))

    if (not debug) and (show == "IS" or show == "time"):
        ALL_METHODS.append(("smc", 100000))

    if (not debug) and (show == "WD" or show == "time"):
       ALL_METHODS.append(("HMC", None))

    ROUND_DIGITS = 5

    # print("*******************")
    # print("********************** " + show + " **********************")
    # print("*******************")

    full_output_h = ""

    if data is not None:
        SETTINGS = [(-1, None, None, True)]
    elif d is None and data is None:
        SETTINGS = [(10, None, None, True), (100, None, None, True), (1000, None, None, True)]
    else:
        SETTINGS = [(d, None, None, True)]

        if (annealing is not None) or (divergence is not None) or (not use_best_result):
            SETTINGS.append((d, annealing, divergence, use_best_result))
    

    result_matrix_value = numpy.zeros((len(ALL_METHODS), len(SETTINGS)))
    result_matrix_std = numpy.zeros((len(ALL_METHODS), len(SETTINGS)))

    all_best_value_indices = numpy.zeros(len(SETTINGS)) * numpy.nan

    all_true_log_marginal_liks = numpy.zeros(len(SETTINGS)) * numpy.nan

    df = pd.DataFrame()

    for j, (D, annealing_local, divergence_local, use_best_result_local) in enumerate(SETTINGS):
        
        all_m = numpy.zeros(len(ALL_METHODS))
        all_std =  numpy.zeros(len(ALL_METHODS))
        all_available = numpy.zeros(len(ALL_METHODS))

        all_m_abs_diff = numpy.zeros(len(ALL_METHODS))
        # all_std_abs_diff =  numpy.zeros(len(ALL_METHODS))

        for i, (flow_type, method) in enumerate(ALL_METHODS):
            
            if flow_type == "NeuralSpline":
                nr_flows_this_time = 16
            else:
                nr_flows_this_time = nr_flows
            
            target, _, args = run_experiments.simple_init(target_name, D, flow_type, method, nr_flows_this_time, annealing = annealing_local, divergence = divergence_local, var = var, iteration_setting = iteration_setting, lambd = lambd, data = data)
            # print("args.lambd = ", args.lambd)
            print("target = ", target)
            print("target.true_log_marginal = ", target.true_log_marginal)

            available = True

            all_runtimes = None

            if flow_type == "HMC":
                
                infoStrMCMC = commons.getMCMC_spec_str(args.target, run_experiments.getDataDescriptorStr(args), commons.NUM_SAMPLES_MCMC, commons.NUM_WARMUP_MCMC, commons.NUM_CHAINS_MCMC, commons.THINNING_MCMC, device = "cpu")
                
                try:
                    if show == "time":
                        result = numpy.load("all_results/" + infoStrMCMC + "_timing_results.npy", allow_pickle = True).item()
                        all_runtimes = result["sampling_time_in_minutes"] * commons.REPETITIONS_FOR_MC_ERROR
                    else:
                        all_WDs = numpy.load("all_results/" + infoStrMCMC + "_WD_to_Truth.npy")

                except FileNotFoundError:
                    print(f"WARNING COULD NOT FIND FILE, spec = {flow_type} - {D}")
                    all_WDs = numpy.zeros(commons.REPETITIONS_FOR_MC_ERROR) * numpy.nan
                    available = False
                    assert(False)

                # print("all_WDs = ", all_WDs)
                # assert(False)

            elif flow_type == "smc":
                BASELINE_NAME = flow_type
                if args.target == "Funnel": # Funnel is too slow with SMC
                    BASELINE_NUM_TEMPS = 10000
                else:
                    BASELINE_NUM_TEMPS = method

                try:
                    infoStrSMC = commons.getSMC_spec_str(args.target, run_experiments.getDataDescriptorStr(args), BASELINE_NUM_TEMPS)
                    recorded_results = numpy.load("all_results/" + infoStrSMC + ".npy", allow_pickle = True).item()
                    all_IS = recorded_results["all_logZ"]
                    runtime_in_seconds = recorded_results["all_times"]
                    all_runtimes = (runtime_in_seconds / 60.0) * commons.REPETITIONS_FOR_MC_ERROR
                except FileNotFoundError:
                    print(f"WARNING COULD NOT FIND FILE, spec = {BASELINE_NAME} - {BASELINE_NUM_TEMPS} - {D}")
                    all_IS = numpy.zeros(commons.REPETITIONS_FOR_MC_ERROR) * numpy.nan
                    available = False
                    assert(False)
                
            else:
                
                try:
                    if use_best_result_local:
                        # best ELBO result over all iteration:
                        result = commons.loadStatistics("log_marginal_eval_result_best")
                    else:
                        # ELBO result after last iteration:
                        result = commons.loadStatistics("log_marginal_eval_result")
                    
                    nr_optimization_steps = commons.loadArray("optSteps").item()
                    
                    if show == "ELBO" or show == "IS":
                        all_ELBO = result["ELBO"]
                        all_IS = result["IS"]
                    elif show == "time":
                        all_runtimes = result["training_time_in_minutes"] + result["sampling_time_in_minutes"]
                        # print("training time = ", result["training_time_in_minutes"])
                        # print("sampling time = ", result["sampling_time_in_minutes"])
                    elif show == "gradVariance":
                        all_grads_norm = commons.loadArray("all_grads_norm_stats")
                        all_grads_norm_without_nan = commons.loadArray("all_grads_norm_without_nan_stats")
                    else:
                        all_WDs = numpy.load("all_results/" + commons.INFO_STR + "_WD_to_Truth.npy")

                except FileNotFoundError:
                    print(f"WARNING COULD NOT FIND FILE, spec = {flow_type} - {method} - {D}")
                    all_ELBO = numpy.zeros(commons.REPETITIONS_FOR_MC_ERROR) * numpy.nan
                    all_IS = numpy.zeros(commons.REPETITIONS_FOR_MC_ERROR) * numpy.nan
                    all_WDs = numpy.zeros(commons.REPETITIONS_FOR_MC_ERROR) * numpy.nan
                    all_runtimes = numpy.zeros(commons.REPETITIONS_FOR_MC_ERROR) * numpy.nan
                    available = False
                    # assert(False)
            
            if show == "ELBO":
                assert(all_ELBO.shape[0] == commons.REPETITIONS_FOR_MC_ERROR)
                if filterInfValues:
                    all_ELBO = commons.filterInfValues(all_ELBO)
                
                if method == "ATAF" and target_name == "HorseshoePriorLogisticRegression" and D == 1000 and filterInfValues:
                    # remove outlier from plot of ATAF for visualiziation (Note that this is in favor for ATAF)
                    print("***************************** WARNING ********************************")
                    print("before = ", all_ELBO)
                    med = numpy.nanmedian(all_ELBO)
                    all_ELBO[numpy.abs(all_ELBO - med) > 100000] = numpy.nan
                    print("after = ", all_ELBO)
                    # assert(False)
                
                all_results = all_ELBO
            elif show == "IS":
                assert(all_IS.shape[0] == commons.REPETITIONS_FOR_MC_ERROR)
                # assert(not numpy.any(numpy.isinf(all_IS)))
                all_results = all_IS
                if numpy.isnan(target.true_log_marginal):
                    all_results_abs_diff = None
                else:
                    # all_results_abs_diff = numpy.abs(all_IS - target.true_log_marginal)
                    # all_m_abs_diff[i], all_std_abs_diff[i] = evaluation.showAvgAndStd(all_results_abs_diff, ROUND_DIGITS = ROUND_DIGITS)

                    all_m_abs_diff[i] = numpy.abs(numpy.nanmean(all_IS) - target.true_log_marginal)
            elif show == "optSteps":
                all_results = numpy.asarray([nr_optimization_steps])
                assert(all_results.shape[0] == commons.REPETITIONS_FOR_MC_ERROR)
            elif show == "WD":
                assert(all_WDs.shape[0] == commons.REPETITIONS_FOR_MC_ERROR)
                # if numpy.any(numpy.isnan(all_WDs)):
                #     print("all_WDs = ", all_WDs)
                #     assert(False)
                all_results = all_WDs
            elif show == "gradVariance":
                all_results = all_grads_norm
                # print("all_results = ", all_results)
                # print("all_results.shape = ", all_results.shape)
                assert(all_results.shape[0] == 256)
            elif show == "time":
                all_results = all_runtimes
                # print("all_results = ", all_results)
                # assert(False)
            else:
                assert(False)
            
            if len(SETTINGS) == 1:
                method_name = getTableNameForPlot(flow_type, method)
                df.insert(len(df.columns), method_name, all_results)
                

            # print("flow_type = ", flow_type)
            # print("all_results = ", all_results)
            all_m[i], all_std[i] = evaluation.showAvgAndStd(all_results, ROUND_DIGITS = ROUND_DIGITS)
            all_available[i] = available
        

        result_matrix_value[:,j] = all_m
        result_matrix_std[:,j] = all_std
        
        if show == "ELBO" or ((show == "IS") and numpy.isnan(target.true_log_marginal)):
            best_value_index = numpy.nanargmax(all_m)
        elif (show == "IS") and (not numpy.isnan(target.true_log_marginal)):
            best_value_index = numpy.nanargmin(all_m_abs_diff)
            all_true_log_marginal_liks[j] = target.true_log_marginal
        elif show == "WD":
            best_value_index = numpy.nanargmin(all_m)
        else:
            best_value_index = -1
        
        all_methods_results = []
        for i, _ in enumerate(ALL_METHODS):
            if not all_available[i]:
                all_methods_results.append("missing")
            elif numpy.isnan(all_m[i]) or ~numpy.isfinite(all_m[i]):
                all_methods_results.append("NA")
            elif show == "time":
                value_m = convert_float_to_str(round(all_m[i], ROUND_DIGITS))
                all_methods_results.append(value_m)
            else:
                value_m = convert_float_to_str(round(all_m[i], ROUND_DIGITS))
                if i == best_value_index:
                    # highlight result
                    value_m =  "\\textbf{" + value_m + "}"
                
                all_methods_results.append(value_m + " (" + convert_float_to_str(round(all_std[i], ROUND_DIGITS)) + ")")

        if (annealing is None) and (divergence is None) and use_best_result:
            all_best_value_indices[j] = best_value_index

        full_output_h += "\n"
        full_output_h += f"${D}$ & " + " & ".join(all_methods_results) + " \\\\"

    
    if show == "ELBO":
        infoStr = target_name  # + "| learning rate = $10^{-" + str(lr_exp) + "}$" + ", " + annealing_info + ", " + divergence_info + extra_info
        output_text_h = "% " + infoStr + "\n"
        output_text_h += '''\multicolumn{4}{c}{  ''' + target_name + '''  } \\\\
\midrule ''' + extra_info_output + full_output_h + "\n" + "\midrule"
    # \\bfseries $d$ & \\bfseries  Gaussian-MF  &  \\bfseries  NF  &   \\bfseries NF+LOFT    \\\\
    # \midrule''' + full_output + "\n" + "\midrule")
        # print(full_output)
    
    else:
        infoStr = target_name
        output_text_h = '''\multicolumn{5}{c}{  ''' + infoStr + '''  } \\\\
\midrule ''' + extra_info_output + full_output_h + "\n" + "\midrule"


    if target_name == "HorseshoePriorLogisticRegression":
        infoText = f"{getTableNameForTargetModel(target_name)} ($r = {nr_flows}$) -- "
        if len(SETTINGS) == 3 or SETTINGS[0][0] == 1000:
            infoText += "Synthetic Data -- "
        else:
            infoText += "Colon Data -- "
        
        if show == "ELBO":
            infoText += "ELBO"
        elif show == "IS":
            infoText += "Log Marginal Likelihood Estimate"
        elif show == "time":
            infoText += "runtime in minutes"
        else:
            assert(False)

    else:    
        infoText = getTableNameForTargetModel(target_name)

    output_text_v = ""
    output_text_v += "\midrule " + "\n"
    COLUMN_NR = len(SETTINGS) + 1
    output_text_v += '''\multicolumn{''' + str(COLUMN_NR) + '''}{c}{  ''' + infoText + '''  } \\\\ '''  + "\n"
    output_text_v += "\midrule " + "\n"

    if (annealing is not None) or (divergence is not None) or (not use_best_result):
        assert(len(SETTINGS) == 2)
        column_explanation_str = []
        for (_, annealing_local, divergence_local, use_best_result_local) in SETTINGS:
            if annealing_local is not None:
                assert(annealing_local == "yes" and divergence_local is None)
                column_explanation_str.append("with annealing")
            elif not use_best_result_local:
                assert(annealing_local is None and divergence_local is None)
                column_explanation_str.append("last model")
            elif divergence_local is not None:
                assert(divergence_local == "reverse_kld" and annealing_local is None)
                column_explanation_str.append("with score term")
            else:
                if annealing is not None:
                    column_explanation_str.append("without annealing")  # without score term and without annealing (and best result)
                elif not use_best_result:
                    column_explanation_str.append("lowest loss model")  # (without score term and without annealing) and best result
                else:
                    assert(divergence is not None)
                    column_explanation_str.append("without score term")  # without score term and without annealing (and best result)
    else:
        if target_name == "ConjugateLinearRegression":
            column_explanation_str = [f"\\bfseries $d = {d + 1}$" for (d, _, _, _) in SETTINGS]
        elif target_name == "HorseshoePriorLogisticRegression":
            column_explanation_str = [f"\\bfseries $d' = {d}$" for (d, _, _, _) in SETTINGS]
        else:
            column_explanation_str = [f"\\bfseries $d = {d}$" for (d, _, _, _) in SETTINGS]

    output_text_v += ''' \\bfseries Method & ''' + " & ".join(column_explanation_str) +  ''' \\\\  '''
    output_text_v += "\n"
    output_text_v += "\midrule " + "\n"

    if (show == "IS") and (not numpy.isnan(target.true_log_marginal)):
        output_text_v += "true value" + " & "
        value_strs = [str(round(v, ROUND_DIGITS)) for v in all_true_log_marginal_liks]
        output_text_v += " & ".join(value_strs)
        output_text_v +=  '''  \\\\ '''  + "\n"
        output_text_v += "\midrule" + "\n"

    for i, (flow_type, method) in enumerate(ALL_METHODS):
        output_text_v += getTableName(flow_type, method) + " & "

        results_one_row = []

        if (annealing is not None) or (divergence is not None) or (not use_best_result):
            # compare across settings
            best_setting_id = numpy.nanargmax(result_matrix_value[i,:])
        else:
            best_setting_id = None
        
        for j, _ in enumerate(SETTINGS):
            value = convert_float_to_str(round(result_matrix_value[i,j], ROUND_DIGITS))
            std = convert_float_to_str(round(result_matrix_std[i,j], ROUND_DIGITS))
            if (i == all_best_value_indices[j]) or (j == best_setting_id):
                results_one_row.append(" \\textbf{" +  value + "} " + f"({std})")
            elif show == "time":
                value = convert_float_to_str(int(round(result_matrix_value[i,j], -1)))
                results_one_row.append(f" {value}")
            else:
                results_one_row.append(f" {value} ({std})")

        output_text_v += " & ".join(results_one_row) +  ''' \\\\  '''
        output_text_v += "\n"
        if method == "no_loft_proposed_withStudentT":
            output_text_v += "\midrule " + "\n"

    
    output_text_v = output_text_v.strip()

    return output_text_h, output_text_v, (df, target.true_log_marginal)




def showAblationTable(target_name, show, nr_flows, d, full = True, extra_info_output = "", filterInfValues = True, iteration_setting = None, debug = False, lambd = 100.0, data = None, set_method = None):


    # annealing = None # "yes"
    # divergence = None # "reverse_kld"
    # use_best_result = True # False

    ALL_METHODS = [(set_method[0], set_method[1], None, None, True), (set_method[0], set_method[1], "yes", None, True), (set_method[0], set_method[1], None, "reverse_kld", True), (set_method[0], set_method[1], None, None, False)] 
    
    
    ROUND_DIGITS = 5

    # print("*******************")
    # print("********************** " + show + " **********************")
    # print("*******************")

    full_output_h = ""

    # result_matrix_value = numpy.zeros(len(ALL_METHODS))
    # result_matrix_std = numpy.zeros(len(ALL_METHODS))

    # all_best_value_indices = numpy.zeros(len(SETTINGS)) * numpy.nan
    # all_true_log_marginal_liks = numpy.zeros(len(SETTINGS)) * numpy.nan

    df = pd.DataFrame()

    all_m = numpy.zeros(len(ALL_METHODS))
    all_std =  numpy.zeros(len(ALL_METHODS))
    all_available = numpy.zeros(len(ALL_METHODS))

    all_m_abs_diff = numpy.zeros(len(ALL_METHODS))
    
    for i, (flow_type, method, annealing_local, divergence_local, use_best_result_local) in enumerate(ALL_METHODS):
        
        if flow_type == "NeuralSpline":
            nr_flows_this_time = 16
        else:
            nr_flows_this_time = nr_flows
        
        target, _, args = run_experiments.simple_init(target_name, d, flow_type, method, nr_flows_this_time, annealing = annealing_local, divergence = divergence_local, var = None, iteration_setting = iteration_setting, lambd = lambd, data = data)
        # print("args.lambd = ", args.lambd)
        print("target = ", target)
        print("target.true_log_marginal = ", target.true_log_marginal)

        available = True

        all_runtimes = None
  
            
        try:
            if use_best_result_local:
                # best ELBO result over all iteration:
                result = commons.loadStatistics("log_marginal_eval_result_best")
            else:
                # ELBO result after last iteration:
                result = commons.loadStatistics("log_marginal_eval_result")
            
            all_ELBO = result["ELBO"]
            
        except FileNotFoundError:
            print(f"WARNING COULD NOT FIND FILE, spec = {flow_type} - {method} - {d}")
            all_ELBO = numpy.zeros(commons.REPETITIONS_FOR_MC_ERROR) * numpy.nan
            available = False
            assert(False)
    
        assert(all_ELBO.shape[0] == commons.REPETITIONS_FOR_MC_ERROR)
        if filterInfValues:
            all_ELBO = commons.filterInfValues(all_ELBO)
        
        # if method == "ATAF" and target_name == "HorseshoePriorLogisticRegression" and D == 1000 and filterInfValues:
        #     # remove outlier from plot of ATAF for visualiziation (Note that this is in favor for ATAF)
        #     print("***************************** WARNING ********************************")
        #     print("before = ", all_ELBO)
        #     med = numpy.nanmedian(all_ELBO)
        #     all_ELBO[numpy.abs(all_ELBO - med) > 100000] = numpy.nan
        #     print("after = ", all_ELBO)
        #     # assert(False)
        
        all_results = all_ELBO
        
        
        # if len(SETTINGS) == 1:
        # method_name = getTableNameForPlot(flow_type, method)
        # df.insert(len(df.columns), method_name, all_results)
        

        # print("flow_type = ", flow_type)
        # print("all_results = ", all_results)
        all_m[i], all_std[i] = evaluation.showAvgAndStd(all_results, ROUND_DIGITS = ROUND_DIGITS)
        all_available[i] = available
    
    # print("all_m = ", all_m)
    # assert(False)

    # best_value_index = numpy.nanargmax(all_m)
    
    # all_methods_results = []
    # for i, _ in enumerate(ALL_METHODS):
    #     if not all_available[i]:
    #         all_methods_results.append("missing")
    #     elif numpy.isnan(all_m[i]) or ~numpy.isfinite(all_m[i]):
    #         all_methods_results.append("NA")
    #     else:
    #         value_m = convert_float_to_str(round(all_m[i], ROUND_DIGITS))
    #         if i == best_value_index:
    #             # highlight result
    #             value_m =  "\\textbf{" + value_m + "}"
            
    #         all_methods_results.append(value_m + " (" + convert_float_to_str(round(all_std[i], ROUND_DIGITS)) + ")")

    
    # full_output_h += "\n"
    # full_output_h += f"${d}$ & " + " & ".join(all_methods_results) + " \\\\"

    
    

    infoText = getTableNameForTargetModel(target_name)

    output_text_v = ""
    output_text_v += "\midrule " + "\n"
    COLUMN_NR = 5
    output_text_v += '''\multicolumn{''' + str(COLUMN_NR) + '''}{c}{  ''' + infoText + '''  } \\\\ '''  + "\n"
    output_text_v += "\midrule " + "\n"

    # if (annealing is not None) or (divergence is not None) or (not use_best_result):
    #     assert(len(SETTINGS) == 2)
    #     column_explanation_str = []
    #     for (_, annealing_local, divergence_local, use_best_result_local) in SETTINGS:
    #         if annealing_local is not None:
    #             assert(annealing_local == "yes" and divergence_local is None)
    #             column_explanation_str.append("with annealing")
    #         elif not use_best_result_local:
    #             assert(annealing_local is None and divergence_local is None)
    #             column_explanation_str.append("last model")
    #         elif divergence_local is not None:
    #             assert(divergence_local == "reverse_kld" and annealing_local is None)
    #             column_explanation_str.append("with score term")
    #         else:
    #             if annealing is not None:
    #                 column_explanation_str.append("without annealing")  # without score term and without annealing (and best result)
    #             elif not use_best_result:
    #                 column_explanation_str.append("lowest loss model")  # (without score term and without annealing) and best result
    #             else:
    #                 assert(divergence is not None)
    #                 column_explanation_str.append("without score term")  # without score term and without annealing (and best result)
    # else:
    #     if target_name == "ConjugateLinearRegression":
    #         column_explanation_str = [f"\\bfseries $d = {d + 1}$" for (d, _, _, _) in SETTINGS]
    #     elif target_name == "HorseshoePriorLogisticRegression":
    #         column_explanation_str = [f"\\bfseries $d' = {d}$" for (d, _, _, _) in SETTINGS]
    #     else:
    #         column_explanation_str = [f"\\bfseries $d = {d}$" for (d, _, _, _) in SETTINGS]

    column_explanation_str = "DEBUG"

    # output_text_v += ''' \\bfseries Method & ''' + " & ".join(column_explanation_str) +  ''' \\\\  '''
    output_text_v += ''' \\bfseries Method & ''' + " annealing & score term & loss " + " & " + "ELBO" +  ''' \\\\  '''
    output_text_v += "\n"
    output_text_v += "\midrule " + "\n"

    best_setting_id = numpy.nanargmax(all_m)
    
    def getSpecStr(annealing, divergence, use_best_result):
        spec = ""
        if annealing is None:
            spec += " & without annealing "
        else:
            assert(annealing == "yes")
            spec += " & with annealing "
        
        if divergence is None:
            spec += " & without score term "
        else:
            assert(divergence == "reverse_kld")
            spec += " & with score term "
        
        if use_best_result:
            spec += "& lowest loss model "
        else:
            spec += "& last model "
        
        # return " - " + spec + ""
        return spec
    

    for i, (flow_type, method, annealing_local, divergence_local, use_best_result_local) in enumerate(ALL_METHODS):
        output_text_v += getTableName(flow_type, method) + getSpecStr(annealing_local, divergence_local, use_best_result_local) + " & "

        results_one_row = []
        
        value = convert_float_to_str(round(all_m[i], ROUND_DIGITS))
        std = convert_float_to_str(round(all_std[i], ROUND_DIGITS))
        if i == best_setting_id:
            results_one_row.append(" \\textbf{" +  value + "} " + f"({std})")
        else:
            results_one_row.append(f" {value} ({std})")

        output_text_v += " & ".join(results_one_row) +  ''' \\\\  '''
        output_text_v += "\n"
        if method == "no_loft_proposed_withStudentT":
            output_text_v += "\midrule " + "\n"

    output_text_v = output_text_v.strip()

    return output_text_v






def append_results(results_one_row, all_m, all_std, i, best_id, ROUND_DIGITS, smc = False):
    value = convert_float_to_str(round(all_m[i], ROUND_DIGITS))
    std = convert_float_to_str(round(all_std[i], ROUND_DIGITS))
    if (i == best_id):
        results_one_row.append(" \\textbf{" +  value + "} " + f"({std})")
    else:
        if smc and numpy.isnan(float(value)):
            results_one_row.append(f" (not available) ")
        elif numpy.isnan(float(value)) or (numpy.abs(float(value)) > 813909528707761041603297280):
            results_one_row.append(f" nan (nan) ")
        else:
            results_one_row.append(f" {value} ({std})")

    return

def showTable_ELBO_and_IS(target_name, nr_flows = 64, d = None, filterInfValues = True,  iteration_setting = None, lambd = 100.0, data = None):
    
    ALL_METHODS = [("RealNVP_small", "standard"), ("RealNVP_small", "ATAF"), ("RealNVP_small", "SymClip"), ("RealNVP_small", "proposed"), ("RealNVP_small", "proposed_withStudentT"), ("RealNVP_small", "no_loft_proposed"), ("RealNVP_small", "no_loft_proposed_withStudentT"), ("GaussianOnly", None)]
    ALL_METHODS.append(("NeuralSpline", "standard"))
    ALL_METHODS.append(("smc", 100000))
    
    ROUND_DIGITS = 5

    all_m = {}
    all_m["ELBO"] = numpy.zeros(len(ALL_METHODS))
    all_m["IS"] = numpy.zeros(len(ALL_METHODS))

    all_std = {}
    all_std["ELBO"] = numpy.zeros(len(ALL_METHODS))
    all_std["IS"] = numpy.zeros(len(ALL_METHODS))

    all_available = numpy.zeros(len(ALL_METHODS))

    if data is None:
        D = d
    else:
        D = -1

    true_available = None

    for i, (flow_type, method) in enumerate(ALL_METHODS):
        
        all_ELBO = None
        all_IS = None

        if flow_type == "NeuralSpline":
            nr_flows_this_time = 16
        else:
            nr_flows_this_time = nr_flows
        
        target, _, args = run_experiments.simple_init(target_name, D, flow_type, method, nr_flows_this_time, iteration_setting = iteration_setting, lambd = lambd, data = data)
        
        available = True

        if flow_type == "smc":
            BASELINE_NAME = flow_type
            if args.target == "Funnel": # Funnel is too slow with SMC
                BASELINE_NUM_TEMPS = 10000
            else:
                BASELINE_NUM_TEMPS = method

            try:
                infoStrSMC = commons.getSMC_spec_str(args.target, run_experiments.getDataDescriptorStr(args), BASELINE_NUM_TEMPS)
                recorded_results = numpy.load("all_results/" + infoStrSMC + ".npy", allow_pickle = True).item()
                all_IS = recorded_results["all_logZ"]
            except FileNotFoundError:
                print(f"WARNING COULD NOT FIND FILE, spec = {BASELINE_NAME} - {BASELINE_NUM_TEMPS} - {D}")
                all_IS = numpy.zeros(commons.REPETITIONS_FOR_MC_ERROR) * numpy.nan
                available = False
                
            
            all_ELBO = numpy.zeros(commons.REPETITIONS_FOR_MC_ERROR) * numpy.nan

        else:
            
            try:
                # best ELBO result over all iteration:
                result = commons.loadStatistics("log_marginal_eval_result_best")
                
                nr_optimization_steps = commons.loadArray("optSteps").item()
                
                all_ELBO = result["ELBO"]
                all_IS = result["IS"]
                
            except FileNotFoundError:
                print(f"WARNING COULD NOT FIND FILE, spec = {flow_type} - {method} - {D}")
                all_ELBO = numpy.zeros(commons.REPETITIONS_FOR_MC_ERROR) * numpy.nan
                all_IS = numpy.zeros(commons.REPETITIONS_FOR_MC_ERROR) * numpy.nan
                available = False
                
        
        if filterInfValues:
            all_ELBO = commons.filterInfValues(all_ELBO)

        
        if numpy.isnan(target.true_log_marginal):
            LOG_MARGINAL_TEXT = "Log Marginal Likelihood"
            true_available = False
        else:
            all_IS = numpy.abs(all_IS - target.true_log_marginal)
            LOG_MARGINAL_TEXT = "Error in Marginal Likelihood"
            true_available = True

        assert(all_IS.shape[0] == commons.REPETITIONS_FOR_MC_ERROR)

        all_m["ELBO"][i], all_std["ELBO"][i] = evaluation.showAvgAndStd(all_ELBO, ROUND_DIGITS = ROUND_DIGITS)
        all_m["IS"][i], all_std["IS"][i] = evaluation.showAvgAndStd(all_IS, ROUND_DIGITS = ROUND_DIGITS)
        all_available[i] = available
    
    # replace -inf value (occured once in Bayessian Lasso of standard method)
    all_m["ELBO"][numpy.isinf(all_m["ELBO"])] = numpy.nan
    all_m["IS"][numpy.isinf(all_m["IS"])] = numpy.nan

    best_id = {}
    best_id["ELBO"] = numpy.nanargmax(all_m["ELBO"])
    if true_available:
        best_id["IS"] = numpy.nanargmin(all_m["IS"])
    else:
        best_id["IS"] = numpy.nanargmax(all_m["IS"])
    
    
    infoText = getTableNameForTargetModel(target_name)
    if data is not None:
        infoText += " -- " + getNiceDataStr(data)
    elif target_name == "HorseshoePriorLogisticRegression" or target_name == "BayesianLasso":
        assert(d == 1000)
        infoText += " -- " + "Synthetic data ($d'=1000$)"
    else:
        infoText += " -- " + "$d'=1000$"
    
    output_text_v = ""
    output_text_v += "\midrule " + "\n"
    COLUMN_NR = 3
    output_text_v += '''\multicolumn{''' + str(COLUMN_NR) + '''}{c}{  ''' + infoText + '''  } \\\\ '''  + "\n"
    output_text_v += "\midrule " + "\n"

    output_text_v += ''' \\bfseries Method ''' + " & " + "ELBO" + " & " + LOG_MARGINAL_TEXT +  ''' \\\\  '''
    output_text_v += "\n"
    output_text_v += "\midrule " + "\n"


    for i, (flow_type, method) in enumerate(ALL_METHODS):
        output_text_v += getTableName(flow_type, method) + " & "

        results_one_row = []

        append_results(results_one_row, all_m["ELBO"], all_std["ELBO"], i, best_id["ELBO"], ROUND_DIGITS = ROUND_DIGITS, smc = (flow_type == "smc"))
        append_results(results_one_row, all_m["IS"], all_std["IS"], i, best_id["IS"], ROUND_DIGITS = ROUND_DIGITS)


        output_text_v += " & ".join(results_one_row) +  ''' \\\\  '''
        output_text_v += "\n"
        if method == "no_loft_proposed_withStudentT":
            output_text_v += "\midrule " + "\n"


    output_text_v = output_text_v.strip()

    print("***********************")
    print(output_text_v)

    return output_text_v


def showTable_gradVariance(target_name, nr_flows, d, annealing = None, divergence = None, use_best_result = True, iteration_setting = None, lambd = 100.0, data = None):

    SAMPLE_SIZE = 256

    if target_name == "BayesianLasso":
        data="multidrug_ABCB1"
    elif target_name == "HorseshoePriorLogisticRegression":
        data="colon"

    # annealing, divergence, and use_best_result are used showing the results of the ablation test -> at most one should be specified
    assert(((annealing is None) and (divergence is None) and use_best_result) or  \
        ((annealing is not None) and (divergence is None) and use_best_result) or \
        ((annealing is None) and (divergence is not None) and use_best_result) or \
        ((annealing is None) and (divergence is None) and (not use_best_result)))

    
    ALL_METHODS = [("RealNVP_small", "standard"), ("RealNVP_small", "ATAF"), ("RealNVP_small", "SymClip"), ("RealNVP_small", "proposed"), ("RealNVP_small", "proposed_withStudentT")]
    
    
    ROUND_DIGITS = 5

    full_output_h = ""

    if data is not None:
        SETTINGS = [(-1, None, None, True)]
    elif d is None and data is None:
        SETTINGS = [(10, None, None, True), (100, None, None, True), (1000, None, None, True)]
    else:
        SETTINGS = [(d, None, None, True)]

        if (annealing is not None) or (divergence is not None) or (not use_best_result):
            SETTINGS.append((d, annealing, divergence, use_best_result))
    

    result_matrix_value = numpy.zeros((len(ALL_METHODS), len(SETTINGS)))
    result_matrix_std = numpy.zeros((len(ALL_METHODS), len(SETTINGS)))

    all_best_value_indices = numpy.zeros(len(SETTINGS)) * numpy.nan

    all_true_log_marginal_liks = numpy.zeros(len(SETTINGS)) * numpy.nan

    df = pd.DataFrame()

    for j, (D, annealing_local, divergence_local, use_best_result_local) in enumerate(SETTINGS):
        
        all_vars = numpy.zeros(len(ALL_METHODS))
        all_nr_invalid =  numpy.zeros(len(ALL_METHODS))
        
        for i, (flow_type, method) in enumerate(ALL_METHODS):
            
            if flow_type == "NeuralSpline":
                nr_flows_this_time = 16
            else:
                nr_flows_this_time = nr_flows
            
            target, _, args = run_experiments.simple_init(target_name, D, flow_type, method, nr_flows_this_time, annealing = annealing_local, divergence = divergence_local, iteration_setting = iteration_setting, lambd = lambd, data = data)
            
            try:
                all_invalid_values = commons.loadArray("all_invalid_values_stats_new2")
                all_grads_norm = commons.loadArray("all_grads_norm_stats_new2")
                # if all_grads_norm.shape[0] == 0:
                #     all_grads_norm = numpy.zeros(SAMPLE_SIZE) * numpy.nan
                #     all_grads_norm_without_nan = numpy.zeros(SAMPLE_SIZE) * numpy.nan
                # # assert(all_grads_norm.shape[0] == SAMPLE_SIZE)
                # assert(all_grads_norm_without_nan.shape[0] == SAMPLE_SIZE)
            except FileNotFoundError:
                print(f"WARNING COULD NOT FIND FILE, spec = {flow_type} - {method} - {D}")
                all_invalid_values = numpy.zeros(SAMPLE_SIZE) * numpy.nan
                all_grads_norm = numpy.zeros(SAMPLE_SIZE) * numpy.nan
                
            # print("all_grads_norm = ", all_grads_norm)
            all_vars[i] = round(numpy.nanvar(all_grads_norm), ROUND_DIGITS)
            all_nr_invalid[i] = round(numpy.nanmean(all_invalid_values), ROUND_DIGITS)
            # all_vars[i] = round(numpy.nanvar(all_grads_norm_without_nan), ROUND_DIGITS)

    print("all_vars = ")
    print(all_vars)

    print("all_nr_invalid = ")
    print(all_nr_invalid)

    return all_nr_invalid


def showGradientVariances():
    
    nr_flows = 64
    d = 1000
    assert(nr_flows == 64 or nr_flows == 16)
    
    ALL_TARGET_NAMES = ["Funnel", "MultivariateNormalMixture", "ConjugateLinearRegression", "MultivariateStudentT", "BayesianLasso", "HorseshoePriorLogisticRegression"]
    # ALL_TARGET_NAMES = ["BayesianLasso"]
    # ALL_TARGET_NAMES = ["HorseshoePriorLogisticRegression"]
    # ALL_TARGET_NAMES = ["BayesianLasso"]
    
    all_table = []
    for target_name in ALL_TARGET_NAMES:
        
        all_nr_invalid = showTable_gradVariance(target_name = target_name, nr_flows = nr_flows, d = d)
        all_table.append(all_nr_invalid)
    
    print("all_table:")
    print(all_table)
    
    return

def showStandardModels(nr_flows, show = "ELBO_and_IS", d = None, annealing = None, divergence = None, full = True, filterInfValues = True, use_best_result = True, debug = False):
    assert(nr_flows == 64 or nr_flows == 16)
    
    ALL_TARGET_NAMES = ["Funnel", "MultivariateStudentT", "MultivariateNormalMixture", "ConjugateLinearRegression"]
    
    all_elbo_table_v = []
    all_is_table_v = []
    all_wd_table_v = []

    for target_name in ALL_TARGET_NAMES:
        
        if show == "WD":
            wd_table_h, wd_table_v, _ = showTable(target_name = target_name, show = "WD", nr_flows = nr_flows, d = d, full = full, annealing = annealing, divergence = divergence, filterInfValues = filterInfValues, use_best_result = use_best_result)
            all_wd_table_v.append(wd_table_v)
        else:
            elbo_table_h, elbo_table_v, _ = showTable(target_name = target_name, show = "ELBO", nr_flows = nr_flows, d = d, full = full, annealing = annealing, divergence = divergence, filterInfValues = filterInfValues, use_best_result = use_best_result, debug = debug)
            is_table_h, is_table_v, _ = showTable(target_name = target_name, show = "IS", nr_flows = nr_flows, d = d, full = full, annealing = annealing, divergence = divergence, filterInfValues = filterInfValues, use_best_result = use_best_result, debug = debug)
            all_elbo_table_v.append(elbo_table_v)
            all_is_table_v.append(is_table_v)
            

    print("NR_FLOWS = ", nr_flows)
    
    if show == "WD":
        all_WD_str = "\n".join(all_wd_table_v)
        print("******** WD ***********")
        print(all_WD_str)
        print("*******************")
    elif show == "ELBO":
        all_ELBO_str = "\n".join(all_elbo_table_v)
        print("******** ELBO ***********")
        print(all_ELBO_str)
        print("*******************")
    else:
        assert(show == "ELBO_and_IS")
        all_ELBO_str = "\n".join(all_elbo_table_v)
        print("******** ELBO ***********")
        print(all_ELBO_str)
        print("*******************")
        
        all_is_str = "\n".join(all_is_table_v)
        print("******** IS ***********")
        print(all_is_str)
        print("*******************")
    
    return


def showAblationTests():
    
    # showStandardModels(nr_flows = 64, show = "ELBO", d = 1000, use_best_result = False, filterInfValues = True, full = False)

    # shows all results using annealing:
    # showStandardModels(nr_flows = 64, show = "ELBO", d = 1000, annealing = "yes", divergence = None, filterInfValues = True, full = False)

    # shows all results using KL with score:
    # showStandardModels(nr_flows = 64, show = "ELBO", d = 1000, divergence = "reverse_kld", filterInfValues = True, full = False)

    # ALL_METHODS = [("GaussianOnly", None), ("RealNVP_small", "standard"), ("RealNVP_small", "ATAF"), ("RealNVP_small", "SymClip"), ("RealNVP_small", "proposed"), ("RealNVP_small", "proposed_withStudentT")]
    # method = ("GaussianOnly", None)
    # method = ("RealNVP_small", "standard")
    # method = ("RealNVP_small", "ATAF")
    # method = ("RealNVP_small", "SymClip")
    # method = ("RealNVP_small", "proposed")
    method = ("RealNVP_small", "proposed_withStudentT")
    
    nr_flows = 64
    d = 1000
    full = False
    show = "ELBO"
    debug = False
    filterInfValues = True
 
    # annealing = None # "yes"
    # divergence = None # "reverse_kld"
    # use_best_result = True # False
    
    ALL_TARGET_NAMES = ["Funnel", "MultivariateStudentT", "MultivariateNormalMixture", "ConjugateLinearRegression"]
    
    all_elbo_table_v = []
    all_is_table_v = []
    all_wd_table_v = []

    for target_name in ALL_TARGET_NAMES:
        
        elbo_table_v = showAblationTable(target_name = target_name, show = "ELBO", nr_flows = nr_flows, d = d, full = full, filterInfValues = filterInfValues, debug = debug, set_method = method)
        all_elbo_table_v.append(elbo_table_v)
        

    print("NR_FLOWS = ", nr_flows)
  
    all_ELBO_str = "\n".join(all_elbo_table_v)
    print("******** ELBO ***********")
    print(all_ELBO_str)
    print("*******************")

    return


def showChallengingTarget(nr_flows, target_name, d = None, data = None):
    assert(nr_flows == 64 or nr_flows == 16)
    assert(d is None or d == 2000 or d == 1000)
    
    ALL_TARGET_NAMES = [target_name]
    
    debug = False

    if target_name == "BayesianLasso":
        lambd = 100.0
    else:
        lambd = None
    
    # showTable(target_name, show, nr_flows, d, full = True, extra_info_output = "", filterInfValues = True, annealing = None, divergence = None, use_best_result = True, iteration_setting = None)

    all_elbo_table_v = []
    all_is_table_v = []
    for target_name in ALL_TARGET_NAMES:

        elbo_table_h, elbo_table_v, _ = showTable(target_name = target_name, show = "ELBO", nr_flows = nr_flows, d = d, full = True, data = data, lambd = lambd, debug = debug)
        is_table_h, is_table_v, _ = showTable(target_name = target_name, show = "IS", nr_flows = nr_flows, d = d, full = True, data = data, lambd = lambd, debug = debug)

        all_elbo_table_v.append(elbo_table_v)
        all_is_table_v.append(is_table_v)

    print("NR_FLOWS = ", nr_flows)
    all_ELBO_str = "\n".join(all_elbo_table_v)
    print("******** ELBO and IS ***********")
    print(all_ELBO_str)
    # print("*******************")

    # print("\midrule")

    all_is_str = "\n".join(all_is_table_v)
    # print("******** IS ***********")
    print(all_is_str)
    print("*******************")


def showNewExperiments(nr_flows, d = None, lambd = None):
    
    assert(nr_flows == 64 or nr_flows == 16)
    assert(d is None or d == 1000)
    
    # iteration_setting = "short_try"
    iteration_setting = None

    ALL_TARGET_NAMES = ["BayesianLasso"]
    # ALL_TARGET_NAMES = ["MultivariateStudentT"]
    # ALL_TARGET_NAMES = ["Funnel", "MultivariateStudentT", "MultivariateNormalMixture", "ConjugateLinearRegression"]
    

    all_elbo_table_v = []
    all_is_table_v = []
    for target_name in ALL_TARGET_NAMES:
        elbo_table_h, elbo_table_v, _ = showTable(target_name = target_name, show = "ELBO", nr_flows = nr_flows, d = d, full = True, iteration_setting = iteration_setting, debug = True, lambd = lambd)
        is_table_h, is_table_v, _ = showTable(target_name = target_name, show = "IS", nr_flows = nr_flows, d = d, full = True, iteration_setting = iteration_setting, debug = True, lambd = lambd)

        all_elbo_table_v.append(elbo_table_v)
        all_is_table_v.append(is_table_v)

    print("NR_FLOWS = ", nr_flows)
    all_ELBO_str = "\n".join(all_elbo_table_v)
    print("******** ELBO and IS ***********")
    print(all_ELBO_str)
    # print("*******************")

    # print("\midrule")

    all_is_str = "\n".join(all_is_table_v)
    # print("******** IS ***********")
    print(all_is_str)
    print("*******************")
    print("lambd = ", lambd)


def showStandardModelsSimple():
    all_txt = ""
    ALL_TARGET_NAMES = ["Funnel", "MultivariateStudentT", "MultivariateNormalMixture", "ConjugateLinearRegression"]
    for target_name in ALL_TARGET_NAMES:
        t = showTable_ELBO_and_IS(target_name, d = 1000) 
        all_txt += "\n"
        all_txt += t
    print("************************")
    print(all_txt)

def showChallengeModelsSimple():
    all_txt = ""

    all_txt += showTable_ELBO_and_IS(target_name = "BayesianLasso", d = 1000)
    all_txt += "\n"
    
    all_txt += showTable_ELBO_and_IS(target_name = "BayesianLasso", data = "multidrug_ABCB1") 
    all_txt += "\n"

    all_txt += showTable_ELBO_and_IS(target_name = "HorseshoePriorLogisticRegression", d = 1000) 
    all_txt += "\n"

    all_txt += showTable_ELBO_and_IS(target_name = "HorseshoePriorLogisticRegression", data = "colon") 
    all_txt += "\n"

    print("************************")
    print(all_txt)


if __name__ == "__main__":
    
    # showStandardModels(nr_flows = 64, show = "ELBO_and_IS", d = 1000, debug = True)
    # showChallengingTarget(nr_flows = 64, target_name = "BayesianLasso", d = 1000)
    # showChallengingTarget(nr_flows = 64, target_name = "BayesianLasso", data = "multidrug_ABCB1")

    # showNewExperiments(nr_flows = 64, d = 1000, lambd = 100.0)
    # # showNewExperiments(nr_flows = 64, d = 1000, lambd = 0.1)

    # _, output_text_v, _ = showTable(target_name = "BayesianLasso", show = "time", d = 1000, nr_flows = 64, lambd = 100.0, debug = True)
    # print("********************************")
    # print(output_text_v)


    # shows variance of gradients:
    # showGradientVariances()


    # shows runtime results:
    # _, output_text_v, _ = showTable(target_name = "HorseshoePriorLogisticRegression", show = "time", d = 1000, nr_flows = 64)
    # _, output_text_v, _ = showTable(target_name = "MultivariateStudentT", show = "time", d = 1000, nr_flows = 64)
    # print("********************************")
    # print(output_text_v)
    
    # shows all results using last model:
    # showStandardModels(nr_flows = 64, show = "ELBO", d = 1000, use_best_result = False, filterInfValues = True, full = False)

    # shows all results using annealing:
    # showStandardModels(nr_flows = 64, show = "ELBO", d = 1000, annealing = "yes", divergence = None, filterInfValues = True, full = False)

    # shows all results using KL with score:
    # showStandardModels(nr_flows = 64, show = "ELBO", d = 1000, divergence = "reverse_kld", filterInfValues = True, full = False)

    showAblationTests()

    # shows wasserstein distance results:
    # showStandardModels(nr_flows = 64, show = "WD")
    
    # show results for all standard models with standard setting
    # showStandardModels(nr_flows = 64, show = "ELBO_and_IS")
    # showStandardModels(nr_flows = 16, show = "ELBO_and_IS")

    # showStandardModelsSimple()
    # showChallengeModelsSimple()

    
