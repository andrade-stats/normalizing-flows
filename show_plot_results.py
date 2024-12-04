import commons
import evaluation
import matplotlib.pyplot as plt

# import seaborn as sns

import show_table

def extract_and_remove_extra_info(df, remove_column_name):

    # if remove_column_name in df:
    all_results = df[remove_column_name]
    df = df.drop(labels = remove_column_name, axis = 1)

    all_results = commons.filterInfValues(all_results)
    
    ROUND_DIGITS = 2

    res = evaluation.showAvgAndStd_str(all_results, ROUND_DIGITS = ROUND_DIGITS)

    if res.startswith("nan"):
        res = "not available"
    
    # else:
    #     res = "not available"
    
    return df, remove_column_name + " = " + res


def showPlot(target_name, d = None, data = None):
    
    nr_flows = 64
    
    _, _, (df_elbo, _) = show_table.showTable(target_name = target_name, show = "ELBO", nr_flows = nr_flows, d = d, full = True, filterInfValues = True, data = data)
    _, _, (df_is, true_log_marginal) = show_table.showTable(target_name = target_name, show = "IS", nr_flows = nr_flows, d = d, full = True, data = data)

    df_elbo, elbo_mfg = extract_and_remove_extra_info(df_elbo, "mean field gaussian")
    df_elbo, elbo_ns = extract_and_remove_extra_info(df_elbo, "neural spline")

    df_is, log_marginal_mfg = extract_and_remove_extra_info(df_is, "mean field gaussian")
    df_is, log_marginal_ns = extract_and_remove_extra_info(df_is, "neural spline")
    df_is, log_marginal_smc = extract_and_remove_extra_info(df_is, "SMC")

    if target_name == "BayesianLasso":
        df_elbo, elbo_standard = extract_and_remove_extra_info(df_elbo, "standard")
        elbo_additional_methods = "\n" + elbo_standard
        df_is, log_marginal_standard = extract_and_remove_extra_info(df_is, "standard")
        log_marginal_additional_methods = "\n" + log_marginal_standard

        df_elbo, elbo_ATAF = extract_and_remove_extra_info(df_elbo, "ATAF")
        elbo_additional_methods += "\n" + elbo_ATAF
        df_is, log_marginal_ATAF = extract_and_remove_extra_info(df_is, "ATAF")
        log_marginal_additional_methods += "\n" + log_marginal_ATAF
        
    else:
        elbo_additional_methods = ""
        log_marginal_additional_methods = ""

    fig, all_subplots = plt.subplots(2, figsize=(15, 20))
    
    # fig.suptitle(target_name, fontsize=20)

    current_sub_plot = all_subplots[0]
    current_sub_plot.set_title("ELBO: \n" + elbo_mfg + "\n" + elbo_ns + elbo_additional_methods, fontsize = 20, loc="left")
    df_elbo.boxplot(ax = current_sub_plot, fontsize = 20)
    current_sub_plot.set_ylabel("ELBO", fontsize=20)

    # current_sub_plot = all_subplots[1]
    # current_sub_plot.set_title("log marginal likelihood: \n" + log_marginal_mfg + "\n" + log_marginal_ns + log_marginal_additional_methods + "\n" + log_marginal_smc, fontsize = 20, loc="left") 
    # df_is.boxplot(ax = current_sub_plot, fontsize = 20)
    # current_sub_plot.set_ylabel("log marginal likelihood", fontsize=20)

    plt.axhline(true_log_marginal, c='r')

    if d is not None:
        assert(data is None)
        file_post_fix = str(d) + "d" 
    else:
        file_post_fix = data

    # DIRECTORY = "all_plots_final/"
    DIRECTORY = "../NormalizingFlows_private/latex/IOP/"
    
    OUTPUT_FILENAME = DIRECTORY + "results_ELBO_IS_" + target_name + "_" + file_post_fix + ".png"
    fig.tight_layout(pad = 3.0)
    # fig.savefig(OUTPUT_FILENAME)
    plt.show()

    print("FINISHED")
    return


def showPlot_simple(target_name, d = None, data = None, save = False, with_marginal_likelihood = True):
    
    nr_flows = 64
    
    _, _, (df_elbo, _) = show_table.showTable(target_name = target_name, show = "ELBO", nr_flows = nr_flows, d = d, full = True, filterInfValues = True, data = data)
    _, _, (df_is, true_log_marginal) = show_table.showTable(target_name = target_name, show = "IS", nr_flows = nr_flows, d = d, full = True, data = data)

    # print(df_elbo)
    # assert(False)

    # "proposed_withStudentT"
    porposed_method_variations = ["proposed",  "no_loft_proposed", "no_loft_proposed_withStudentT"]
    for method in porposed_method_variations:
        df_elbo, _ = extract_and_remove_extra_info(df_elbo, show_table.getTableNameForPlot("RealNVP_small", method))
        df_is, _ = extract_and_remove_extra_info(df_is, show_table.getTableNameForPlot("RealNVP_small", method))

    df_elbo, elbo_mfg = extract_and_remove_extra_info(df_elbo, "mean field gaussian")
    df_elbo, elbo_ns = extract_and_remove_extra_info(df_elbo, "neural spline")
    df_is, log_marginal_mfg = extract_and_remove_extra_info(df_is, "mean field gaussian")
    df_is, log_marginal_ns = extract_and_remove_extra_info(df_is, "neural spline")
    df_is, log_marginal_smc = extract_and_remove_extra_info(df_is, "SMC")

    if target_name == "BayesianLasso":
        df_elbo, elbo_standard = extract_and_remove_extra_info(df_elbo, "standard")
        elbo_additional_methods = "\n" + elbo_standard
        df_is, log_marginal_standard = extract_and_remove_extra_info(df_is, "standard")
        log_marginal_additional_methods = "\n" + log_marginal_standard

        df_elbo, elbo_ATAF = extract_and_remove_extra_info(df_elbo, "ATAF")
        elbo_additional_methods += "\n" + elbo_ATAF
        df_is, log_marginal_ATAF = extract_and_remove_extra_info(df_is, "ATAF")
        log_marginal_additional_methods += "\n" + log_marginal_ATAF
        
    else:
        elbo_additional_methods = ""
        log_marginal_additional_methods = ""

    if with_marginal_likelihood:
        fig, all_subplots = plt.subplots(2, figsize=(15, 20))
    else:
        fig, all_subplots = plt.subplots(figsize=(15, 10))

    # fig.suptitle(target_name, fontsize=20)

    LARGE_SIZE = 25

    if with_marginal_likelihood:
        current_sub_plot = all_subplots[0]
        # current_sub_plot.set_title("ELBO: \n" + elbo_mfg + "\n" + elbo_ns + elbo_additional_methods, fontsize = 20, loc="left")
        df_elbo.boxplot(ax = current_sub_plot, fontsize = LARGE_SIZE)
        current_sub_plot.set_ylabel("ELBO", fontsize=LARGE_SIZE)
        
        current_sub_plot = all_subplots[1]
        # current_sub_plot.set_title("log marginal likelihood: \n" + log_marginal_mfg + "\n" + log_marginal_ns + log_marginal_additional_methods + "\n" + log_marginal_smc, fontsize = 20, loc="left") 
        df_is.boxplot(ax = current_sub_plot, fontsize = LARGE_SIZE)
        current_sub_plot.set_ylabel("log marginal likelihood", fontsize=LARGE_SIZE)
    else:
        current_sub_plot = all_subplots
        # current_sub_plot.set_title("ELBO: \n" + elbo_mfg + "\n" + elbo_ns + elbo_additional_methods, fontsize = 20, loc="left")
        df_elbo.boxplot(ax = current_sub_plot, fontsize = LARGE_SIZE)
        current_sub_plot.set_ylabel("ELBO", fontsize=LARGE_SIZE)

    plt.axhline(true_log_marginal, c='r')

    if d is not None:
        assert(data is None)
        file_post_fix = str(d) + "d" 
    else:
        file_post_fix = data

    DIRECTORY = "../NormalizingFlows_private/latex/IOP/IOP_final/"
    
    OUTPUT_FILENAME = DIRECTORY + "results_ELBO_IS_" + target_name + "_" + file_post_fix + "_simple" + ".png"
    fig.tight_layout(pad = 3.0)
    
    if save:
        fig.savefig(OUTPUT_FILENAME)
    else:
        plt.show()

    return

def showPlot_simple_basic_four(save = False):
    
    d = 1000
    ALL_TARGET_NAMES = ["Funnel", "MultivariateStudentT", "MultivariateNormalMixture", "ConjugateLinearRegression"]

    all_positions = [(0,0), (0,1), (1,0), (1,1)]

    fig, all_subplots = plt.subplots(2, 2, figsize=(20, 15))

    for i, target_name in enumerate(ALL_TARGET_NAMES):
        nr_flows = 64
        
        _, _, (df_elbo, _) = show_table.showTable(target_name = target_name, show = "ELBO", nr_flows = nr_flows, d = d, full = True, filterInfValues = True, data = None)
        _, _, (df_is, true_log_marginal) = show_table.showTable(target_name = target_name, show = "IS", nr_flows = nr_flows, d = d, full = True, data = None)

        # print(df_elbo)
        # assert(False)

        # "proposed_withStudentT"
        porposed_method_variations = ["proposed",  "no_loft_proposed", "no_loft_proposed_withStudentT"]
        for method in porposed_method_variations:
            df_elbo, _ = extract_and_remove_extra_info(df_elbo, show_table.getTableNameForPlot("RealNVP_small", method))
            df_is, _ = extract_and_remove_extra_info(df_is, show_table.getTableNameForPlot("RealNVP_small", method))

        df_elbo, elbo_mfg = extract_and_remove_extra_info(df_elbo, "mean field gaussian")
        df_elbo, elbo_ns = extract_and_remove_extra_info(df_elbo, "neural spline")
        df_is, log_marginal_mfg = extract_and_remove_extra_info(df_is, "mean field gaussian")
        df_is, log_marginal_ns = extract_and_remove_extra_info(df_is, "neural spline")
        df_is, log_marginal_smc = extract_and_remove_extra_info(df_is, "SMC")

        if target_name == "BayesianLasso":
            df_elbo, elbo_standard = extract_and_remove_extra_info(df_elbo, "standard")
            elbo_additional_methods = "\n" + elbo_standard
            df_is, log_marginal_standard = extract_and_remove_extra_info(df_is, "standard")
            log_marginal_additional_methods = "\n" + log_marginal_standard

            df_elbo, elbo_ATAF = extract_and_remove_extra_info(df_elbo, "ATAF")
            elbo_additional_methods += "\n" + elbo_ATAF
            df_is, log_marginal_ATAF = extract_and_remove_extra_info(df_is, "ATAF")
            log_marginal_additional_methods += "\n" + log_marginal_ATAF
            
        else:
            elbo_additional_methods = ""
            log_marginal_additional_methods = ""

        
        
        LARGE_SIZE = 25

        pos = all_positions[i]
        current_sub_plot = all_subplots[pos[0],pos[1]]
        # current_sub_plot.set_title("ELBO: \n" + elbo_mfg + "\n" + elbo_ns + elbo_additional_methods, fontsize = 20, loc="left")
        df_elbo.boxplot(ax = current_sub_plot, fontsize = LARGE_SIZE)
        if target_name == "MultivariateStudentT":
            current_sub_plot.set_ylim(-1.5, 0.07)

        if pos[1] == 0:
            current_sub_plot.set_ylabel("ELBO", fontsize=LARGE_SIZE)

        # if pos[0] == 0:
        #    current_sub_plot.set_xticks([0, 1, 2, 3], labels = ["", "", "", ""])

        current_sub_plot.set_title(show_table.getTableNameForTargetModel(target_name), fontsize=20)

        current_sub_plot.axhline(true_log_marginal, c='r')

    
    DIRECTORY = "../NormalizingFlows_private/latex/IOP/IOP_final/"
    
    OUTPUT_FILENAME = DIRECTORY + "results_ELBO_" + "all_basic_four"  + "_simple" + ".png"
    fig.tight_layout(pad = 3.0)
    
    if save:
        fig.savefig(OUTPUT_FILENAME)
    else:
        plt.show()

    return

if __name__ == "__main__":
    
    # ALL_TARGET_NAMES = ["Funnel", "MultivariateStudentT", "MultivariateNormalMixture", "ConjugateLinearRegression"] #  "HorseshoePriorLogisticRegression"]
    
    # for target_name in ALL_TARGET_NAMES:
    #     showPlot(target_name, d = 1000)

    # showPlot("Funnel", d = 1000)

    # showPlot_simple("Funnel", d = 1000, with_marginal_likelihood = False)
    # showPlot_simple("MultivariateStudentT", d = 1000, with_marginal_likelihood = False)
    # showPlot_simple("ConjugateLinearRegression", d = 1000, with_marginal_likelihood = False)

    showPlot_simple_basic_four(save = True)

    # showPlot_simple("ConjugateLinearRegression", d = 1000, with_marginal_likelihood = False)

    # showPlot_simple("HorseshoePriorLogisticRegression", data = "colon", with_marginal_likelihood = False)

    # showPlot_simple("BayesianLasso", data = "multidrug_ABCB1")

    # showPlot("BayesianLasso", d = 1000)

    