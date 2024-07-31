import commons
import evaluation
import matplotlib.pyplot as plt

# import seaborn as sns

import show_table

def extract_and_remove_extra_info(df, remove_column_name):
    all_results = df[remove_column_name]
    df = df.drop(labels = remove_column_name, axis = 1)

    all_results = commons.filterInfValues(all_results)
    
    ROUND_DIGITS = 2

    return df, remove_column_name + " = " + evaluation.showAvgAndStd_str(all_results, ROUND_DIGITS = ROUND_DIGITS)


def showPlot(target_name, d = None, data = None):
    
    nr_flows = 64
    
    _, _, (df_elbo, _) = show_table.showTable(target_name = target_name, show = "ELBO", nr_flows = nr_flows, d = d, full = True, filterInfValues = True, data = data)
    _, _, (df_is, true_log_marginal) = show_table.showTable(target_name = target_name, show = "IS", nr_flows = nr_flows, d = d, full = True, data = data)

    df_elbo, elbo_mfg = extract_and_remove_extra_info(df_elbo, "mean field gaussian")
    df_elbo, elbo_ns = extract_and_remove_extra_info(df_elbo, "neural spline")
    # elbo_ns = ""
    # log_marginal_ns = ""

    df_is, log_marginal_mfg = extract_and_remove_extra_info(df_is, "mean field gaussian")
    df_is, log_marginal_ns = extract_and_remove_extra_info(df_is, "neural spline")
    df_is, log_marginal_smc = extract_and_remove_extra_info(df_is, "SMC")

    fig, all_subplots = plt.subplots(2, figsize=(15, 20))
    
    # fig.suptitle(target_name, fontsize=20)

    current_sub_plot = all_subplots[0]
    current_sub_plot.set_title("ELBO: \n" + elbo_mfg + "\n" + elbo_ns, fontsize = 20, loc="left")
    df_elbo.boxplot(ax = current_sub_plot, fontsize = 20)
    current_sub_plot.set_ylabel("ELBO", fontsize=20)

    current_sub_plot = all_subplots[1]
    current_sub_plot.set_title("log marginal likelihood: \n" + log_marginal_mfg + "\n" + log_marginal_ns + "\n" + log_marginal_smc, fontsize = 20, loc="left") 
    df_is.boxplot(ax = current_sub_plot, fontsize = 20)
    current_sub_plot.set_ylabel("log marginal likelihood", fontsize=20)

    plt.axhline(true_log_marginal, c='r')

    if d is not None:
        assert(data is None)
        file_post_fix = str(d) + "d" 
    else:
        file_post_fix = data

    OUTPUT_FILENAME = "all_plots_final/" + "results_ELBO_IS_" + target_name + "_" + file_post_fix + ".png"
    fig.tight_layout(pad = 3.0)
    fig.savefig(OUTPUT_FILENAME)

    return


if __name__ == "__main__":
    
    # ALL_TARGET_NAMES = ["Funnel", "MultivariateStudentT", "MultivariateNormalMixture", "ConjugateLinearRegression", "HorseshoePriorLogisticRegression"]
    
    # for target_name in ALL_TARGET_NAMES:
    #     showPlot(target_name, d = 1000)

    # showPlot("HorseshoePriorLogisticRegression", data = "colon")

    showPlot("BayesianLasso", data = "multidrug_ABCB1")

    # showPlot("BayesianLasso", d = 1000)

    