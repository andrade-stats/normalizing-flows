import commons
import numpy
import matplotlib.pyplot as plt
import scipy.stats

import run_experiments

import pandas as pd
import seaborn as sns
import show_table

def showPlot(all_subplots, target_name, pos, include_GaussianOnly, confidence_interval_use_bootstrap, show_x_label, show_y_label):
    
    LARGE_SIZE = 25
    MEDIUM_SIZE = 20

    NR_FLOWS_LOW = 16
    NR_FLOWS_HIGH = 64

    if include_GaussianOnly:
        ALL_METHODS = [("GaussianOnly", None, None)]
    else:
        ALL_METHODS = []

    ALL_METHODS += [("RealNVP_small", "standard", NR_FLOWS_HIGH), ("RealNVP_small", "ATAF", NR_FLOWS_HIGH), ("RealNVP_small", "SymClip", NR_FLOWS_HIGH), ("RealNVP_small", "proposed", NR_FLOWS_HIGH), ("RealNVP_small", "proposed_withStudentT", NR_FLOWS_HIGH)]
    ALL_METHODS += [("RealNVP_small", "no_loft_proposed", NR_FLOWS_HIGH), ("RealNVP_small", "no_loft_proposed_withStudentT", NR_FLOWS_HIGH)]
    
    ALL_METHODS += [("RealNVP_small", "standard", NR_FLOWS_LOW), ("RealNVP_small", "ATAF", NR_FLOWS_LOW), ("RealNVP_small", "SymClip", NR_FLOWS_LOW), ("RealNVP_small", "proposed", NR_FLOWS_LOW), ("RealNVP_small", "proposed_withStudentT", NR_FLOWS_LOW)]
    
    D = 1000
        
    all_results_negative_ELBO = numpy.zeros(len(ALL_METHODS)) * numpy.nan
    all_results_Z_abs_diff =  numpy.zeros(len(ALL_METHODS)) * numpy.nan
    

    for i, (flow_type, method, nr_flows) in enumerate(ALL_METHODS):
        
        run_experiments.simple_init(target_name, D, flow_type, method, nr_flows)
       
        try:
            result = commons.loadStatistics("log_marginal_eval_result_best")
            nr_optimization_steps = commons.loadArray("optSteps").item()
            
            all_ELBO = result["ELBO"]
            all_IS = result["IS"]
            runtime = result["training_time_in_minutes"]
            true_log_marginal = result["true_log_marginal"]
            
            # in the previous version of the source code the true_log_marginal was wrongly calculated, therefore correct it here
            if target_name == "ConjugateLinearRegression":
                assert(D == 1000)
                true_log_marginal = -320.5926669536922  
                
            print("true_log_marginal = ", true_log_marginal)

        except FileNotFoundError:
            print(f"ERROR COULD NOT FIND FILE, spec = {flow_type} - {method} - {D}")
            assert(False)
            
        all_ELBO = commons.filterInfValues(all_ELBO)
        
        all_results_negative_ELBO[i] =  - numpy.nanmean(all_ELBO)

        assert(not numpy.any(numpy.isinf(all_IS)))

        all_results_abs_diff = numpy.abs(all_IS - true_log_marginal)
        all_results_Z_abs_diff[i] = numpy.nanmean(all_results_abs_diff)
    
    print("len(ALL_METHODS) = ", len(ALL_METHODS))
    print("all_results_negative_ELBO = ", all_results_negative_ELBO)
    print("all_results_Z_abs_diff = ", all_results_Z_abs_diff)

    all_results_ELBO = - all_results_negative_ELBO
    
    Y_LABEL = "error in marginal likelihood"
    X_LABEL = "ELBO"

    # OUTPUT_FILENAME = "all_plots_new/" + target_name + "_negELBO_vs_Zdiff_" + str(D) + "dim.png"
    
    # all_subplots.plot(all_results_Z_abs_diff, all_results_negative_ELBO, 'o', color = "blue")
    
    # plt.ylabel(Y_LABEL) # same as KL-divergence (up to a constant)
    # plt.xlabel(X_LABEL)
    # plt.tight_layout()

    # plt.plot(x, b + m * x, '--', color = "grey")
    # plt.savefig(OUTPUT_FILENAME)
    # plt.show()

    current_sub_plot = all_subplots[pos[0],pos[1]]
    # current_sub_plot = all_subplots[pos[0]]
    
    # df = pd.DataFrame({X_LABEL: all_results_Z_abs_diff, Y_LABEL: all_results_negative_ELBO})
    # my_sns_plot = sns.lmplot(x=X_LABEL, y=Y_LABEL, data=df, fit_reg=True, ci=95, n_boot=1000, scatter_kws={"color": "blue", "alpha" : 1.0, "s" : 2.0}, line_kws={"color": "blue",  "linewidth" : 0.5}) # "linestyle" : "--",
    # sns.regplot(ax = current_sub_plot, x=X_LABEL, y=Y_LABEL, data=df, fit_reg=True, ci=95, n_boot=1000) #  scatter_kws={"color": "blue", "alpha" : 1.0, "s" : 2.0}, line_kws={"color": "blue",  "linewidth" : 0.5})
    
    df = pd.DataFrame({X_LABEL: all_results_ELBO, Y_LABEL: all_results_Z_abs_diff})
    sns.regplot(ax = current_sub_plot, x=X_LABEL, y=Y_LABEL, data=df, fit_reg=True, ci=95, n_boot=1000)

    # if pos[0] == 0:
    #     current_sub_plot.set_xlabel("")
    
    # if pos[1] == 1:
    #     current_sub_plot.set_ylabel("")

    # x = numpy.linspace(numpy.min(all_results_Z_abs_diff), numpy.max(all_results_Z_abs_diff) + 0.1)

    # import statsmodels.api as sm
    # df = pd.DataFrame({"Zerror": all_results_Z_abs_diff, "negELBO": all_results_negative_ELBO})
    # model = sm.formula.ols(formula="negELBO ~ Zerror", data=df)
    # fitted_model = model.fit()
    # print(fitted_model.summary())
    # df_prediction = pd.DataFrame({"Zerror": x})
    # pred_ols = fitted_model.get_prediction(df_prediction)
    
    # alpha = 0.05
    # # mean_prediction = pred_ols.summary_frame(alpha)["mean"]
    # # iv_l = pred_ols.summary_frame()["obs_ci_lower"]
    # # iv_u = pred_ols.summary_frame()["obs_ci_upper"]
    # iv_l = pred_ols.summary_frame(alpha)["mean_ci_lower"]
    # iv_u = pred_ols.summary_frame(alpha)["mean_ci_upper"]

    # # plt.plot(x, mean_prediction, "b-.", label="OLS")
    # plt.plot(x, iv_u, "r--")
    # plt.plot(x, iv_l, "r--")
    # plt.savefig(OUTPUT_FILENAME)

    result = scipy.stats.pearsonr(all_results_Z_abs_diff, all_results_ELBO)

    pearson_corr = result.statistic
    pearson_corr_p_value = result.pvalue
    ROUND_DIGITS = 3
    print("person_corr = ", round(pearson_corr, ROUND_DIGITS))
    print(f"( person_corr_p_value = {pearson_corr_p_value})")
    pearson_ci = result.confidence_interval(confidence_level=0.95)
    print("pearson_ci (classic) = ", pearson_ci)
    
    if confidence_interval_use_bootstrap:
        pearson_ci = result.confidence_interval(confidence_level=0.95, method = scipy.stats.BootstrapMethod())
        print("pearson_ci (bootstrap) = ", pearson_ci)
    
    # 95% confidence level of pearson correlation 
    DIGITS = 2
    interval_str = f"[{round(pearson_ci.low, DIGITS)}, {round(pearson_ci.high, DIGITS)}]"
    
    corr_info_text = r"$\rho \in " + interval_str + "$"
    title_text = show_table.getTableNameForTargetModel(target_name)  + ", " + corr_info_text
    current_sub_plot.set_title(title_text, fontsize=LARGE_SIZE)
    
    if target_name == "ConjugateLinearRegression":
        # define manually for better layout
        current_sub_plot.set_xticks([-2250, -1750, -1250, -750, -250], labels=[-2250, -1750, -1250, -750, -250])
        
    if show_x_label:
        current_sub_plot.set_xlabel(X_LABEL, fontsize = MEDIUM_SIZE)
    else:
        current_sub_plot.set_xlabel("", fontsize = MEDIUM_SIZE)

    if show_y_label:
        current_sub_plot.set_ylabel(Y_LABEL, fontsize = MEDIUM_SIZE)
    else:
        current_sub_plot.set_ylabel("", fontsize = MEDIUM_SIZE)

    current_sub_plot.xaxis.set_tick_params(labelsize=MEDIUM_SIZE)
    current_sub_plot.yaxis.set_tick_params(labelsize=MEDIUM_SIZE)

    return

if __name__ == "__main__":

    fig, all_subplots = plt.subplots(2, 2, figsize=(20, 10))

    WITH_GAUSSIAN = True
    USE_BOOTSTRAP = True

    showPlot(all_subplots, target_name = "Funnel", pos = (0,0), include_GaussianOnly = WITH_GAUSSIAN, confidence_interval_use_bootstrap = USE_BOOTSTRAP, show_x_label = False, show_y_label = True)
    showPlot(all_subplots, target_name = "MultivariateStudentT", pos = (0,1), include_GaussianOnly = WITH_GAUSSIAN, confidence_interval_use_bootstrap = USE_BOOTSTRAP, show_x_label = False, show_y_label = False)
    showPlot(all_subplots, target_name = "MultivariateNormalMixture", pos = (1,0), include_GaussianOnly = WITH_GAUSSIAN, confidence_interval_use_bootstrap = USE_BOOTSTRAP, show_x_label = True, show_y_label = True)
    showPlot(all_subplots, target_name = "ConjugateLinearRegression", pos = (1,1), include_GaussianOnly = WITH_GAUSSIAN, confidence_interval_use_bootstrap = USE_BOOTSTRAP, show_x_label = True, show_y_label = False)

    OUTPUT_FILENAME = "../NormalizingFlows_private/latex/IOP/IOP_final/"  + "negELBO_vs_Zdiff_all" + ".png"
    fig.tight_layout(pad = 2.0)
    
    plt.show()
    # fig.savefig(OUTPUT_FILENAME)
