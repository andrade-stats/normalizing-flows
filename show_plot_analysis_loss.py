
import analysis
import numpy
import commons
import matplotlib.pyplot as plt
import run_experiments
import evaluation
import show_table

LEGEND_FONT_SIZE = 20
AXIS_FONT_SIZE = 15
MEDIUM_SIZE = 20

def loadParamsAndStatistics():
    all_time_params = commons.loadStatistics("params")
    for layer_name in analysis.SELECTED_LAYERS:
        all_time_params[layer_name] = numpy.stack(all_time_params[layer_name])
        
    # get quantiles
    all_time_param_quantiles = {}
    for layer_name in analysis.SELECTED_LAYERS:
        m = numpy.abs(all_time_params[layer_name])
        q = numpy.quantile(m, analysis.PRESPECIFIED_QUANTILES, interpolation='midpoint', axis = 1).transpose()
        all_time_param_quantiles[layer_name] = q
        
    return all_time_params, all_time_param_quantiles




def getVarStatistics_forGrads(all_time_grads):
    all_time_grad_var_quantiles = {}
    
    for layer_name in analysis.SELECTED_LAYERS:
        grad_var_quantiles_list = []
        for t in range(len(all_time_grads[layer_name])):
            all_grads_one_batch = all_time_grads[layer_name][t]
            all_grads_variance = numpy.var(all_grads_one_batch, axis = 0)
            grad_var_quantiles = analysis.getStatisticsVec_numpy(all_grads_variance)
            grad_var_quantiles_list.append(grad_var_quantiles)
            

        all_time_grad_var_quantiles[layer_name] = numpy.stack(grad_var_quantiles_list)
    
    return all_time_grad_var_quantiles




def show_plot_z_values_several_layers(target_name, method, figure_number = 0, ALL_LAYERS = [4, 32, 64]):
    assert(method in ["standard", "ATAF", "SymClip", "proposed_withStudentT"])
    
    D = 1000

    flow_type = "RealNVP_small"
    nr_flows = 64

    _, _, args = run_experiments.simple_init(target_name, D, flow_type, method, nr_flows)
    info = show_table.getTableName(flow_type, method)


    # LARGE_SIZE = 25

    # plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    # plt.rc('axes', titlesize=LARGE_SIZE)     # fontsize of the axes title
    # plt.rc('axes', labelsize=LARGE_SIZE)    # fontsize of the x and y labels
    # plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    # plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    # plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    # plt.rc('figure', titlesize=BIGGER_SIZE)

    assert(args.divergence == "reverse_kld_ws_debug")

    all_stat_z = commons.loadStatistics("layer_z_stats")
    
    if len(ALL_LAYERS) == 1:
        figure_size = (10, 8)
    else:
        figure_size = (15, 20)
    
    fig, all_subplots = plt.subplots(len(ALL_LAYERS), figsize=figure_size, num = figure_number)
    if len(ALL_LAYERS) == 1:
        all_subplots = [all_subplots]
    
    all_time_stats_true_losses = commons.loadStatistics("true_losses_stats")
    # all_time_stats_losses = commons.loadStatistics("losses_stats")
    
    MAX_TIME = all_time_stats_true_losses["quantiles"].shape[0]
    
    START_ID = 10000

    
    # legendInfo = "max (red), 99% (yellow), 75% (blue) quantiles"
    # fig.suptitle(f"{args.target}, d={args.D}, {args.cushion}, {args.lr_exp}, {divergence_info} " + "\n" + legendInfo)
    # fig.suptitle(info, fontsize = 30, x = 0.1, horizontalalignment="left") 
    
    # analysis.showQuantiles(all_subplots[0], all_time_stats_true_losses["quantiles"], START_ID, MAX_TIME, quantiles = [0.75, 0.99, 1.0])
    # analysis.showQuantiles(all_subplots[1], all_time_stats_losses["quantiles"], START_ID, MAX_TIME, quantiles = [0.75, 0.99, 1.0])
    
    # y_limits = None
    # y_limits = (0, 10000)
    # y_limits = (0, 100)

    if len(ALL_LAYERS) == 1:
        colors = ["red"]
        if args.target == "Funnel":
            all_y_limits = [(0, 200)]
        else:
            all_y_limits = [(0, 10000)]
    else:
        colors = ["red"]
        if args.target == "Funnel":
            all_y_limits = [(0, 200),(0, 200),(0, 200)]
        else:
            all_y_limits = [(0, 100),(0, 100),(0, 10000)]

    

    STATISTIC = "max"
    # STATISTIC = "higher"
    # STATISTIC = "high"

    # ALL_LAYERS = [4, 32, 64]
    # ALL_LAYERS = [4, 32, 64]

    for i, layer_id in enumerate(ALL_LAYERS):
        current_sub_plot = all_subplots[i]
        current_sub_plot.set_title(f"Layer {layer_id}", fontsize = MEDIUM_SIZE)
        analysis.showLayerValues(current_sub_plot, all_stat_z[STATISTIC], START_ID, MAX_TIME, layer_ids = [layer_id], y_limits = all_y_limits[i], colors = colors, axis_font_size = MEDIUM_SIZE) # show z value after "layer_id"-th layer
        
        current_sub_plot.set_ylabel("|z|", fontsize = MEDIUM_SIZE)
        
        if i == len(ALL_LAYERS)-1:
            current_sub_plot.set_xlabel("iterations", fontsize = MEDIUM_SIZE)

    # analysis.showLayerValues(all_subplots[1], all_stat_z[STATISTIC], START_ID, MAX_TIME, layer_ids = [32], y_limits = y_limits) # after 32th layer
    # analysis.showLayerValues(all_subplots[2], all_stat_z[STATISTIC], START_ID, MAX_TIME, layer_ids = [64], y_limits = y_limits) # after last layer


    # plt.tight_layout()
    plt.tight_layout(pad = 2.5)
    plt.savefig("../NormalizingFlows_private/latex/IOP/IOP_final/" + "large_value_analysis" + "_" + method + "_" + args.target + ".png")
    # plt.show()
    return


def show_plot(args, current_sub_plot, statistic, details = None, START_ID = 10000):

    # SMALL_SIZE = 10
    # MEDIUM_SIZE = 15
    # BIGGER_SIZE = 20

    # plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    # plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    # plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    # plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    # plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    # plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
    # plt.rc('figure', titlesize=BIGGER_SIZE)

    SHOW = "losses"

    if SHOW == "losses":
        if args.divergence == "reverse_kld_without_score" or args.divergence == "chi2_reduced" or args.divergence.startswith("reverse_kld_ws"):
            all_time_stats = commons.loadStatistics("true_losses_stats")
        else:
            assert(False)
            all_time_stats = commons.loadStatistics("losses_stats")
    else:
        assert(SHOW == "samples")
        all_time_stats = commons.loadStatistics("samples_stats")

    # MAX_TIME = START_ID + 100
    # MAX_TIME = 30000
    MAX_TIME = all_time_stats["mean"].shape[0]
    
    # if args.annealing == "yes":
    #     START_ID = run_experiments.getAnnealingInterations(args.max_iterations)
    # else:
    #     assert(args.annealing == "no")
    
    
    if args.annealing == "yes":
        annealing_info = "with annealing"
    else:
        annealing_info = "no annealing"

    if args.divergence == "reverse_kld":
        divergence_info = "with score term"
    else:
        divergence_info = "no score term"

    # fig.suptitle(statistic + f" (cushion-type = {args.cushion})")
    # fig.suptitle(f"{args.target}, median, learning rate = 10^-{args.lr_exp}, annealing = {args.annealing}, cushion-type = {args.cushion})")
    # fig.suptitle(f"{args.target}, median, learning rate = 10^-{args.lr_exp}, annealing = {args.annealing}, cushion-type = {args.cushion}), {divergence_info}")
    
    print("args.target = ", args.target)
    if args.target == "MultivariateStudentT":
        y_limits = (0, 10)
    elif args.target == "MultivariateNormalMixture":
        y_limits = None
    elif args.target == "Funnel":
        y_limits = (0, 10)
    elif args.target == "ConjugateLinearRegression":
        y_limits = (300, 500)
    else:
        y_limits = None
    # if details == "upper":
    #     legendInfo = "max (red), 99% (yellow), 75% (blue) quantiles"
    # else:
    #     legendInfo = statistic
    
    # fig.suptitle(f"{args.target}, d={args.D}, {args.cushion}, {args.lr_exp}, {divergence_info} " + "\n" + legendInfo)
    # fig.suptitle(f"{args.target}, d={args.D}, {args.cushion}, {args.lr_exp}, {annealing_info} " + "\n" + legendInfo)

    if statistic == "quantiles": 
        if details == "upper":
            quantiles = [0.75, 0.99, 1.0]
            # quantiles = [0.5, 0.75, 0.99]
        elif details == "median":
            quantiles = [0.5]
        else:
            assert(False)
        analysis.showQuantiles(current_sub_plot, all_time_stats[statistic], START_ID, MAX_TIME, quantiles)
    elif statistic == "nan_count" or statistic == "mean" or statistic == "variance":
        x_values = numpy.arange(START_ID, MAX_TIME)

        current_sub_plot.plot(x_values, all_time_stats[statistic][START_ID: MAX_TIME], label = details) # color = "black"

        all_values = all_time_stats[statistic][START_ID: MAX_TIME]

        if y_limits is not None:
            current_sub_plot.set_ylim(y_limits)

        print(f"********** {statistic} ***********")
        print("min_value = ", numpy.min(all_values))
        print("min at iteration = ", START_ID + numpy.argmin(all_values))
        print("median = ", numpy.median(all_values))
        
        NR_TAIL_VALUES = 30
        tail_values = all_values[all_values.shape[0] - NR_TAIL_VALUES : all_values.shape[0]]
        print(f"tail value mean = {numpy.mean(tail_values)}, std = {numpy.std(tail_values)}, median = {numpy.median(tail_values)}")
        
    elif statistic == "max_param_stats":
        assert(False)
        max_param_stats = commons.loadStatistics("max_param_stats")
        fig, all_subplots = plt.subplots(2)
        x_values = numpy.arange(START_ID, MAX_TIME)
        all_subplots[0].plot(x_values, max_param_stats["max_abs_s"][START_ID: MAX_TIME], color = "red")
        all_subplots[1].plot(x_values, max_param_stats["max_abs_t"][START_ID: MAX_TIME], color = "red")
    else:
        assert(False)


    return



def show_loss_over_iterations_comparison_all_four(save = False):
    # STANDARD_FONT_SIZE = 15
    
    ITERATION_SETTING = None
    # ITERATION_SETTING = "medium"

    
    # target_name = "MultivariateNormalMixture"
    # target_name = "MultivariateStudentT"
    # target_name = "Funnel"
    # target_name = "ConjugateLinearRegression"
    # ALL_METHODS = ["standard", "proposed_withStudentT"]
    ALL_METHODS = ["standard", "ATAF", "SymClip", "proposed_withStudentT"]
    # ALL_METHODS = ["standard", "proposed_withStudentT"]

    # ALL_METHODS = ["standard", "proposed"]
    # ALL_METHODS = ["standard", "no_loft_no_clamp_proposed", "proposed"]
    # ALL_METHODS = ["standard", "proposed_reverse", "proposed"]
    # ALL_METHODS = ["standard", "proposed"]

    ALL_TARGETS = ["Funnel", "MultivariateStudentT", "MultivariateNormalMixture", "ConjugateLinearRegression"]

    D = 1000

    flow_type = "RealNVP_small"
    nr_flows = 64

    fig, all_subplots = plt.subplots(4, figsize=(15, 20))

    # fig, plot = plt.subplots(1, 1, figsize=(15,10))
    # fig.suptitle(plot_name)

    # plot.set_title(show_table.getTableName(flow_type, method), fontsize = 25) # loc="left") 
    
    for target_name_id, target_name in enumerate(ALL_TARGETS):
        all_elbo = {}
        for _, method in enumerate(ALL_METHODS):
            print(f"********** {method} **************")
            _, _, args = run_experiments.simple_init(target_name, D, flow_type, method, nr_flows, only_args = True, iteration_setting = ITERATION_SETTING)
            
            plot = all_subplots[target_name_id]

            show_plot(args, plot, "mean", details = show_table.getTableName(flow_type, method), START_ID = 10000)
            
            result = commons.loadStatistics("log_marginal_eval_result_best")

            # remove -inf value from standard method (this is in favour for the standard method !)
            result["ELBO"] = result["ELBO"][~ numpy.isinf(result["ELBO"])]

            print("ELBO result = ", result["ELBO"])
            print("ELBO mean result = ", numpy.nanmean(result["ELBO"]))
            all_elbo[method] = numpy.nanmean(result["ELBO"])

            plot.set_title(show_table.getTableNameForTargetModel(target_name), fontsize = 25) # loc="left") 

            plot.xaxis.set_tick_params(labelsize=MEDIUM_SIZE)
            plot.yaxis.set_tick_params(labelsize=MEDIUM_SIZE)

            plot.set_ylabel("negative ELBO", fontsize = MEDIUM_SIZE)

            if target_name_id == 0:
                plot.legend(fontsize=LEGEND_FONT_SIZE)

            if target_name_id == len(ALL_TARGETS) - 1:
                plot.set_xlabel("iterations", fontsize = MEDIUM_SIZE)

            plot.grid(axis='x', color='0.95')
            plot.grid(axis='y', color='0.95')

        improvement_ratio_rel_standard = (all_elbo["proposed_withStudentT"] - all_elbo["standard"]) / numpy.abs(all_elbo["standard"])
        improvement_ratio_rel_proposed = (all_elbo["proposed_withStudentT"] - all_elbo["standard"]) / numpy.abs(all_elbo["proposed_withStudentT"])
        print("improvement_ratio (relative to standard method) % = ", improvement_ratio_rel_standard * 100)
        print("improvement_ratio (relative to proposed method) % = ", improvement_ratio_rel_proposed * 100)

    # plt.legend(fontsize=LEGEND_FONT_SIZE)
    plt.tight_layout(pad = 3.0)

    filename = "../NormalizingFlows_private/latex/IOP/IOP_final/" + "loss_comparison" + "_" + "all" + "_" + str(D) + ".png"
    
    if save:
        plt.savefig(filename)
        print("saved in ", filename)
    else:
        plt.show()
    # print("saved as " + filename)

    print(" ratio = ", (((-0.29) - (-1.07)) / 1.07) * 100)
    return



def show_loss_over_iterations_comparison_single_target(target_name):
    
    LAMBDA_FOR_BAYESIAN_LASSO = 100.0

    ITERATION_SETTING = None
    
    ALL_METHODS = ["standard", "ATAF", "SymClip", "proposed_withStudentT"]
    
    D = 1000

    flow_type = "RealNVP_small"
    nr_flows = 64

    fig, plot = plt.subplots(1, 1, figsize=(15,10))
    
    # plot.set_title(show_table.getTableName(flow_type, method), fontsize = 25) # loc="left") 
    
    for i, method in enumerate(ALL_METHODS):
        print(f"********** {method} **************")
        _, _, args = run_experiments.simple_init(target_name, D, flow_type, method, nr_flows, only_args = True, iteration_setting = ITERATION_SETTING, lambd = LAMBDA_FOR_BAYESIAN_LASSO)
        show_plot(args, plot, "mean", details = show_table.getTableName(flow_type, method), START_ID = 10000) 
       
        result = commons.loadStatistics("log_marginal_eval_result_best")
        print("ELBO result = ", result["ELBO"])
        print("ELBO mean result = ", numpy.mean(result["ELBO"]))


    plot.xaxis.set_tick_params(labelsize=MEDIUM_SIZE)
    plot.yaxis.set_tick_params(labelsize=MEDIUM_SIZE)

    
    if target_name == "BayesianLasso":
        y_limits = (200, 700)
        plot.set_ylim(y_limits)


    plot.set_ylabel("negative ELBO", fontsize = MEDIUM_SIZE)
    plot.set_xlabel("iterations", fontsize = MEDIUM_SIZE)

    plot.grid(axis='x', color='0.95')
    plot.grid(axis='y', color='0.95')

    plt.legend(fontsize=LEGEND_FONT_SIZE)
    plt.tight_layout(pad = 3.0)
    

    filename = "../NormalizingFlows_private/latex/IOP/IOP_final/" + "loss_comparison" + "_" + target_name + "_" + str(D) + ".png"
    plt.savefig(filename)
    # plt.show()
    print("saved as " + filename)
    return



def show_plot_z_values_comparison(target_name, save = False):

    D = 1000

    flow_type = "RealNVP_small"
    nr_flows = 64

    # ALL_METHODS = ["standard", "ATAF", "SymClip", "proposed_withStudentT"]
    
    # for figure_number, method in enumerate(ALL_METHODS):
    #     target, _, args = run_experiments.simple_init(target_name, D, flow_type, method, nr_flows)
    #     show_plot_z_values(args, method, show_table.getTableName(flow_type, method), figure_number)



    # comparison of last layer:
    # ALL_METHODS = ["standard", "proposed_withStudentT"]
    # ALL_METHODS = ["standard" , "ATAF", "SymClip", "proposed_withStudentT"]
    ALL_METHODS = ["standard" , "ATAF", "SymClip","proposed", "proposed_withStudentT"]
    
    # SMALL_SIZE = 10
    # MEDIUM_SIZE = 15
    # BIGGER_SIZE = 20
    # LARGE_SIZE = 25

    # plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    # plt.rc('axes', titlesize=LARGE_SIZE)     # fontsize of the axes title
    # plt.rc('axes', labelsize=LARGE_SIZE)    # fontsize of the x and y labels
    # plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    # plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    # plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    # plt.rc('figure', titlesize=BIGGER_SIZE)
    
    figure_size = (10, 15)
    
    fig, all_sub_plots = plt.subplots(len(ALL_METHODS), figsize=figure_size) 
    
    STATISTIC = "max"
    

    all_data = []

    for method in ALL_METHODS:
        _, _, args = run_experiments.simple_init(target_name, D, flow_type, method = method, nr_flows = nr_flows)
        all_stat_z_statistic = commons.loadStatistics("layer_z_stats")[STATISTIC]
        all_data.append(all_stat_z_statistic)

        all_time_stats_true_losses = commons.loadStatistics("true_losses_stats")
        MAX_TIME = all_time_stats_true_losses["quantiles"].shape[0]
    

    START_ID = 10000

    colors = ["red"]
    Y_LIMITS = (0, 10000)
    layer_id = 64
    
    for i in range(len(all_data)):
        current_sub_plot = all_sub_plots[i]
        method = ALL_METHODS[i]
        current_sub_plot.set_title(f"{show_table.getTableName(flow_type, method)} after layer {layer_id}", fontsize = MEDIUM_SIZE)
        analysis.showLayerValues(current_sub_plot, all_data[i], START_ID, MAX_TIME, layer_ids = [layer_id], y_limits = Y_LIMITS, colors = colors, axis_font_size = AXIS_FONT_SIZE) # show z value after "layer_id"-th layer
    
        # plt.yticks(fontsize=LARGE_SIZE)
        # current_sub_plot.tick_params(axis = "y", labelsize=MEDIUM_SIZE)
        current_sub_plot.set_ylabel("|z|", fontsize = MEDIUM_SIZE)
        
        if i == len(all_data) - 1:
            current_sub_plot.set_xlabel("iterations", fontsize = MEDIUM_SIZE)

    plt.tight_layout(pad = 2.5)
    if save:
        plt.savefig("../NormalizingFlows_private/latex/IOP/IOP_final/" + "large_value_analysis" + "_" + "comparison" + "_" + args.target + ".png")
    else:
        plt.show()

    return


if __name__ == "__main__":

    # print("normflows - Version = ", normflows.__version__)
    # print("PyTorch - Version = ", torch.__version__)
    # print("Numpy - Version = ", numpy.__version__)

    # show_loss_over_iterations_comparison_single_target("HorseshoePriorLogisticRegression")
    # show_loss_over_iterations_comparison_single_target("BayesianLasso")
    show_loss_over_iterations_comparison_all_four(save = True)

    # show_plot_z_values_comparison(target_name = "Funnel")
    
    # show_plot_z_values_comparison(target_name = "HorseshoePriorLogisticRegression", save = True)

    # show_plot_z_values_several_layers(target_name = "HorseshoePriorLogisticRegression", method = "standard")