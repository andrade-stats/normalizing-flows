
import analysis
import numpy
import commons
import matplotlib.pyplot as plt
import run_experiments
import evaluation
import show_table

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





def show_plot_z_values(args, method, info, figure_number):
    SMALL_SIZE = 10
    MEDIUM_SIZE = 15
    BIGGER_SIZE = 20

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)

    assert(args.divergence == "reverse_kld_ws_debug")

    all_stat_z = commons.loadStatistics("layer_z_stats")
    
    fig, all_subplots = plt.subplots(3, figsize=(15, 20), num = figure_number)
    
    all_time_stats_true_losses = commons.loadStatistics("true_losses_stats")
    # all_time_stats_losses = commons.loadStatistics("losses_stats")
    
    MAX_TIME = all_time_stats_true_losses["quantiles"].shape[0]
    
    START_ID = 10000

    
    # legendInfo = "max (red), 99% (yellow), 75% (blue) quantiles"
    # fig.suptitle(f"{args.target}, d={args.D}, {args.cushion}, {args.lr_exp}, {divergence_info} " + "\n" + legendInfo)
    fig.suptitle(info, fontsize = 30, x = 0.1, horizontalalignment="left") 
    
    # analysis.showQuantiles(all_subplots[0], all_time_stats_true_losses["quantiles"], START_ID, MAX_TIME, quantiles = [0.75, 0.99, 1.0])
    # analysis.showQuantiles(all_subplots[1], all_time_stats_losses["quantiles"], START_ID, MAX_TIME, quantiles = [0.75, 0.99, 1.0])
    
    # y_limits = None
    # y_limits = (0, 10000)
    # y_limits = (0, 100)

    if args.target == "Funnel":
        all_y_limits = [(0, 200),(0, 200),(0, 200)]
    else:
        all_y_limits = [(0, 100),(0, 1000),(0, 10000)]

    
    STATISTIC = "max"
    # STATISTIC = "higher"
    # STATISTIC = "high"

    ALL_LAYERS = [4, 32, 64]

    STANDARD_FONT_SIZE = 15

    for i, layer_id in enumerate(ALL_LAYERS):
        current_sub_plot = all_subplots[i]
        current_sub_plot.set_title(f"Layer {layer_id}", fontsize = 25)
        analysis.showLayerValues(current_sub_plot, all_stat_z[STATISTIC], START_ID, MAX_TIME, layer_ids = [layer_id], y_limits = all_y_limits[i]) # show z value after "layer_id"-th layer
        
        current_sub_plot.set_ylabel("|z|", fontsize = STANDARD_FONT_SIZE)
        
        if i == len(ALL_LAYERS)-1:
            current_sub_plot.set_xlabel("iterations", fontsize = STANDARD_FONT_SIZE)

    # analysis.showLayerValues(all_subplots[1], all_stat_z[STATISTIC], START_ID, MAX_TIME, layer_ids = [32], y_limits = y_limits) # after 32th layer
    # analysis.showLayerValues(all_subplots[2], all_stat_z[STATISTIC], START_ID, MAX_TIME, layer_ids = [64], y_limits = y_limits) # after last layer


    # plt.tight_layout()
    plt.tight_layout(pad = 2.5)
    plt.savefig("all_plots_final/" + "large_value_analysis" + "_" + method + "_" + args.target + ".png")
    return


def show_plot(args, current_sub_plot, statistic, details = None):

    SMALL_SIZE = 10
    MEDIUM_SIZE = 15
    BIGGER_SIZE = 20

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)

    SHOW = "losses"

    if SHOW == "losses":
        if args.divergence == "reverse_kld_without_score" or args.divergence == "chi2_reduced" or args.divergence.startswith("reverse_kld_ws"):
            all_time_stats = commons.loadStatistics("true_losses_stats")
        else:
            all_time_stats = commons.loadStatistics("losses_stats")
    else:
        assert(SHOW == "samples")
        all_time_stats = commons.loadStatistics("samples_stats")

    MAX_TIME = all_time_stats["mean"].shape[0]
    
    # if args.annealing == "yes":
    #     START_ID = run_experiments.getAnnealingInterations(args.max_iterations)
    # else:
    #     assert(args.annealing == "no")
    
    START_ID = 10000
    # START_ID = 0
    
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
    
    # y_limits = (100, 200)
    y_limits = None

    if details == "upper":
        legendInfo = "max (red), 99% (yellow), 75% (blue) quantiles"
    else:
        legendInfo = statistic
    
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
        current_sub_plot.plot(x_values, all_time_stats[statistic][START_ID: MAX_TIME], color = "black")

        all_values = all_time_stats[statistic][START_ID: MAX_TIME]

        if y_limits is not None:
            current_sub_plot.set_ylim(y_limits)

        print(f"********** {statistic} ***********")
        print("min_value = ", numpy.min(all_values))
        print("min at iteration = ", START_ID + numpy.argmin(all_values))
        print("median = ", numpy.median(all_values))
    elif statistic == "max_param_stats":
        assert(False)
        max_param_stats = commons.loadStatistics("max_param_stats")
        fig, all_subplots = plt.subplots(2)
        x_values = numpy.arange(START_ID, MAX_TIME)
        all_subplots[0].plot(x_values, max_param_stats["max_abs_s"][START_ID: MAX_TIME], color = "red")
        all_subplots[1].plot(x_values, max_param_stats["max_abs_t"][START_ID: MAX_TIME], color = "red")
    else:
        assert(False)


    # if SHOW == "losses":
    #     plt.ylabel("negative ELBO") # same as KL-divergence (up to a constant)
    #     plt.xlabel("iterations")

    

    # plt.show()
    # plt.tight_layout()
    # plt.savefig("all_plots_new/" + commons.INFO_STR + "_show_" + statistic + "_" + details + ".png")


def show_loss_over_iterations_comparison():
    STANDARD_FONT_SIZE = 15

    target_name = "HorseshoePriorLogisticRegression"
    D = 1000

    flow_type = "RealNVP_small"
    nr_flows = 64

    ALL_METHODS = ["standard", "ATAF", "SymClip", "proposed_withStudentT"]
    
    fig, all_subplots = plt.subplots(4, figsize=(15, 20))

    for i, method in enumerate(ALL_METHODS):
        current_sub_plot = all_subplots[i]
        current_sub_plot.set_title(show_table.getTableName(flow_type, method), fontsize = 25) # loc="left") 
        
        target, _, args = run_experiments.simple_init(target_name, D, flow_type, method, nr_flows)
        show_plot(args, current_sub_plot, "mean", details = "")
        
        for tick in current_sub_plot.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize=STANDARD_FONT_SIZE) 
        
        for tick in current_sub_plot.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize=STANDARD_FONT_SIZE) 

        current_sub_plot.set_ylabel("negative ELBO", fontsize = STANDARD_FONT_SIZE)
        
        if i == len(ALL_METHODS)-1:
            current_sub_plot.set_xlabel("iterations", fontsize = STANDARD_FONT_SIZE)

        current_sub_plot.grid(axis='x', color='0.95')
        current_sub_plot.grid(axis='y', color='0.95')

    plt.tight_layout(pad = 3.0)

    plt.savefig("all_plots_final/" + "loss_comparison" + "_" + target_name + "_" + str(D) + ".png")
    return




def show_plot_z_values_all(target_name):

    D = 1000

    flow_type = "RealNVP_small"
    nr_flows = 64

    ALL_METHODS = ["standard", "ATAF", "SymClip", "proposed_withStudentT"]

    for figure_number, method in enumerate(ALL_METHODS):
        target, _, args = run_experiments.simple_init(target_name, D, flow_type, method, nr_flows)
        show_plot_z_values(args, method, show_table.getTableName(flow_type, method), figure_number)

    return


if __name__ == "__main__":

    # show_loss_over_iterations_comparison()

    # show_plot_z_values_all(target_name = "Funnel")
    show_plot_z_values_all(target_name = "HorseshoePriorLogisticRegression")