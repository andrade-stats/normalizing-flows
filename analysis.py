
import torch
import commons
import numpy
from tqdm import tqdm

START_ANALYSIS_ID = 14000
END_ANALYSIS_ID = 16000

SELECTED_LAYERS = ["first", "middle", "last"]

def getAllModuleIndices(nfm, layer_name):
    if layer_name == "first":
        return 0
    elif layer_name == "middle":
        start_search_id = int(len(nfm.flows) / 2)
    elif layer_name == "last":
        start_search_id = len(nfm.flows) - 1
    else:
        assert(False)
    
    for i in range(start_search_id, 0, -1):
        if nfm.flows[i].__class__.__name__ == "MaskedAffineFlow":
            return i
        
    return None



def getVec(tensor_list):
    all_vecs_flat = []
    for tensor_item in tensor_list:
        all_vecs_flat.append(torch.flatten(tensor_item).detach())
    return torch.hstack(all_vecs_flat)

PRESPECIFIED_QUANTILES = [0.0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0]

def getEmptyStatDic(max_iter):
    all_stat = {}
    all_stat["quantiles"] = numpy.zeros((max_iter, len(PRESPECIFIED_QUANTILES)))
    all_stat["mean"] = numpy.zeros(max_iter)
    all_stat["variance"] = numpy.zeros(max_iter)
    all_stat["nan_count"] = numpy.zeros(max_iter)
    return all_stat


def logAllStats(all_stats, it, vec, nan_sample_count):
    assert(len(vec.shape) == 1)

    if vec.shape[0] == 0:
        all_stats["quantiles"][it] = torch.nan
        all_stats["mean"][it] = torch.nan
        all_stats["variance"][it] = torch.nan
    else:
        all_stats["quantiles"][it] = getQuantileVec(vec)
        all_stats["mean"][it] = torch.nanmean(vec).item()
        all_stats["variance"][it] = torch.var(vec[~vec.isnan()])

    assert(torch.sum(torch.isnan(vec)).item() == 0)
    all_stats["nan_count"][it] = nan_sample_count
    return

def getQuantileVec(vec):
    QUANTILES = commons.moveToDevice(torch.tensor(PRESPECIFIED_QUANTILES, dtype=vec.dtype))
    all_quantiles = torch.nanquantile(vec, QUANTILES, interpolation='midpoint')
    return all_quantiles.cpu().numpy()

def getQuantileVec_numpy(vec):
    all_quantiles = numpy.nanquantile(vec, PRESPECIFIED_QUANTILES, interpolation='midpoint')
    return all_quantiles

def showQuantiles(subplot, all_stat_over_time, start_id, end_id, quantiles):
    
    x_values = numpy.arange(start_id, end_id)

    assert(len(quantiles) <= 3)
    colors = ["blue", "yellow", "red"]

    for i in range(len(quantiles)):
        quantile_index = PRESPECIFIED_QUANTILES.index(quantiles[i])
        subplot.plot(x_values, all_stat_over_time[start_id: end_id, quantile_index], color = colors[i])

    return

def showLayerValues(subplot, all_stat_over_time, start_id, end_id, layer_ids, y_limits = None, colors = ["red"], axis_font_size = 15):
    x_values = numpy.arange(start_id, end_id)

    assert(len(layer_ids) <= 3)
    # colors = ["blue", "yellow", "red"]
    colors = ["red"]

    for i in range(len(layer_ids)):
        subplot.plot(x_values, all_stat_over_time[start_id:end_id, layer_ids[i]], color = colors[i])
        if y_limits is not None:
            subplot.set_ylim(y_limits)

    # STANDARD_FONT_SIZE = 15

    # for tick in subplot.xaxis.get_major_ticks():
    #     tick.label.set_fontsize(fontsize=STANDARD_FONT_SIZE) 
    
    # for tick in subplot.yaxis.get_major_ticks():
    #     tick.label.set_fontsize(fontsize=STANDARD_FONT_SIZE) 

    subplot.xaxis.set_tick_params(labelsize=axis_font_size)
    subplot.yaxis.set_tick_params(labelsize=axis_font_size)

    # subplot.set_ylabel("negative ELBO", fontsize = STANDARD_FONT_SIZE)
    # current_sub_plot.set_xlabel("iterations", fontsize = STANDARD_FONT_SIZE)

    subplot.grid(axis='x', color='0.95')
    subplot.grid(axis='y', color='0.95')
    
    return

def saveLogDensityStatistics(all_time_densities, log_q, log_p):
    all_time_densities["log_q"].append(log_q.detach().cpu().numpy())
    all_time_densities["log_p"].append(log_p.detach().cpu().numpy())
    commons.saveStatistics(all_time_densities, "log_q_log_p")
    return

def recordMaxAbsValues(all_stats, it, all_flows_mixture):

    max_values = {}
    for key in all_stats.keys():
        max_values[key] = - torch.inf

    for name, param_group in all_flows_mixture.named_parameters():
        for key in all_stats.keys():
            if name.endswith(key):
                for param in param_group:
                    max_values[key] = max(max_values[key], torch.max(torch.abs(param)).item())
                    if (max_values[key] > 0.0) and (key == "weight" or key == "bias"):
                        print("name = ", name)
                        print("param_group = ", param_group)
                        print("max_values = ", max_values)
                        assert(False)

    for key in all_stats.keys():
        assert(max_values[key] > - torch.inf)
        all_stats[key][it] = max_values[key]

    return


def saveParams(all_time_params, nfm):
    for layer_name in SELECTED_LAYERS:
        layer_id = getAllModuleIndices(nfm.all_flows[0], layer_name)
        
        params = list(nfm.all_flows[0].flows[layer_id].parameters())
        params_one_vec = getVec(params).detach().cpu().numpy()
        all_time_params[layer_name].append(params_one_vec)
    
    commons.saveStatistics(all_time_params, "params")
    return

def getLogProbStats(all_log_densities):
    min_stat = numpy.min(all_log_densities, axis = 1)
    max_stat = numpy.max(all_log_densities, axis = 1)
    var_stat = numpy.var(all_log_densities, axis = 1)
    nan_stat = numpy.sum(numpy.isnan(all_log_densities), axis = 1)
    return nan_stat, min_stat, max_stat, var_stat


def saveAllsGrads(all_time_grads, nfm, loss):
    for layer_name in SELECTED_LAYERS:
        layer_id = getAllModuleIndices(nfm.all_flows[0], layer_name)
        
        all_grads = []
        for i in range(loss.shape[0]):
            grads = torch.autograd.grad(loss[i], list(nfm.all_flows[0].flows[layer_id].parameters()), retain_graph = True)
            grads_one_vec = getVec(grads)
            all_grads.append(grads_one_vec)
        all_grads = torch.stack(all_grads)
        all_time_grads[layer_name].append(all_grads.cpu().numpy())
    
    commons.saveStatistics(all_time_grads, "all_grads")
    return 


def get_params_with_require_grad(nfm):
    all_param = []

    for param in nfm.parameters():
        if param.requires_grad:
            all_param.append(param)

    return all_param


def saveAllGradsL2Norm(nfm, loss):

    all_grads_norm = numpy.zeros(loss.shape[0])
    all_invalid_values = numpy.zeros(loss.shape[0])
    
    for i in tqdm(range(loss.shape[0])):
        grads = torch.autograd.grad(loss[i], get_params_with_require_grad(nfm), retain_graph = True)
        grads_one_vec = getVec(grads)
        grads_one_vec = grads_one_vec.cpu().numpy()
        
        nr_invalid_values = numpy.sum(numpy.logical_or(numpy.isnan(grads_one_vec), numpy.isinf(grads_one_vec)))

        all_invalid_values[i] = (nr_invalid_values / grads_one_vec.shape[0]) * 100
        
        # print("all_invalid_values[i] = ", (all_invalid_values[i] / grads_one_vec.shape[0]) * 100)
        # assert(False)
        # all_grads_norm[i] = grads_one_vec[max_id]

        all_grads_norm[i] = numpy.nanmax(numpy.abs(grads_one_vec))

        # remove nan-values
        # grads_one_vec = grads_one_vec[~ numpy.isnan(grads_one_vec)]
        # all_grads_norm_without_nan[i] = numpy.linalg.norm(grads_one_vec)

    print("nr all_invalid_values = ", numpy.mean(all_invalid_values))

    return all_invalid_values, all_grads_norm


if __name__ == "__main__":
    commons.setGPU()
    torch.manual_seed(432432)
    
    vec = torch.tensor([3.2, -23.4, -80.2, -55.3, -3.0, 0.1, 34.9, 2.3, 93.2])
    r = getQuantileVec(vec)
    print("r = ", r)


