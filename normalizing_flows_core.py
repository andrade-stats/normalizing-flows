
import numpy
import torch
import commons
import normflows as nf
from tqdm import tqdm
import core_adjusted
from normflows.flows.affine.coupling import AffineConstFlow
import analysis
import new_flows

def getNormalizingFlow(target, flow_type, number_of_flows, nr_samples_for_act_norm, initial_loc = None, cushion_type = None, cushion_t = None, nr_cushions = None, no_act_norm = None, realNVP_threshold = None, realNVP_variation = None, realNVP_spec = None, scaleShiftLayer = None, trainable_base = None, use_student_base = None):
    assert(no_act_norm is not None)

    if flow_type != "GaussianOnly":
        assert(cushion_type is not None)
    else:
        assert(trainable_base == "yes")
        assert(cushion_type is None)
        assert(nr_cushions is None)

    latent_size = target.D
    binary_mask = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(latent_size)])
    binary_mask = commons.moveToDevice(binary_mask)
    flows = []

    if (cushion_type is not None) and (cushion_type != "none"):
        assert(nr_cushions is not None)

    cushion_ids = []
    if (nr_cushions is not None) and (cushion_type != "none"):
        if nr_cushions == -1:
            cushion_ids.append(0)
        elif nr_cushions == 1:
            cushion_ids.append(number_of_flows - 1)
        elif nr_cushions == 12:
            cushion_ids = list(numpy.arange(start = 0, stop = number_of_flows, step = (number_of_flows // (nr_cushions-2))))
            cushion_ids.append(number_of_flows - 1)
            assert(len(cushion_ids) == nr_cushions)
        elif nr_cushions == number_of_flows - 2:
            cushion_ids = numpy.arange(start = 0, stop = number_of_flows - 2, step = 1)
            assert(cushion_ids.shape[0] == nr_cushions)
        elif nr_cushions == number_of_flows - 1:
            cushion_ids = numpy.arange(start = 0, stop = number_of_flows - 1, step = 1)
            assert(cushion_ids.shape[0] == nr_cushions)
        else:
            cushion_ids = numpy.arange(start = 0, stop = number_of_flows, step = number_of_flows // nr_cushions)
            assert(cushion_ids.shape[0] == nr_cushions)

    
    if flow_type != "GaussianOnly":
        
        for i in range(number_of_flows):
            if flow_type.startswith("RealNVP"):
                if flow_type == "RealNVP":
                    hidden_layer_size = 2 * latent_size
                elif flow_type == "RealNVP_small":
                    hidden_layer_size = 100
                else:
                    assert(False)

                scale_nn = nf.nets.MLP([latent_size, hidden_layer_size, latent_size], init_zeros=True)
                translation_nn = nf.nets.MLP([latent_size, hidden_layer_size, latent_size], init_zeros=True)

                if realNVP_spec is not None:
                    assert(realNVP_threshold is not None)
                    MaskedAffineFlow_var_constructor = getattr(new_flows, realNVP_spec)
                    if i % 2 == 0:
                        flows += [MaskedAffineFlow_var_constructor(binary_mask, translation_nn, scale_nn, realNVP_threshold)]
                    else:
                        flows += [MaskedAffineFlow_var_constructor(1 - binary_mask, translation_nn, scale_nn, realNVP_threshold)]
                elif realNVP_threshold is not None:
                    if i % 2 == 0:
                        flows += [new_flows.MaskedAffineFlowThresholded(binary_mask, translation_nn, scale_nn, realNVP_threshold, realNVP_variation)]
                    else:
                        flows += [new_flows.MaskedAffineFlowThresholded(1 - binary_mask, translation_nn, scale_nn, realNVP_threshold, realNVP_variation)]
                else:
                    if i % 2 == 0:
                        flows += [nf.flows.MaskedAffineFlow(binary_mask, translation_nn, scale_nn)]
                    else:
                        flows += [nf.flows.MaskedAffineFlow(1 - binary_mask, translation_nn, scale_nn)]
                
                if not no_act_norm:
                    assert(False)
                    flows += [ActNorm(latent_size)]
                
                if i in cushion_ids:
                    if cushion_type == "LOFT":
                        flows += [new_flows.TrainableLOFTLayer(latent_size, cushion_t, train_t = False)]
                    elif cushion_type == "TrainableLOFT":
                        assert(False)
                        flows += [TrainableLOFTLayer(latent_size, cushion_t, train_t = True)]
                    elif cushion_type == "ScaledTanh":
                        assert(False)
                        flows += [ScaledTanhLayer(latent_size, cushion_t)]
                    elif cushion_type == "Truncation":
                        assert(False)
                        flows += [TruncationLayer(latent_size, threshold = cushion_t)]
                    else:
                        assert(cushion_type == "none")
                    print("Added cushion to layer nr ", i)
                
            elif flow_type == "MAF":
                flows += [nf.flows.MaskedAffineAutoregressive(latent_size,2*latent_size)]

            elif flow_type == "Planar":
                flows += [nf.flows.Planar((latent_size,), act = "leaky_relu")]
            
            # elif flow_type == "Autoregressive":
            #     flows += [autoregressive.MaskedAffineAutoregressive(features=target.D, hidden_features=30, num_blocks=5)]
            #     # flows += [nf.flows.CircularAutoregressiveRationalQuadraticSpline(target.D, 1, 512, [1], num_bins=10, tail_bound=torch.tensor([5., torch.pi]), permute_mask=True)]  # not recommended because requires too much memory
            else:
                assert(False)
    else:
        assert(number_of_flows is None)
    

    if scaleShiftLayer is not None:
        if scaleShiftLayer == "ssL":
            assert(flow_type != "GaussianOnly")
            flows += [AffineConstFlow(latent_size)]
        else:
            assert(False)
    

    if target.get_pos_contraint_ids() is not None:
        flows += [new_flows.PositiveConstraintLayer(target.get_pos_contraint_ids(), target.D)]
    
    assert(trainable_base == "yes" or trainable_base == "no")
    if use_student_base == "yes":
        q0 = core_adjusted.DiagStudentT(target.D, initial_loc, trainable = (trainable_base == "yes")) # default is trainable = True
    else:
        q0 = core_adjusted.DiagGaussian(target.D, initial_loc, trainable = (trainable_base == "yes")) # default is trainable = True

    target = commons.moveToDevice(target)

    # Construct flow model
    nfm = nf.NormalizingFlow(q0=q0, flows=flows, p=target)
    nfm = commons.moveToDevice(nfm)

    if not no_act_norm:
        # Initialize ActNorm
        print("*** Sample for ActNorm ***")
        _, _ = nfm.sample(num_samples=nr_samples_for_act_norm)

    return nfm


    
class FlowsMixture(torch.nn.Module):

    def __init__(self, target, K, flow_type, number_of_flows, nr_samples_for_act_norm, learn_mixture_weights = False, initial_loc_spec = None, cushion_type = None, cushion_t = None ,nr_cushions = None, no_act_norm = None, realNVP_threshold = None, realNVP_variation = None, realNVP_spec = None, redGradVarEst = None, scaleShiftLayer = None, trainable_base = None, use_student_base = None):
        super().__init__()
        self.D = target.D

        self.redGradVarEst = redGradVarEst
        self.cushion_t = cushion_t

        self.p = commons.moveToDevice(target)
        self.number_of_flows = number_of_flows

        uniform_dist = torch.ones(K) / K
        self.mixture_weights = commons.moveToDevice(uniform_dist)

        if learn_mixture_weights:
            self.mixture_weights = torch.nn.Parameter(self.mixture_weights)
        
        all_flows = []
        for k in range(K):

            if initial_loc_spec == "random":
                initial_loc_for_flow = torch.rand(size = (1,target.D)) * 20.0 - 10.0
            elif initial_loc_spec == "random_small":
                initial_loc_for_flow = torch.rand(size = (1,target.D)) * 2.0 - 1.0
            elif initial_loc_spec == "zeros":
                assert(K == 1)
                initial_loc_for_flow = torch.zeros(size = (1,target.D))
            else:
                assert(False)
            
            initial_loc_for_flow = commons.moveToDevice(initial_loc_for_flow)

            nfm = getNormalizingFlow(target, flow_type = flow_type, number_of_flows = number_of_flows, nr_samples_for_act_norm = nr_samples_for_act_norm, initial_loc = initial_loc_for_flow, cushion_type = cushion_type, cushion_t = cushion_t, nr_cushions = nr_cushions, no_act_norm = no_act_norm, realNVP_threshold = realNVP_threshold, realNVP_variation = realNVP_variation, realNVP_spec = realNVP_spec, scaleShiftLayer = scaleShiftLayer, trainable_base = trainable_base, use_student_base = use_student_base)
            all_flows.append(nfm)

        self.all_flows = torch.nn.ModuleList(all_flows)  # register all flows (this makes all parameters learnable)

        return
    
    # checked
    def sample(self, num_samples):

        multinomial = torch.distributions.multinomial.Multinomial(num_samples, probs= self.mixture_weights)
        freq_each_comp = multinomial.sample()

        assert(len(self.all_flows) == self.mixture_weights.shape[0])

        log_comp_weights = torch.log(self.mixture_weights)

        all_z = []
        all_log_q = []
        for k in range(len(self.all_flows)):
            z, log_q_comp_k = self.all_flows[k].sample(int(freq_each_comp[k]))
            all_z.append(z)

            # assert(torch.sum(torch.isnan(z)) == 0)

            log_q_each_comp = []
            log_q_each_comp.append(log_q_comp_k + log_comp_weights[k])
            for j in range(len(self.all_flows)):
                if j != k:
                    log_q_comp_j = self.all_flows[j].log_prob(z)
                    log_q_each_comp.append(log_q_comp_j + log_comp_weights[j])
            
            log_q_each_comp = torch.stack(log_q_each_comp)
            log_q = torch.logsumexp(log_q_each_comp, dim = 0)
            all_log_q.append(log_q)

        all_z = torch.vstack(all_z)
        all_log_q = torch.hstack(all_log_q)

        assert(all_z.shape[0] == num_samples and all_z.shape[1] == self.D)
        assert(all_log_q.shape[0] == num_samples)

        # set out-of-range samples to NaN
        if self.redGradVarEst == "v":
            assert(self.cushion_t >= 10.0)
            z_abs = torch.abs(all_z)
            all_z[z_abs > self.cushion_t] = torch.nan
        else:
            assert(self.redGradVarEst is None)

        return all_z, all_log_q
    
    # checked
    def log_prob(self, z):

        log_comp_weights = torch.log(self.mixture_weights)

        log_q_each_comp = []

        for k in range(len(self.all_flows)):
            log_q_comp_k = self.all_flows[k].log_prob(z)
            log_q_each_comp.append(log_q_comp_k + log_comp_weights[k]) 
            
        log_q_each_comp = torch.stack(log_q_each_comp)
        all_log_q = torch.logsumexp(log_q_each_comp, dim = 0)
        
        assert(all_log_q.shape[0] == z.shape[0])

        return all_log_q
    



# adapted from normflows package
class ActNorm(AffineConstFlow):
    """
    An AffineConstFlow but with a data-dependent initialization,
    where on the very first batch we clever initialize the s,t so that the output
    is unit gaussian. As described in Glow paper.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dep_init_done_cpu = torch.tensor(0.0)
        self.register_buffer("data_dep_init_done", self.data_dep_init_done_cpu)
        
        self.s.requires_grad = False
        self.t.requires_grad = False

    def reset(self):
        self.data_dep_init_done_cpu = torch.tensor(0.0)
        self.register_buffer("data_dep_init_done", self.data_dep_init_done_cpu)

    def forward(self, z):    
        # first batch is used for initialization, c.f. batchnorm
        if not self.data_dep_init_done > 0.0:
            assert self.s is not None and self.t is not None
            s_init = -torch.log(z.std(dim=self.batch_dims, keepdim=True) + 1e-6)
            self.s.data = s_init.data
            self.t.data = (
                -z.mean(dim=self.batch_dims, keepdim=True) * torch.exp(self.s)
            ).data

            assert((not torch.any(torch.isnan(self.s.data))) and (not torch.any(torch.isinf(self.s.data))))
            assert((not torch.any(torch.isnan(self.t.data))) and (not torch.any(torch.isinf(self.t.data))))
            self.data_dep_init_done = torch.tensor(1.0)

        new_value, log_det = super().forward(z)
        return new_value, log_det

    def inverse(self, z):
        assert(self.data_dep_init_done == 1.0)
        return super().inverse(z)
    





def train(nfm, max_iter, anneal_iter = None, show_iter = None, learning_rate = None, l2_strength = 0.0, l2_clip_norm = None, divergence = None, target_p = None, q_aux = None, num_samples = None, lambda_value = 1.0, record_stats = True, annealing = True, redGradVarEst = None, cushion_t = None, opt = None):

    print("max_iter = ", max_iter)
    print("anneal_iter = ", anneal_iter)
    print("show_iter = ", show_iter)
    
    print("annealing = ", annealing)
    print("divergence_for_optimization = ", divergence)
    

    all_time_densities = {}
    all_time_densities["log_p"] = []
    all_time_densities["log_q"] = []
    
    all_time_losses_stats = analysis.getEmptyStatDic(max_iter)
    all_time_true_losses_stats = analysis.getEmptyStatDic(max_iter)
    all_time_samples_stats = analysis.getEmptyStatDic(max_iter)

    RECORD_TYPES = ["loc", "log_scale", "weight", "bias"]
    all_time_param_stats = {}
    for rt in RECORD_TYPES:
        all_time_param_stats[rt] = numpy.zeros(max_iter)
    
    if divergence == "reverse_kld_ws_debug":
        all_stat_z = {}
        all_stat_z["median"] = numpy.zeros((max_iter, nfm.number_of_flows + 1))
        all_stat_z["high"] = numpy.zeros((max_iter, nfm.number_of_flows + 1))
        all_stat_z["higher"] = numpy.zeros((max_iter, nfm.number_of_flows + 1))
        all_stat_z["max"] = numpy.zeros((max_iter, nfm.number_of_flows + 1))

    assert(l2_strength == 0.0)    
    
    if opt == "Adam":
        optimizer = torch.optim.Adam(nfm.parameters(), lr=learning_rate, weight_decay=l2_strength)
    elif opt == "ASGD":
        optimizer = torch.optim.ASGD(nfm.parameters(), lr=learning_rate, weight_decay=l2_strength)
    elif opt == "SGD":
        optimizer = torch.optim.SGD(nfm.parameters(), lr=learning_rate, weight_decay=l2_strength)
    elif opt == "RAdam":
        optimizer = torch.optim.RAdam(nfm.parameters(), lr=learning_rate, weight_decay=l2_strength)
    else:
        assert(False)

    nr_optimization_steps = 0

    current_best_true_loss = torch.inf

    for it in tqdm(range(max_iter)):

        optimizer.zero_grad()
        
        if annealing:
            beta_annealing = numpy.min([1., 0.01 + it / anneal_iter]) #  min(1, 0.01 + t/10000) is suggested in "Variational Inference with Normalizing Flows"
        else:
            beta_annealing = 1.0
        
        if divergence == "reverse_kld":
            loss, log_q, log_p = core_adjusted.reverse_kld_for_analysis(nfm, num_samples, beta=beta_annealing)
            true_loss = loss
        elif divergence == "reverse_kld_without_score":
            assert(redGradVarEst is None)
            loss, true_loss, log_q, log_p, z_prime = core_adjusted.reverse_kld_without_score_for_analysis(nfm, num_samples, beta=beta_annealing)
        elif divergence == "reverse_kld_ws_debug":
            loss, true_loss,  z_stats_median, z_stats_high, z_stats_higher, z_stats_max = core_adjusted.reverse_kld_without_score_debug(nfm, num_samples, redGradVarEst = redGradVarEst, cushion_t = cushion_t,  beta=beta_annealing)
        elif divergence == "forward_kld":
            assert(False)
            loss = core_adjusted.forward_kld(target_p, q_aux, nfm, num_samples, lambda_value)
        elif divergence == "chi2":
            _, loss = core_adjusted.chi2(nfm, num_samples, beta=beta_annealing)
            true_loss = loss
        elif divergence == "chi2_reduced":
            loss, true_loss = core_adjusted.chi2(nfm, num_samples, beta=beta_annealing)
        else:
            assert(False)
        

        num_samples_after_filtering = loss.shape[0]
        assert(num_samples_after_filtering <= num_samples)

        if record_stats:
            analysis.logAllStats(all_time_losses_stats, it, loss.detach(), num_samples - num_samples_after_filtering)
            commons.saveStatistics(all_time_losses_stats, "losses_stats")

            # analysis.recordMaxAbsValues(all_time_param_stats, it, nfm)
            # commons.saveStatistics(all_time_param_stats, "max_param_stats")

            if divergence == "reverse_kld_without_score" or divergence == "chi2_reduced" or divergence.startswith("reverse_kld_ws"):
                analysis.logAllStats(all_time_true_losses_stats, it, true_loss.detach(),  num_samples - num_samples_after_filtering)
                commons.saveStatistics(all_time_true_losses_stats, "true_losses_stats")

            if divergence == "reverse_kld_without_score":
                assert(z_prime.shape[0] == num_samples_after_filtering)
                analysis.logAllStats(all_time_samples_stats, it, torch.abs(torch.flatten(z_prime)),  num_samples - num_samples_after_filtering)
                commons.saveStatistics(all_time_samples_stats, "samples_stats")

            if divergence == "reverse_kld_ws_debug":
                assert(z_stats_median.shape[0] == nfm.number_of_flows + 1)
                all_stat_z["median"][it] = z_stats_median
                all_stat_z["high"][it] = z_stats_high
                all_stat_z["higher"][it] = z_stats_higher
                all_stat_z["max"][it] = z_stats_max
                
        # all_time_losses[it] = loss.detach().cpu().numpy()
        # commons.saveStatistics(all_time_losses, "all_losses")

        # all_time_true_losses[it] = true_loss.detach().cpu().numpy()
        # commons.saveStatistics(all_time_true_losses, "all_true_losses")

        # analysis.saveLogDensityStatistics(all_time_densities, log_q, log_p)

        # all_time_samples.append(z_prime.cpu().numpy())
        # commons.saveStatistics(all_time_samples, "all_samples")
        # analysis.saveParams(all_time_params, nfm)

        # *** too computationally and memory expensive ***
        # if it >= analysis.START_ANALYSIS_ID and it <= analysis.END_ANALYSIS_ID:
        #    print("SAVING GRADIENTS")
        #    analysis.saveAllsGrads(all_time_grads, nfm, loss)
        
        # print("num_samples_after_filtering = ", num_samples_after_filtering)

        loss = torch.mean(loss)
        true_loss = torch.mean(true_loss)

        if ~(torch.isnan(loss) | torch.isinf(loss)):

            if (it > (max_iter / 2)) and ~(torch.isnan(true_loss) | torch.isinf(true_loss)) and (true_loss < current_best_true_loss):
                print("update best model")
                torch.save(nfm.state_dict(), commons.get_model_filename_best())
                current_best_true_loss = true_loss.detach().to('cpu')

            loss.backward()
            if l2_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(nfm.parameters(), l2_clip_norm, error_if_nonfinite = True)
            
            invalid_grad = False
            for name, param in nfm.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        print("invalid gradients found ! SKIP")
                        invalid_grad = True
                        break
            
            if not invalid_grad:
                optimizer.step()
                nr_optimization_steps += 1

        if it % show_iter == 0:
            print(f"loss = {true_loss.to('cpu')} (without score = {loss.to('cpu')})")

        
    
    if record_stats:
        commons.saveStatistics(numpy.asarray([nr_optimization_steps]), "optSteps")

    if divergence == "reverse_kld_ws_debug":
        commons.saveStatistics(all_stat_z, "layer_z_stats")
        print("** SAVED layer_z_stats **")
    
    return nr_optimization_steps, current_best_true_loss
