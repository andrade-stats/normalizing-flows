import torch
import core_adjusted
import commons



NUM_SAMPLES = 20000

def importance_sampling(nfm, nr_samples = NUM_SAMPLES):

    z,log_q = nfm.sample(num_samples = nr_samples)
    z,log_q, _, _ = core_adjusted.filter_illegal_values_from_samples(z, log_q)
    log_p = nfm.p.log_prob(z)

    log_mC = torch.logsumexp(log_p - log_q, dim = 0)
    log_is_estimate = log_mC - torch.log(torch.tensor(nr_samples))

    return log_is_estimate


def getRepeatedEstimates(nfm, type, nr_samples = NUM_SAMPLES):
    
    with torch.no_grad():
        all_estimates = torch.zeros(commons.REPETITIONS_FOR_MC_ERROR)
        for i in range(commons.REPETITIONS_FOR_MC_ERROR):
            if type == "ELBO":
                reverse_kld, _, _ = core_adjusted.reverse_kld_for_analysis(nfm, num_samples=nr_samples)
                all_estimates[i] = - torch.mean(reverse_kld)
            elif type == "chi2_simple":
                all_estimates[i] = core_adjusted.chi2_simple(nfm, num_samples = nr_samples)
            elif type == "chi2_advanced":
                all_estimates[i] = core_adjusted.chi2_advanced(p = nfm.p, q_aux = nfm, q_new = nfm, num_samples = nr_samples)
            elif type == "forward_kld":
                all_estimates[i] = core_adjusted.forward_kld(p = nfm.p, q_aux = nfm, q_new = nfm, num_samples = nr_samples, lambda_value = 1.0)
            elif type == "IS":
                all_estimates[i] = importance_sampling(nfm, nr_samples)
            else:
                assert(False)

    return all_estimates.numpy()
