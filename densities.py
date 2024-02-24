import torch

def get_log_prob_inv_gamma(sigmaSquare, alpha, beta):
    assert(alpha > 0.0 and beta > 0.0)
    assert(torch.all(sigmaSquare > 0.0))

    # if not torch.all(sigmaSquare > 0.0):
    #    print("sigmaSquare = ", sigmaSquare)
    # assert(torch.all(torch.logical_or(sigmaSquare > 0.0, torch.isnan(sigmaSquare))))

    log_prob = alpha * torch.log(beta) - torch.lgamma(alpha) \
        - (alpha + 1.0) * torch.log(sigmaSquare) - beta / sigmaSquare
    
    return log_prob
    

# CHECKED
def get_log_prob_multivariate_t(loc, invSigma, df, y):
    assert(loc.shape[0] == invSigma.shape[0])
    assert(y.shape == loc.shape)

    L = torch.linalg.cholesky(invSigma)
    zL = (y - loc) @ L
    
    logdet_invSigma = 2.0 * torch.sum(torch.log(torch.diag(L)))
    
    dim = loc.shape[0]
    
    log_prob = torch.lgamma((df + dim)/2.0) \
                        - torch.lgamma(df/2.0) - dim/2.0 * torch.log(df) \
                        - dim/2.0 * torch.log(torch.tensor(torch.pi)) + 0.5 * logdet_invSigma

    log_prob += - (df + dim)/2.0 * torch.log(1.0 + (1.0/df)*torch.sum(torch.square(zL)))

    return log_prob
