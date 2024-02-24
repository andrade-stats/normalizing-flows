
import numpy as np
import scipy.special

def eval_variable_selection(beta_mean_approx, true_beta):

    # calculate r-precision
    true_positions = np.where(np.abs(true_beta) > 0)[0]
    nr_true_vars = true_positions.shape[0]

    # print("true_positions = ", true_positions)
    # print("nr_true_vars = ", nr_true_vars)
    assert(nr_true_vars == 3)
    
    predicted_true_vars_pos = np.argsort(-np.abs(beta_mean_approx))[0:nr_true_vars]

    # print("predicted_true_vars_pos = ")
    # print(predicted_true_vars_pos)
    
    true_pos_vec = np.zeros_like(beta_mean_approx)
    true_pos_vec[true_positions] = 1
    
    # print("true_pos_vec = ")
    # print(true_pos_vec)
    
    true_positivies = np.sum(true_pos_vec[predicted_true_vars_pos])

    r_precision = true_positivies / nr_true_vars

    return r_precision


def getX(rng, data_size, data_dim, rho):

    covariance_matrix = np.zeros((data_dim, data_dim))
    for i in range(data_dim):
        for j in range(data_dim):
            covariance_matrix[i, j] = rho**(abs(i - j))

    X = rng.multivariate_normal(mean = np.zeros(data_dim), cov = covariance_matrix, size = data_size)
    return X

def get_sample_data_linear_regression(rng, data_size, data_dim, rho, bias, sigma=3.0):
    assert(data_dim > 5)
    assert(rho == 0.5)
    assert(bias == 0.0)

    beta = np.zeros(data_dim)
    beta[0] = 3.0
    beta[1] = 1.5
    beta[4] = 2.0
    
    X = getX(rng, data_size, data_dim, rho)
    epsilon = rng.normal(loc = 0.0, scale = sigma, size = data_size)

    y = X @ beta + bias + epsilon
    gamma = np.where(beta != 0, 1, 0)

    return X, y, beta, bias, gamma

def  getIQR_length(samples):
    lower_quantile = np.quantile(samples, q = 0.25)
    upper_quantile = np.quantile(samples, q = 0.75)
    return upper_quantile - lower_quantile

def show_regression_result_statistics(beta_samples, intercept_samples, true_beta = None, true_intercept = None):

    INDEX_FOR_IQR = 0

    beta_mean = np.mean(beta_samples, axis = 0)
    iqr = getIQR_length(beta_samples[:, INDEX_FOR_IQR])

    
    print("beta_mean (first 10 dimensions) = ", beta_mean[0:10])
    print("true beta (first 10 dimensions) = ", true_beta[0:10])
    print("IQR = ", iqr)

    if intercept_samples is not None:
        intercept_mean = np.mean(intercept_samples, axis = 0)
        print("intercept_mean = ", intercept_mean)
    else:
        intercept_mean = np.nan

    avg_mse_to_truth_all = get_regression_avg_mse_to_truth(beta_samples, intercept_samples, true_beta, true_intercept)
    avg_mse_to_truth_beta = get_regression_avg_mse_to_truth(beta_samples, None, true_beta, None)
    
    print("avg_mse_to_truth_beta = ", avg_mse_to_truth_beta)

    return beta_mean, intercept_mean, iqr, avg_mse_to_truth_all, avg_mse_to_truth_beta


def getMSE(estimated_beta, true_beta):
    assert(np.array_equal(estimated_beta.shape, true_beta.shape))
    squared_diffs_beta = np.square(estimated_beta - true_beta)
    return np.sum(squared_diffs_beta) / (true_beta.shape[0])


def get_regression_avg_mse_to_truth(beta_samples, intercept_samples, true_beta = None, true_intercept = None):
    
    NR_SAMPLES = beta_samples.shape[0]

    # get average squared error
    if true_beta is not None:
        assert(beta_samples.shape[1] == true_beta.shape[0])
        squared_diffs_beta = np.square(beta_samples - true_beta)
            
        if intercept_samples is not None:
            assert(beta_samples.shape[0] == intercept_samples.shape[0])
            squared_diffs_intercept = np.square(intercept_samples - true_intercept)
            average_squared_error = (np.sum(squared_diffs_beta) + np.sum(squared_diffs_intercept)) / ((true_beta.shape[0] + 1) * NR_SAMPLES)
        else:
            average_squared_error = np.sum(squared_diffs_beta) / (true_beta.shape[0] * NR_SAMPLES)

    else:
        average_squared_error = np.nan
    
    return average_squared_error


# bias = 1.0  # leads to roughly 58~70% true labels
def get_sample_data_logistic_regression(rng, data_size, data_dim, rho = 0.1, bias = 0.0):
    assert(data_dim > 5)

    beta = np.zeros(data_dim)
    beta[0] = 3.0
    beta[1] = 1.5
    beta[4] = 2.0
    
    X = getX(rng, data_size, data_dim, rho)
    
    logits = X @ beta + bias
    true_probs = scipy.special.expit(logits)

    y = scipy.stats.bernoulli(true_probs).rvs(random_state = rng)

    gamma = np.where(beta != 0, 1, 0)
    return X, y, beta, bias, gamma


def generate_data(data_type, n):
    
    for d in [10, 100, 1000]:

        rng = np.random.default_rng(293309)
        
        NR_FOLDS = 20

        for foldNr in range(NR_FOLDS):
            if data_type == "synthetic_regression":
                rho = 0.5
                bias = 0.0
                X, y, true_beta, true_bias, _ = get_sample_data_linear_regression(rng, data_size = n, data_dim = d, rho = rho, bias = bias)
            else:
                rho = 0.1
                bias = 1.0
                assert(data_type == "synthetic_logistic")
                X, y, true_beta, true_bias, _  = get_sample_data_logistic_regression(rng, data_size = n, data_dim = d, rho = rho, bias = bias)

            whole_data = {}
            whole_data["X"] = X
            whole_data["y"] = y
            whole_data["true_beta"] = true_beta
            whole_data["true_bias"] = true_bias
            np.save("synthetic_data/" + data_type + f"_{n}n_{d}d_{rho}rho_{bias}bias_{foldNr}fold", whole_data)

            print("saved all prepared data")
        
def load_data(data_type, n, d, rho, bias, foldNr):

    whole_data = np.load("synthetic_data/" + data_type + f"_{n}n_{d}d_{rho}rho_{bias}bias_{foldNr}fold" + ".npy", allow_pickle=True).item()       
    X = whole_data["X"]
    y = whole_data["y"]
    true_beta = whole_data["true_beta"]
    true_bias = whole_data["true_bias"]

    return X, y, true_beta, true_bias


if __name__ == "__main__":

    generate_data("synthetic_logistic", n = 100)
    generate_data("synthetic_regression", n = 100)

    # X, y, true_beta, true_bias = load_data("synthetic_regression", n = 20, d = 10, rho = 0.5, bias = 0.0, foldNr = 0)
    # print("y = ", y)
    # print("X = ", X)