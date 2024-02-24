
import argparse
import time

import numpy as np

import jax
import jax.numpy as jnp
import jax.random as random

import numpyro
import scipy.special

import sklearn.metrics
import sklearn.linear_model

import multiprocessing

from mcmc_inference import run_inference
from mcmc_inference import horsehoe_model_bernoulli_likelihood


def getAvgAndStd_str(results_allFolds, ROUND_DIGITS = 3):
    m = np.mean(results_allFolds)
    std = np.std(results_allFolds)
    return f"{round(m, ROUND_DIGITS)} ({round(std, ROUND_DIGITS)})"



def get_log_pred_density(model, posterior_samples, X_test, y_test):
    
    all_ll = numpyro.infer.log_likelihood(model, posterior_samples, X = X_test, y = y_test)

    nr_samples = all_ll["y"].shape[0]

    log_liklihood_each_test_point = scipy.special.logsumexp(all_ll["y"], axis = 0) - jnp.log(nr_samples)
    
    return log_liklihood_each_test_point


def get_mean_prediction(rng_key_predict, model, posterior_samples, X_test):
    predictive_eval_obj = numpyro.infer.Predictive(model, posterior_samples)
    predictions = predictive_eval_obj(rng_key_predict, X = X_test)
    return np.mean(predictions["y"], axis = 0)

def process_one_fold(args, foldId):
    all_data_one_fold = np.load("microarray_data_prepared/" + "microarray_fold_" + str(foldId) + ".npy", allow_pickle=True).item()        

    start = time.time()

    if args.method == "logistic_regression":
        baseline = sklearn.linear_model.LogisticRegression(random_state=0, fit_intercept = True).fit(all_data_one_fold["X_train"], all_data_one_fold["y_train"])
        most_freq_prediction = baseline.predict(all_data_one_fold["X_test"])
        log_prob_each_test_point = baseline.predict_log_proba(all_data_one_fold["X_test"])
        log_liklihood_each_test_point = log_prob_each_test_point[np.arange(all_data_one_fold["y_test"].shape[0]),  all_data_one_fold["y_test"]]
        prediction_scores_for_class_one = log_prob_each_test_point[:, 1]
    elif args.method == "hs":
        rng_key, rng_key_predict = random.split(random.PRNGKey(0))
        posterior_samples = run_inference(horsehoe_model_bernoulli_likelihood, args, rng_key, all_data_one_fold["X_train"], all_data_one_fold["y_train"])
    
        mean_prediction = get_mean_prediction(rng_key_predict, horsehoe_model_bernoulli_likelihood, posterior_samples, all_data_one_fold["X_test"])
        most_freq_prediction = np.where(mean_prediction >= 0.5, 1, 0) 
        prediction_scores_for_class_one = get_log_pred_density(horsehoe_model_bernoulli_likelihood, posterior_samples, X_test = all_data_one_fold["X_test"], y_test = np.ones_like(all_data_one_fold["y_test"]))
        log_liklihood_each_test_point = get_log_pred_density(horsehoe_model_bernoulli_likelihood, posterior_samples, X_test = all_data_one_fold["X_test"], y_test = all_data_one_fold["y_test"])
    else:
        assert(False)

    runtime = (time.time() - start) / 60.0

    accuracy = sklearn.metrics.accuracy_score(y_true = all_data_one_fold["y_test"], y_pred = most_freq_prediction)
    auc = sklearn.metrics.roc_auc_score(y_true = all_data_one_fold["y_test"], y_score = prediction_scores_for_class_one)
    ll = jnp.mean(log_liklihood_each_test_point)

    return runtime, accuracy, auc, ll

def main(args):

    NR_FOLDS = 10

    if args.device == "cpu":
        
        all_args = []
        for foldId in range(NR_FOLDS):
            all_args.append((args, foldId))

        with multiprocessing.Pool(NR_FOLDS) as p:
            all_results = p.starmap(process_one_fold, all_args)
            
        # convert to 2d numpy array
        r = np.array(list(map(np.array, all_results)))
        assert(r.shape[0] == NR_FOLDS and r.shape[1] == 4)
        all_runtimes = r[:, 0]
        all_accuracy = r[:, 1]
        all_auc  = r[:, 2]
        all_ll = r[:, 3]

    else:
        all_runtimes = np.zeros(NR_FOLDS)
        all_accuracy = np.zeros(NR_FOLDS)
        all_auc = np.zeros(NR_FOLDS)
        all_ll = np.zeros(NR_FOLDS)

        for foldId in range(NR_FOLDS):
            print("*************************")
            print("RUN FOLD ", foldId)
            all_runtimes[foldId], all_accuracy[foldId], all_auc[foldId], all_ll[foldId] = process_one_fold(args, foldId)
        

    print("--")
    print("MCMC runtime (in minutes) = " + getAvgAndStd_str(all_runtimes))
    print("--")
    print("accuracy = " + getAvgAndStd_str(all_accuracy))
    print("AUC = " + getAvgAndStd_str(all_auc))
    print("log-likelihood = " + getAvgAndStd_str(all_ll))



# run with
# python horseshoe_prior_analysis.py --num-samples=2000  --num-warmup=1000 --num-chains=2 --device=cpu
# python horseshoe_prior_analysis.py --num-samples=5  --num-warmup=2 --num-chains=2 --device=gpu
# CUDA_VISIBLE_DEVICES=3 python horseshoe_prior_analysis.py --num-samples=2000  --num-warmup=1000 --num-chains=2 --device=gpu
# CUDA_VISIBLE_DEVICES=3 python horseshoe_prior_analysis.py --num-samples=5  --num-warmup=2 --num-chains=2 --device=gpu

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Horseshoe regression")
    
    # parser.add_argument("-n", "--num-samples", default=2000, type=int)
    # parser.add_argument("--num-warmup", default=1000, type=int)
    parser.add_argument("-n", "--num-samples", default=5, type=int) 
    parser.add_argument("--num-warmup", default=2, type=int) 
    parser.add_argument("--method", default="logistic_regression", type=str)
    
    parser.add_argument("--num-chains", default=1, type=int)
    parser.add_argument("--device", default="cpu", type=str, help='use "cpu" or "gpu".')
    args = parser.parse_args()

    assert(args.num_samples >= 5 and args.num_warmup >= 2)
    assert(args.device == "cpu" or args.device == "gpu")

    numpyro.enable_x64()
    numpyro.set_platform(args.device)
        
    main(args)