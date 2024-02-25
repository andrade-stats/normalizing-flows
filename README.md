
# Stable Training of Normalizing Flows for High-dimensional Variational Inference

Implementation of all normalizing flows experiments described in "Stable Training of Normalizing Flows for High-dimensional Variational Inference", 2024.


## Requirements

- Python >= 3.11.2
- PyTorch >= 2.0.1
- normflows >= 1.7.2

## Preparation

Create folders for output using
```bash
mkdir synthetic_data && mkdir all_results && mkdir all_trained_models && mkdir data && mkdir all_plots_final
```

## Usage (Basic Example Workflow)

Example using Friedman (n=100) dataset with 10% outliers.

-------------------------------------------
1. Prepare Data
-------------------------------------------
Create artificial data for (logistic) regression models
```bash
python syntheticData.py
```

Synthetic datasets are saved into folder "synthetic_data/."
(real datasets should be saved into folder "data/." for preparing the colon data set use "prepare_colon_data.py")

-------------------------------------------
2. Run Experiments
-------------------------------------------

Train and run sampling of proposed normalizing flow for the Horseshoe prior model:
```bash
python run_experiments.py --target=HorseshoePriorLogisticRegression --d=1000 --foldId=1 --flow-type=RealNVP_small --method=proposed_withStudentT
```

Results for analysis, ELBO and IS are saved into folder "all_results/."
The trained model is saved into folder "all_trained_models/."

-------------------------------------------
3. Show summary of all results
-------------------------------------------

```bash
python show_table.py
``` 

See comments in "show_table.py" for details.

## Details on command line arguments

The main arguments are as follows:
- target = specify model (target distribution), possible choices are {MultivariateStudentT, Funnel, MultivariateNormalMixture, ConjugateLinearRegression, HorseshoePriorLogisticRegression}
- flow-type = set *RealNVP_small* for normalizing flow and GaussianOnly for Gaussian mean field VI 

target specficies the model:
HorseshoePriorLogisticRegression 

Run student-t GP:
```bash
python runExperiments.py Friedman_n100 student focused None 0.1
``` 

## Details on source code

More details needed here !!

python run_experiments.py --target=MultivariateStudentT --D=1000 --flow-type=RealNVP_small --method=standard
python run_experiments.py --target=Funnel --D=1000 --flow-type=RealNVP_small --method=standard
python run_experiments.py --target=MultivariateNormalMixture --D=1000 --flow-type=RealNVP_small --method=standard
python run_experiments.py --target=ConjugateLinearRegression --d=1000 --flow-type=RealNVP_small --method=standard
