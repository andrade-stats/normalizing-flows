
# Stable Training of Normalizing Flows for High-dimensional Variational Inference

Implementation of all normalizing flow experiments described in [Stabilizing Training of Affine Coupling Layers for High-dimensional Variational Inference](https://doi.org/10.1088/2632-2153/ad9a39), Machine Learning: Science and Technology, 2024.

The code here is mostly for reproducability of the experiments conducted in [Stabilizing Training of Affine Coupling Layers for High-dimensional Variational Inference](https://doi.org/10.1088/2632-2153/ad9a39).
If you are interested in variational inference with normalizing flows for your own research, consider the examples and documentation here:

[Tutorial for High-dimensional Variational Inference](https://github.com/andrade-stats/normalizing-flows-vi)

## Requirements

- Python >= 3.9
- PyTorch >= 2.0.1
- normflows >= 1.7.2

## Preparation

1. Create experiment environment using e.g. conda as follows
```bash
conda create -n newExps python=3.9
conda activate newExps
```

2. Install basic packages:
```bash
pip3 install -U numpy scikit-learn GPUtil normflows tqdm matplotlib pandas
```

<!--3. Install PyTorch for your environment (see official webpage of [PyTorch](https://pytorch.org)).
 ```bash
pip3 install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
``` -->

3. Create folders for output using
```bash
mkdir synthetic_data && mkdir all_results && mkdir all_trained_models && mkdir data && mkdir all_plots_final
```

## Usage (Basic Example Workflow)

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
- *target* = specify model (target distribution); possible choices are {**MultivariateStudentT**, **Funnel**, **MultivariateNormalMixture**, **ConjugateLinearRegression**, **HorseshoePriorLogisticRegression**}.
- *d* = specifies dimension of data used by **ConjugateLinearRegression** and **HorseshoePriorLogisticRegression**; possible choices are {**10**, **100**, **1000**}.
- *D* = specifies dimension of model used by **MultivariateStudentT**, **Funnel**, and **MultivariateNormalMixture**; possible choices are {**10**, **100**, **1000**}.
- *flow-type* = set **RealNVP_small** for normalizing flow and **GaussianOnly** for Gaussian mean field VI.
- *method* = specifies variant of RealNVP; possible  choices are {**standard**, **SymClip**, **ATAF**, **proposed** (proposed with Gaussian base distribution), **proposed_withStudentT** (proposed with student-t base distribution), **no_loft_proposed** (proposed with Gaussian base distribution, but no LOFT), **no_loft_proposed_withStudentT** (proposed with student-t base distribution, but no LOFT)}.

Example (1):  Training mean field VI on MultivariateStudentT with 1000 dimensions:
```bash
python run_experiments.py --target=MultivariateStudentT --D=1000 --flow-type=GaussianOnly
``` 

Example (2): Training proposed method (with student-t base distribution) on HorseshoePriorLogisticRegression with 1000 dimensional data (total model dimension is 4002):
```bash
python run_experiments.py --target=HorseshoePriorLogisticRegression --d=1000 --flow-type=RealNVP_small --method=proposed_withStudentT
``` 

Example (3): Training standard method on ConjugateLinearRegression with 1000 dimensional data (total model dimension is 1001):
```bash
python run_experiments.py --target=ConjugateLinearRegression --d=1000 --flow-type=RealNVP_small --method=standard
``` 


