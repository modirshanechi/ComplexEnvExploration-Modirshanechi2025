# Model fitting

The model-fitting procedure can take a very long time, so we recommend running it on clusters with multiple CPUs. The code automatically uses distributed computing to parallelize optimization over cross-validation folds and models.

The code below performs the coarse-grained optimization, as described in the paper's SI using the Gradient-free optimization algorithm `LN_SBPLX` as implemented in `NLopt.jl`.
The optimization will be done from 3 different initializations specified in `GlobalConstants.jl`.
```julia
using Distributed

@everywhere using Pkg; 
@everywhere Pkg.activate(".")
@everywhere using IMRLExploration

Func_fit_shared_pop_CV(;
      model_set = 5:8, init_set = 1:3, ite_trig = 50, 
      Path_Save = "src/01_ModelFitting/Results/CoarseGrained/",
      NoGrad_maxeval = 100000, Grad_maxeval = 1)
```
The results will be saved in the `Results/CoarseGrained/` subfolder.
You can clean up these results by running `CoarseGrainedCleaning.jl` and perform the follow-up optimization for fine-tuning; this uses the gradient-based optimization algorithm `LBFGS` in `FitPopulations.jl`:
```julia
using Distributed

@everywhere using Pkg; 
@everywhere Pkg.activate(".")
@everywhere using IMRLExploration

Func_fit_shared_pop_CV(
      "src/01_ModelFitting/Figures/CoarseGrained/Params.CSV";
      model_set = 5:8, ite_trig = 5,
      Path_Save = "src/01_ModelFitting/Results/FineTuned/",
      NoGrad_maxeval = 1, Grad_maxeval = 10000)
```
The results will be saved in the `Results/FineTuned/` subfolder.
You can clean up these results by running `FineTunedCleaning.jl`. Then, using the notebook in `figures/Figure4.ipynb`, you can reproduce the results of the model comparison.

*NOTE:* As described in the paper's SI, for the CoarseGrained optimization, we had constrained the search to the case (i) with no action bias and (ii) with assuming β_MBe_2 and β_MFe_2 to be independent of the reward optimism. This choice accelerates the coarse-grained optimization.