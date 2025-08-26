# Model recovery

For model recovery, we must have already done the simulations for the posterior predictive checks (see `figures/Figure5_plus2DF.ipynb`) and saved the simulated data in `src/02_PPCSimulation/Figures/*/Data`.
Having done this step, we can run the script `00_SDataCleaning.jl` to clean the simulated data and make, for each model, five datasets of size 60 participants (20 participants in each goal condition). 
These datasets will be saved in the subfolder `SData`.

Then, specifying the CSV file of the initial values of the parameters in `ParamLoadPath`, running the following code will optimize the model parameters to the simulated data (see `src/01_ModelFitting`):
```julia
DataLoadPath = "src/03_ModelRecovery/SData/"
ParamLoadPath = "src/03_ModelRecovery/Results/Params0.CSV"
Func_fit_shared_pop_CV_recovery(ParamLoadPath, DataLoadPath; 
                                rep_set = 1:5, ite_trig = 10,
                                NoGrad_maxeval = 500, Grad_maxeval = 10000,
                                Path_Save = "src/03_ModelRecovery/Results/")
```
Then, running `01_Reading_results.jl` will read and clean up the results, and running `02_Confusion_matrix.jl` will make the confusion matrices in Figure 4.
