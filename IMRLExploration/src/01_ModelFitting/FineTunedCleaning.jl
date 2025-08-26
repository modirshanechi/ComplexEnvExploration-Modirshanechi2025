using PyPlot
using IMRLExploration
using JLD2
using CSV
using Random
using Statistics
using DataFrames
using LogExpFunctions
using Bootstrap

using FitPopulations
import FitPopulations: parameters, logp, sample, initialize!
import FitPopulations: gradient_logp
import FitPopulations: hessian_logp
import FitPopulations: maximize_logp
import FitPopulations: PopulationModel

using ComponentArrays

PyPlot.svg(true)
rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["svg.fonttype"] = "none"
rcParams["pdf.fonttype"] = 42

Path_Load = "src/01_ModelFitting/Results/FineTuned/"
Path_Save = "src/01_ModelFitting/Figures/FineTuned/"
# ------------------------------------------------------------------------------
# Loading data
# ------------------------------------------------------------------------------
Outliers, Long_Subject, Quit_Subject, Data, Goal_type_Set, Sub_Num =
        Read_processed_data(Plotting = false);
Epi_len = sum(Func_EpiLenghts_all(Data)[1],dims=2)[:]
Data_ns = Str_Input2Agents.(Data);

# ------------------------------------------------------------------------------
# initializing the params
# ------------------------------------------------------------------------------
Param = Str_Param(; total_leak=true)
Rs    = [1.,1.,1.]
ws    = [1.,1.,1.]
A = Str_Agent_Policy(Param, Rs, NIS_object(ws))
p = parameters(A)
p_names = [string(k) for k = keys(param2η(p))]
m_names = [string(m) for m = keys(model_settings)]

# ------------------------------------------------------------------------------
# loading values
# ------------------------------------------------------------------------------
nfolds = 3; CV_inds = [Func_GoalCV(i; nfolds = nfolds) for i = 1:nfolds]
nfold_set = 1:3; model_set = 5:8; 

opt_result = []
for i_fold = nfold_set
      opt_result_temp1 = []
      for i_model = model_set
            opt_result_temp2 = load(string(Path_Load, 
                  "SharedPopCVfold", string(i_fold), 
                  "_model", string(i_model), ".jld2"))
            push!(opt_result_temp1, opt_result_temp2)
      end
      push!(opt_result, opt_result_temp1)
end

# ------------------------------------------------------------------------------
# checking the convergence
# ------------------------------------------------------------------------------
for i_fold = eachindex(nfold_set)
for i_model = eachindex(model_set)
      temp = opt_result[i_fold][i_model]
      temp_iter = parse.(Int,keys(temp)); temp_ind = sortperm(temp_iter)
      temp_logp = [temp[k].logp for k = keys(temp)]
      temp_par = hcat([collect(values(param2η(temp[k].parameters[1]))) for
                                                      k = keys(temp)]...)
      fig = figure(figsize=(12,12)); ax = subplot(6,6,1)
      ax.plot(temp_iter[temp_ind], temp_logp[temp_ind])
      ax.set_xticklabels([])
      ax.set_title("logp")
      for i_p = eachindex(p_names)
            ax = subplot(6,6,1 + i_p)
            ax.plot(temp_iter[temp_ind],temp_par[i_p,temp_ind])
            ax.set_title(p_names[i_p])
            if i_p <25
                  ax.set_xticklabels([])
            end
      end
      tight_layout()
      savefig(Path_Save * "OptPortofolio_" * m_names[model_set[i_model]] * 
                        "_fold" * string(i_fold) * ".pdf")
      close(fig)
end
end


# ------------------------------------------------------------------------------
# loading parameters
# ------------------------------------------------------------------------------
ηts = zeros(length(nfold_set), length(model_set), length(p_names))
param_df = DataFrame(pnames = p_names)
for i_model = eachindex(model_set)
      for i_fold = eachindex(nfold_set)
                  temp = opt_result[i_fold][i_model]
                  temp = temp[string(findmax(parse.(Int,keys(temp)))[1])]
                  ηts[i_fold,i_model,:] .= values(param2η(temp.parameters[1]))
                  param_df[!,m_names[model_set[i_model]] * 
                              "_f" * string(nfold_set[i_fold])] = 
                                                ηts[i_fold,i_model,:]
      end
end

@show param_df
CSV.write(Path_Save * "Params.CSV", param_df)


# ------------------------------------------------------------------------------
# sum logp values
# ------------------------------------------------------------------------------
logp_vals = (; 
      training     = zeros(length(nfold_set), length(model_set)),
      testing      = zeros(length(nfold_set), length(model_set))
      )
for i_model = eachindex(model_set)
      for i_fold = eachindex(nfold_set)
            A = Str_Agent_Policy(Param, Rs, NIS_object(ws));
            p = ComponentArray(parameters(A, ηts[i_fold,i_model,:]))
            lp_train = sum([logp(Data_ns[i_s], A, p) for i_s = CV_inds[i_fold][1]])
            lp_test  = sum([logp(Data_ns[i_s], A, p) for i_s = CV_inds[i_fold][2]])
            logp_vals.training[i_fold,i_model] = lp_train
            logp_vals.testing[i_fold,i_model] = lp_test
            
            println("------------------------")
            @show i_fold
            @show m_names[model_set[i_model]]
            @show logp_vals.training[i_fold,i_model]
            @show logp_vals.testing[i_fold,i_model]
      end
end

figure(figsize = (12,6)); 
for i_fold = eachindex(nfold_set)
      ax = subplot(1, length(nfold_set), i_fold)
      ax.plot(1:length(model_set), logp_vals.testing[i_fold,:],"o-")
      ax.set_xticks(1:length(model_set))
      ax.set_xticklabels(m_names[model_set])
      ax.set_title("fold" * string(nfold_set[i_fold]))
end
tight_layout()
savefig(Path_Save * "Testing_logL.pdf")

figure(figsize = (12,6)); 
for i_fold = eachindex(nfold_set)
      ax = subplot(1, length(nfold_set), i_fold )
      ax.plot(1:length(model_set), logp_vals.training[i_fold,:],"o-")
      ax.set_xticks(1:length(model_set))
      ax.set_xticklabels(m_names[model_set])
      ax.set_title("fold" * string(nfold_set[i_fold]))
end
tight_layout()
savefig(Path_Save * "Training_logL.pdf")
