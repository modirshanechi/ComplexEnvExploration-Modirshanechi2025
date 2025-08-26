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

Path_Save = "src/03_ModelRecovery/Results/"
Path_Load_Pars = "src/03_ModelRecovery/Results/FittedParameters/"
Path_Load_Data = "src/03_ModelRecovery/SData/"
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
nfolds = 3; nfold_set = 1:3; model_set = [5,7,6,8];

for i_rep = 1:5
for i_model_true = model_set
      temp = load(Path_Load_Data * string(keys(model_settings)[i_model_true]) * 
                  "/SData_rep" * string(i_rep) * ".jld2")
      Goal_type_Set = temp["SGoal_type_Set"]
      CV_inds = [Func_GoalCV(i, Goal_type_Set; nfolds = nfolds) for i = 1:nfolds]
      Data_ns = Str_Input2Agents.(temp["SData"]);
      Sub_Num = length(Data_ns)
      opt_result = []
      for i_fold = nfold_set
            opt_result_temp1 = []
            for i_model = model_set
                  println("------------------------------------------------")
                  println("------------------------------------------------")
                  println("------------------------------------------------")
                  println("Loading data")
                  @show i_model_true
                  @show i_rep
                  @show i_fold
                  @show i_model
                  opt_result_temp2 = load(string(Path_Load_Pars, 
                        "SharedPopCV_TrueModel", string(i_model_true),
                        "_rep", string(i_rep),
                        "_fold", string(i_fold), 
                        "_model", string(i_model), "_rec.jld2"))
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
            savefig(Path_Save * "Control/OptPortofolio__True" * m_names[i_model_true] *
                              "_rep" * string(i_rep) * "_mod" * m_names[model_set[i_model]] * 
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

      # ------------------------------------------------------------------------------
      # sum logp values agent-by-agent
      # ------------------------------------------------------------------------------
      N_epi = 5
      logp_vals_SbS = zeros(Sub_Num, N_epi, length(model_set))
      for i_sub = 1:Sub_Num
            for i_model = eachindex(model_set)
                  for i_fold = eachindex(nfold_set)
                        if i_sub ∈ CV_inds[nfold_set[i_fold]][2]
                              A = Str_Agent_Policy(Param, Rs, NIS_object(ws));
                              p = ComponentArray(parameters(A, ηts[i_fold,i_model,:]))
                              lps, _ = logp_pass_agent(Data_ns[i_sub], A, p);
                              logp_vals_SbS[i_sub,:,i_model] .= lps
                              println("------------------------")
                              @show i_sub
                              @show m_names[model_set[i_model]]
                              @show lps
                        end
                  end
            end
      end

      # ------------------------------------------------------------------------------
      # plotting hierarchical random effects
      # ------------------------------------------------------------------------------
      total_logp_vals_SbS = sum(logp_vals_SbS,dims = 2)[:,1,:]

      # all subjects
      subset = vcat([CV_inds[i][2] for i = nfold_set]...)
      L_names = Goal_type_Set[subset]

      L_matrix = deepcopy(total_logp_vals_SbS[subset,:])
      R_matrix_samples, M_matrix_samples, R_samples_all, M_samples_all, 
            exp_r, d_exp_r, xp, pxp, exp_M, BOR = MCMC_BMS_Statistics(L_matrix,
                  N_Chains=100, N_Sampling = Int(2e5), N_Sampling_BOR = Int(2e5),
                  α = 1/size(L_matrix)[2])

      Y = exp_M; Y_names = L_names

      x_names = m_names[model_set]

      # average 
      y = exp_r; dy = d_exp_r; x = 1:length(y)
      fig = figure(figsize=(12,6)); ax = subplot(1,2,1)
      ax.bar(x,y, color="k",alpha=0.7)
      ax.errorbar(1:length(y),y[:],yerr=dy[:],color="k",
                        linewidth=1,drawstyle="steps",linestyle="",capsize=3)
      ax.plot([x[1]-1,x[end]+1],[1,1] ./ length(y), 
                  linestyle="dashed",linewidth=1,color="k")
      title("Posterior Probabilities for Different Models")
      ax.set_xticks(x)
      ax.set_xticklabels(x_names,fontsize=9)
      ax.set_ylabel("E[P(model) | Data ]")
      ax.set_xlim([x[1]-1,x[end]+1])
      ax.set_ylim([0,1.0])

      y = pxp; x = 1:length(y)
      ax = subplot(1,2,2)
      ax.bar(x,y, color="k")
      ax.plot([x[1]-1,x[end]+1],[1,1] ./ length(y), 
                  linestyle="dashed",linewidth=1,color="k")
      title("Protected exceedence probabilities")
      ax.set_xticks(x)
      ax.set_xticklabels(x_names,fontsize=9)
      ax.set_ylabel("P[r_m > r_m' | Data ]")
      ax.set_xlim([x[1]-1,x[end]+1])
      ax.set_ylim([0,1.0])

      tight_layout()

      savefig(Path_Save * "Random_effect_BMS" * string(m_names[i_model_true]) *
                                                "_rep" * string(i_rep) *  ".pdf")
      savefig(Path_Save * "Random_effect_BMS" * string(m_names[i_model_true]) *
                                                "_rep" * string(i_rep) *  ".svg")

      # ------------------------------------------------------------------------------
      # save summary results
      # ------------------------------------------------------------------------------
      save(Path_Save * "Results" * string(m_names[i_model_true]) *
                              "_rep" * string(i_rep) *  ".jld2",
            "ηts",ηts, "param_df",param_df,
            "logp_vals_SbS",logp_vals_SbS, "Goal_type_Set",Goal_type_Set, 
            "exp_r", exp_r, "d_exp_r", d_exp_r, "xp", xp, "pxp", pxp, 
            "exp_M", exp_M, "BOR", BOR
            )

end
end
