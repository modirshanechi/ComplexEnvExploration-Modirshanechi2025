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
# ------------------------------------------------------------------------------
# Loading data
# ------------------------------------------------------------------------------
m_names = [string(m) for m = keys(model_settings)]
# model_set = [5,7,6,8]; rep_set = 1:5
model_set = [5,7,6,8]; rep_set = 1:2

n_model = length(model_set); n_rep = length(rep_set)
exp_r_MAT = zeros(n_rep, n_model, n_model)
d_exp_r_MAT = zeros(n_rep, n_model, n_model)
pxp_MAT = zeros(n_rep, n_model, n_model)


for i_model_true = eachindex(model_set)
        for i_rep = eachindex(rep_set)
                temp = load(Path_Save * "Results" * 
                        string(m_names[model_set[i_model_true]]) *
                        "_rep" * string(rep_set[i_rep]) *  ".jld2")
                exp_r_MAT[i_rep, i_model_true, :] .= temp["exp_r"]
                d_exp_r_MAT[i_rep, i_model_true, :] .= temp["d_exp_r"]
                pxp_MAT[i_rep, i_model_true, :] .= temp["pxp"]
        end
end        

exp_r = mean(exp_r_MAT,dims = 1)[1,:,:]
d_exp_r = std(exp_r_MAT,dims = 1)[1,:,:]
pxp = mean(pxp_MAT,dims = 1)[1,:,:]
d_pxp = std(pxp_MAT,dims = 1)[1,:,:]

# ------------------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------------------
Legends = m_names[model_set]


fig = figure(figsize=(12,5))
Y = exp_r
dY = d_exp_r
ax = subplot(1,2,1)
cp = ax.imshow(Y,vmin=0,vmax=0.6)
for i = 1:4
        for j = 1:4
                ax.text(j - 1, i - 1.1, string(round(Y[i,j],digits=2)),
                        horizontalalignment="center")
                ax.text(j - 1, i - 0.9, string("+-",round(dY[i,j],digits=2)),
                        horizontalalignment="center")
        end
end
fig.colorbar(cp, ax=ax)
ax.set_xticks(0:3); ax.set_xticklabels(Legends)
ax.set_yticks(0:3); ax.set_yticklabels(Legends)
ax.set_ylabel("True model"); ax.set_xlabel("Recovered model")
ax.set_title("Expected post")

Y = pxp
dY = d_pxp
ax = subplot(1,2,2)
cp = ax.imshow(Y,vmin=0,vmax=1.0)
for i = 1:4
        for j = 1:4
                ax.text(j - 1, i - 1.1, string(round(Y[i,j],digits=4)),
                        horizontalalignment="center")
                ax.text(j - 1, i - 0.9, string("+-",round(dY[i,j],digits=4)),
                        horizontalalignment="center")
        end
end
fig.colorbar(cp, ax=ax)
ax.set_xticks(0:3); ax.set_xticklabels(Legends)
ax.set_yticks(0:3); ax.set_yticklabels(Legends)
ax.set_ylabel("True model"); ax.set_xlabel("Recovered model")
ax.set_title("Excedence prob")

tight_layout()


savefig(Path_Save * "Confusion_matrix.pdf")
savefig(Path_Save * "Confusion_matrix.svg")
