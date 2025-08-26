using PyPlot
using IMRLExploration
using JLD2
using Random
using Statistics
using DataFrames
using LogExpFunctions
using CSV

using FitPopulations
using ComponentArrays

PyPlot.svg(true)
rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["svg.fonttype"] = "none"
rcParams["pdf.fonttype"] = 42

Path_Load = "src/02_PPCSimulation/Figures/"
Path_Save = "src/03_ModelRecovery/SData/"
# ------------------------------------------------------------------------------
# Dataframe for Statistics
# ------------------------------------------------------------------------------
model_set = 5:8
m_names = [string(m) for m = keys(model_settings)[model_set]]

# ------------------------------------------------------------------------------
# Plotting and reading results
# ------------------------------------------------------------------------------
n_rep = 5; n_sub_pergroup = 20
for i_model = eachindex(m_names)
        Data_Path_Load = Path_Load * string(m_names[i_model]) * 
                                        "/Data/sdata.jld2"
        temp = load(Data_Path_Load)
        SData = temp["SData"];
        i_pars = Int.(temp["i_pars"]) 
        SOutliers, SLong_Subject, SQuit_Subject, SData, 
                SGoal_type_Set, SSub_Num = 
                        Read_processed_data(Data=SData, 
                                Plotting = true, N_max = Inf)
        i_pars = i_pars[SOutliers .== 0]

        Data_Path_Save = Path_Save * string(m_names[i_model]) * "/"
        for i_rep = 1:n_rep
                inds_temp = [(1:SSub_Num)[SGoal_type_Set .== g][
                        (i_rep-1) * n_sub_pergroup .+  (1:n_sub_pergroup)] for
                                                                 g = 0:2]
                inds_temp = vcat(inds_temp...)
                SData_temp = SData[inds_temp]
                SGoal_type_Set_temp = SGoal_type_Set[inds_temp]
                i_pars_temp = i_pars[inds_temp]
                save(Data_Path_Save * "SData_rep" * string(i_rep) * ".jld2",
                        "inds", inds_temp,
                        "SData", SData_temp,
                        "SGoal_type_Set", SGoal_type_Set_temp,
                        "i_pars", i_pars_temp)
        end
        
end