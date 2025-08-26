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

Path_Save = "src/02_PPCSimulation/Figures/"


# whether to exclude outlier simulations (Fig 5 and S3) or not (Fig S4)
IfNoExc = false
if IfNoExc
        name_tag_save = "NoExc"
else
        name_tag_save = ""
end
# ------------------------------------------------------------------------------
# Dataframe for Statistics
# ------------------------------------------------------------------------------
m_names = [string(m) for m = keys(model_settings)][5:8]
df_stats = DataFrame(Models = vcat(m_names,"Humans"))

# lenght of the 1st episode
df_stats.E1N_μ  = zeros(length(m_names) + 1)
df_stats.dE1N_μ = zeros(length(m_names) + 1)

# correlation of goal vlaue with number of actions in Epi 2-5
df_stats[:,"E25N_ρ"] = 
                zeros(length(m_names) + 1)
df_stats[:,"dE25N_ρ"] = 
                zeros(length(m_names) + 1)
# correlation of goal vlaue with ratio of time spent in stoch. part in Epi 2-5
df_stats[:,"E25r_ρ"] = 
                zeros(length(m_names) + 1)
df_stats[:,"dE25r_ρ"] = 
                zeros(length(m_names) + 1)
for i = 0:2
        # median lenght of episodes 2-5
        df_stats[:,"E25N_μ_G" * string(i)] = 
                        zeros(length(m_names) + 1)
        df_stats[:,"dE25N_μ_G" * string(i)] = 
                        zeros(length(m_names) + 1)
        # ratio of time spent in stochastic part in episodes 2-5
        df_stats[:,"E25r_μ_G" * string(i)] = 
                        zeros(length(m_names) + 1)                        
        df_stats[:,"dE25r_μ_G" * string(i)] = 
                        zeros(length(m_names) + 1)
        # correlation of episode lenght with episode
        df_stats[:,"E25N_ρ_G" * string(i)] = 
                        zeros(length(m_names) + 1)
        df_stats[:,"dE25N_ρ_G" * string(i)] = 
                        zeros(length(m_names) + 1)
        # correlation of ratio of time spent in stochastic part with episode
        df_stats[:,"E25r_ρ_G" * string(i)] = 
                        zeros(length(m_names) + 1)                        
        df_stats[:,"dE25r_ρ_G" * string(i)] = 
                        zeros(length(m_names) + 1)
end
# ratio of time in stochastic part in 1st half of episode 1
df_stats[:,"E11S_μ"]  = zeros(length(m_names) + 1)
df_stats[:,"dE11S_μ"] = zeros(length(m_names) + 1)
# ratio of time in stochastic part in 2nd half of episode 1
df_stats[:,"E12S_μ"]  = zeros(length(m_names) + 1)
df_stats[:,"dE12S_μ"] = zeros(length(m_names) + 1)
# ratio of time in trap states in 1st half of episode 1
df_stats[:,"E11T_μ"]  = zeros(length(m_names) + 1)
df_stats[:,"dE11T_μ"] = zeros(length(m_names) + 1)
# ratio of time in trap states in 2nd half of episode 1
df_stats[:,"E12T_μ"]  = zeros(length(m_names) + 1)
df_stats[:,"dE12T_μ"] = zeros(length(m_names) + 1)

for s = [1:3,[4],5:6]
# rate of choosing different actions in progressing states in episode 2-5
df_stats[:,"ActR1_μ_S" * string(s) * "E1"] =
        zeros(length(m_names) + 1)                        
df_stats[:,"dActR1_μ_S" * string(s) * "E1"] =
        zeros(length(m_names) + 1)
df_stats[:,"ActR2_μ_S" * string(s) * "E1"] =
        zeros(length(m_names) + 1)                        
df_stats[:,"dActR2_μ_S" * string(s) * "E1"] =
        zeros(length(m_names) + 1)
for i = 0:2
        # rate of choosing different actions in progressing states in episode 2-5
        df_stats[:,"ActR1_μ_S" * string(s) * "G" * string(i)  * "E25"] =
                zeros(length(m_names) + 1)                        
        df_stats[:,"dActR1_μ_S" * string(s) * "G" * string(i)  * "E25"] =
                zeros(length(m_names) + 1)
        df_stats[:,"ActR2_μ_S" * string(s) * "G" * string(i)  * "E25"] =
                zeros(length(m_names) + 1)                        
        df_stats[:,"dActR2_μ_S" * string(s) * "G" * string(i)  * "E25"] =
                zeros(length(m_names) + 1)
end
end

# ------------------------------------------------------------------------------
# Plotting and reading results
# ------------------------------------------------------------------------------
Colors = ["#004D66","#0350B5","#00CCF5"]
Legends = ["CHF2","CHF3","CHF4"]
for i_model = 1:(length(m_names) + 1)
        if i_model <= length(m_names)
                Data_Path_Load = Path_Save * string(m_names[i_model]) * 
                                                "/Data/sdata.jld2"
                temp = load(Data_Path_Load)
                SData = temp["SData"]
                if IfNoExc
                        SGoal_type_Set = temp["SGoal_type_Set"]
                        SSub_Num = length(SGoal_type_Set)
                        SOutliers = zeros(SSub_Num)
                else
                        SOutliers, SLong_Subject, SQuit_Subject, SData, 
                                SGoal_type_Set, SSub_Num = 
                                Read_processed_data(Data=SData, Plotting = true)
                end
                
                Fig_Path_Save = Path_Save * string(m_names[i_model]) * "/"
                points_to_plot = 20
                points_to_plot2 = 60
        else
                SOutliers, SLong_Subject, SQuit_Subject, SData, 
                        SGoal_type_Set, SSub_Num =
                                Read_processed_data(Plotting = true);
                Fig_Path_Save = Path_Save * "Humans/"
                points_to_plot = -1
                points_to_plot2 = -1
        end

        # outliers
        println("---------------------------------")
        println(df_stats.Models[i_model])
        println(string("Outliers: ", string(mean(SOutliers) * 100), "%"))
        println(string("delta outliers: ", 
                        string(std(SOutliers) / sqrt(SSub_Num) * 100), "%"))
        
        # lenght of episode 1
        E1_Stat_MedNumb = Func_plot_hist_Epi1(SData, SGoal_type_Set;
                Colors = Colors ,
                Path_Save = Fig_Path_Save * "Length_Epi1/" * name_tag_save)
        df_stats.E1N_μ[i_model]  = E1_Stat_MedNumb.μ
        df_stats.dE1N_μ[i_model] = E1_Stat_MedNumb.dμ
        close("all")

        # lenght of episodes 2-5
        E25_Stat_MedNumb = Func_plot_hist_Epi25(SData, SGoal_type_Set;
                Epis = 2:5,
                Colors = Colors,
                Path_Save = Fig_Path_Save * "Length_Epi25/" * name_tag_save,
                y_lims = [0,400],
                points_to_plot = points_to_plot,
                Sub_testing = false)
        df_stats[i_model,"E25N_ρ"]  = E25_Stat_MedNumb.ρ
        df_stats[i_model,"dE25N_ρ"] = E25_Stat_MedNumb.dρ
        for i = 0:2
                df_stats[i_model,"E25N_μ_G" * string(i)] = 
                        E25_Stat_MedNumb.μ[i+1,2]
                df_stats[i_model,"dE25N_μ_G" * string(i)] = 
                        E25_Stat_MedNumb.dμ[i+1,2]
                df_stats[i_model,"E25N_ρ_G" * string(i)] = 
                        E25_Stat_MedNumb.ρGs[i+1][1]
                df_stats[i_model,"dE25N_ρ_G" * string(i)] = 
                        E25_Stat_MedNumb.ρGs[i+1][2]
        end
        close("all")

        # lenght of episodes 1-5
        Func_plot_hist_Epi15(SData, SGoal_type_Set;
                Epis = 2:5,
                Colors = Colors,
                Path_Save = Fig_Path_Save * "Length_Epi15/" * name_tag_save,
                y_lims = [0,1200],
                points_to_plot = points_to_plot,
                Sub_testing = false)
        close("all")

        # action ratio in all episodes
        Epi = 1
        E1_Stat_ARatios = Func_plot_action_probability(SData, SGoal_type_Set, Epi;
                        Colors = Colors,
                        Legends = ["CHF2","CHF3","CHF4"],
                        Path_Save = Fig_Path_Save * "",
                        name_tag = name_tag_save,
                        fix_TM = false,
                        points_to_plot = points_to_plot)
        Epi = 2
        Func_plot_action_probability(SData, SGoal_type_Set, Epi;
                        Colors = Colors,
                        Legends = ["CHF2","CHF3","CHF4"],
                        Path_Save = Fig_Path_Save * "",
                        name_tag = name_tag_save,
                        fix_TM = false,
                        points_to_plot = points_to_plot)
        Epi = [2,3,4,5]
        E25_Stat_ARatios = Func_plot_action_probability(SData, SGoal_type_Set, Epi;
                        Colors = Colors,
                        Legends = ["CHF2","CHF3","CHF4"],
                        Path_Save = Fig_Path_Save * "",
                        name_tag = name_tag_save,
                        fix_TM = false,
                        points_to_plot = points_to_plot)
        close("all")
        
        # action ratio in all episodes
        for s = [1:3,[4],5:6]
        # rate of choosing different actions in progressing states in episode 2-5
        df_stats[i_model,"ActR1_μ_S" * string(s) * "E1"] =
                mean(E1_Stat_ARatios.μ[s,1,1])
        df_stats[i_model,"dActR1_μ_S" * string(s) * "E1"] =
                mean(E1_Stat_ARatios.dμ[s,1,1])
        df_stats[i_model,"ActR2_μ_S" * string(s) * "E1"] =
                mean(E1_Stat_ARatios.μ[s,2,1])
        df_stats[i_model,"dActR2_μ_S" * string(s) * "E1"] =
                mean(E1_Stat_ARatios.dμ[s,2,1])
        for i = 0:2
                # rate of choosing different actions in progressing states in episode 2-5
                df_stats[i_model,"ActR1_μ_S" * string(s) * "G" * string(i)  * "E25"] =
                        mean(E25_Stat_ARatios.μ[s,1,i+1])
                df_stats[i_model,"dActR1_μ_S" * string(s) * "G" * string(i)  * "E25"] =
                        mean(E25_Stat_ARatios.dμ[s,1,i+1])
                df_stats[i_model,"ActR2_μ_S" * string(s) * "G" * string(i)  * "E25"] =
                        mean(E25_Stat_ARatios.μ[s,2,i+1])
                df_stats[i_model,"dActR2_μ_S" * string(s) * "G" * string(i)  * "E25"] =
                        mean(E25_Stat_ARatios.dμ[s,2,i+1])
        end
        end
                
        # state ratio in sections of episode 1
        E1_Stat_Ratios = Func_plot_state_ratio_Epi1(SData, SGoal_type_Set;
                Colors = Colors[[1,3]],
                Path_Save = Fig_Path_Save * "StateRatio_Epi1/" * name_tag_save,
                Sub_testing = false,
                Traps = [8,9],
                Stoch = [7],
                All_states = Array(1:9),
                points_to_plot = points_to_plot2)
        df_stats[i_model,"E11S_μ"]  = E1_Stat_Ratios.μ[1,2]
        df_stats[i_model,"dE11S_μ"] = E1_Stat_Ratios.dμ[1,2]
        df_stats[i_model,"E12S_μ"]  = E1_Stat_Ratios.μ[2,2]
        df_stats[i_model,"dE12S_μ"] = E1_Stat_Ratios.dμ[2,2]
        df_stats[i_model,"E11T_μ"]  = E1_Stat_Ratios.μ[1,1]
        df_stats[i_model,"dE11T_μ"] = E1_Stat_Ratios.dμ[1,1]
        df_stats[i_model,"E12T_μ"]  = E1_Stat_Ratios.μ[2,1]
        df_stats[i_model,"dE12T_μ"] = E1_Stat_Ratios.dμ[2,1]
        close("all")

        # state ratio in episodes 2-5
        E25_Stat_Ratio = Func_plot_state_ratio_Epi25(SData, SGoal_type_Set;
                Colors = Colors,
                Path_Save = Fig_Path_Save * "StateRatio_Epi25/" * name_tag_save,
                Sub_testing = false,
                Traps = [8,9],
                Stoch = [7],
                All_states = Array(1:9),
                points_to_plot = points_to_plot)
        #
        df_stats[i_model,"E25r_ρ"]  = E25_Stat_Ratio.ρs[2]
        df_stats[i_model,"dE25r_ρ"] = E25_Stat_Ratio.dρs[2]
        for i = 0:2
                df_stats[i_model,"E25r_μ_G" * string(i)] = 
                        E25_Stat_Ratio.μs[2][i+1,2]
                df_stats[i_model,"dE25r_μ_G" * string(i)] = 
                        E25_Stat_Ratio.dμs[2][i+1,2]
                df_stats[i_model,"E25r_ρ_G" * string(i)] = 
                        E25_Stat_Ratio.ρGs[2][i+1][1]
                df_stats[i_model,"dE25r_ρ_G" * string(i)] = 
                        E25_Stat_Ratio.ρGs[2][i+1][2]
        end
        close("all")

end
CSV.write(Path_Save * "PPCStats" * name_tag_save * ".CSV", df_stats)