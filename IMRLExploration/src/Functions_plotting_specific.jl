# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_plot_hist_Epi1(Data, Goal_type_Set; Epi = 1,
             Colors = ["#004D66","#0350B5","#00CCF5"],
             Path_Save = "src/Data_Analysis/Figures/Length_Epi1/",
             Bootstrap_size = 10000,
             ifsave = true)
    # Extracting subjects' info
    mY, dY, MedY, dMedY, Q25_Y, Q75_Y, Y =
        Func_EpiLenghts_statistics(Data,Goal_type_Set)
    y = Y[:,Epi]

    # Plotting
    fig = figure(figsize=(10,7)); ax = gca()
    ax.hist(y,bins= 10 .^ (0:0.15:4), color = Colors[1],
              weights = ones(length(y))./length(y))
    ax.set_xscale("log")
    ax.set_xlim(5e0,5*10^3)
    ax.set_title("Histogram of lenght of the 1st episode")

    y_min = 0
    y_max = 0.3
    ax.set_ylim([y_min,y_max])

    μ = median(y)
    dμ = stderror(bootstrap(median, y, BasicSampling(Bootstrap_size)))[1]

    ax.plot(μ .* ones(2), [y_min,y_max],"--k")
    ax.text(sqrt(10)*μ,(y_max+y_min)*2/3,
            string("median = ", string(μ), "+-", string(round(dμ))),
            fontsize=4, horizontalalignment="center", rotation=0)
    if ifsave
        savefig(string(Path_Save, "Histogram.pdf" ))
        savefig(string(Path_Save, "Histogram.svg" ))
    else
        display(fig)
    end
    return (; μ = μ, dμ = dμ)
end
export Func_plot_hist_Epi1

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_plot_hist_Epi25(Data, Goal_type_Set;
        Epis = 2:5,
        Colors = ["#004D66","#0350B5","#00CCF5"],
        Path_Save = "src/Data_Analysis/Figures/Length_Epi25/",
        y_lims = [0,500],
        points_to_plot = -1,
        Sub_testing = false,
        Sample_size_for_test = 20,
        Repeating_test = 1000,
        ifsave = true
        )

        mY, dY, MedY, dMedY, Q25_Y, Q75_Y, Y = Func_EpiLenghts_statistics(Data, Goal_type_Set)
        
        Title_specifier = "Len_linear"
        Title = "Average number of actions in episode 2-5"

        Title_specifier_med = "Len_linear_median"
        Title_med = "Median number of actions in episode 2-5"
        ρGs = Vector{Tuple{Float64, Float64}}([])
        for i_group = 1:3
            Func_Bar_Plots_specific_oneGroup(i_group, mY[:,Epis], dY[:,Epis], Y[:,Epis],
                             Goal_type_Set,
                             Path_Save, Title_specifier, Title;
                             Colors = Colors,
                             Testing = true, α = 1.1, δα = 0.1,
                             fig_size = (3.5,6), fig_close = false,
                             y_lims = y_lims,
                             points_to_plot = points_to_plot,
                             Sub_testing = Sub_testing,
                             Sample_size_for_test = Sample_size_for_test,
                             Repeating_test = Repeating_test,
                             ifsave = ifsave)

            temp = Func_Bar_Plots_specific_oneGroup(i_group, MedY[:,Epis], dMedY[:,Epis], Y[:,Epis],
                             Goal_type_Set,
                             Path_Save, Title_specifier_med, Title_med;
                             Colors = Colors,
                             Testing = true, α = 1.1, δα = 0.1,
                             fig_size = (3.5,6), fig_close = false,
                             y_lims = y_lims,
                             points_to_plot = points_to_plot,
                             Sub_testing = Sub_testing,
                             Sample_size_for_test = Sample_size_for_test,
                             Repeating_test = Repeating_test,
                             pass_stats = true,
                             ifsave = ifsave)
            push!(ρGs, temp)
        end

        mY, dY, MedY, dMedY, Q25_Y, Q75_Y, Y = 
                    Func_EpiLenghts_statistics_2parts(Data, Goal_type_Set)
        Title_specifier = "Len_2parts_epi25"
        Title = "Average number of actions in episode 2-5"
        Func_Bar_Plots_linear_specific(mY[:,2], dY[:,2], Y[:,2],
                                Goal_type_Set,
                                Path_Save, Title_specifier, Title;
                                Colors = Colors,
                                Spearman_test = false, α = 0.8, δα = 0.1,
                                fig_size = (3.5,6), fig_close = false,
                                y_lims = y_lims,
                                points_to_plot = points_to_plot,
                                Sub_testing = Sub_testing,
                                Sample_size_for_test = Sample_size_for_test,
                                Repeating_test = Repeating_test,
                                x_lims = [0.25,3.75],
                                x_0ticks = ["Average Epi 2:5"],
                                ifsave = ifsave)
        mY, dY, MedY, dMedY, Q25_Y, Q75_Y, Y = 
                    Func_EpiLenghts_statistics_2parts(Data, Goal_type_Set,
                                                        median_parts = true)
        Title_specifier = "Len_2parts_epi25_median"
        Title = "Median number of actions in episode 2-5"
        ρ, dρ = Func_Bar_Plots_linear_specific(MedY[:,2], dMedY[:,2], Y[:,2],
                                Goal_type_Set,
                                Path_Save, Title_specifier, Title;
                                Colors = Colors,
                                Spearman_test = false, α = 0.8, δα = 0.1,
                                fig_size = (3.5,6), fig_close = false,
                                y_lims = y_lims,
                                points_to_plot = points_to_plot,
                                Sub_testing = Sub_testing,
                                Sample_size_for_test = Sample_size_for_test,
                                Repeating_test = Repeating_test,
                                x_lims = [0.25,3.75],
                                x_0ticks = ["Average Epi 2:5"],
                                pass_stats = true,
                                ifsave = ifsave)
        μ = MedY; dμ = dMedY # saving stats for median
        return (; μ = μ, dμ = dμ, ρ = ρ, dρ = dρ, ρGs = ρGs)
end
export Func_plot_hist_Epi25


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_plot_hist_Epi15(Data, Goal_type_Set;
        Epis = 2:5,
        Colors = ["#004D66","#0350B5","#00CCF5"],
        Path_Save = "src/Data_Analysis/Figures/Length_Epi25/",
        y_lims = [0,1200],
        points_to_plot = -1,
        Sub_testing = false,
        Sample_size_for_test = 20,
        Repeating_test = 1000,
        ifsave = true)

        mY, dY, MedY, dMedY, Q25_Y, Q75_Y, Y = Func_EpiLenghts_statistics(Data, Goal_type_Set)
        Title_specifier = "Len"
        Title = "Average number of actions"
        Func_Bar_Plots_specific(mY, dY, Y, Goal_type_Set,
                                Path_Save, Title_specifier, Title;
                                Colors = Colors,
                                Testing = true,fig_size = (10,7),fig_close = false,
                                y_lims = y_lims, α = 0.9, δα=0.05,
                                points_to_plot = points_to_plot,
                                Sub_testing = Sub_testing,
                                Sample_size_for_test = Sample_size_for_test,
                                Repeating_test = Repeating_test,
                                ifsave = ifsave)
        Title_specifier = "Len_Med"
        Title  = "Med number of actions"
        Func_Bar_Plots_specific(MedY, dMedY, Y, Goal_type_Set,
                                Path_Save, Title_specifier, Title;
                                Colors = Colors,
                                Testing = true,fig_size = (10,7),fig_close = false,
                                y_lims = y_lims, α = 0.9, δα=0.05,
                                points_to_plot = points_to_plot,
                                Sub_testing = Sub_testing,
                                Sample_size_for_test = Sample_size_for_test,
                                Repeating_test = Repeating_test,
                                ifsave = ifsave)

        mY, dY, MedY, dMedY, Q25_Y, Q75_Y, Y = Func_EpiLenghts_statistics_2parts(Data, Goal_type_Set)
        Title_specifier = "Len_2parts"
        Title = "Average number of actions"
        Func_Bar_Plots_specific(mY, dY, Y, Goal_type_Set,
                                Path_Save, Title_specifier, Title;
                                Colors = Colors,
                                Testing = true,fig_size = (6,7),fig_close = false,
                                y_lims = y_lims,
                                points_to_plot = points_to_plot,
                                Sub_testing = Sub_testing,
                                Sample_size_for_test = Sample_size_for_test,
                                Repeating_test = Repeating_test,
                                ifsave = ifsave)
end
export Func_plot_hist_Epi15

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_plot_action_probability(Data, Goal_type_Set, Epi;
            Colors = ["#004D66","#0350B5","#00CCF5"],
            Legends = ["CHF2","CHF3","CHF4"],
            Path_Save = "src/Data_Analysis/Figures/",
            Test_function = OneSampleTTest,
            BayTest_function = BIC_OneSampleTTest,
            Sub_testing = false, Scale_std = false, points_to_plot = -1,
            Sample_size_for_test = 20,
            Repeating_test = 1000,
            fix_TM = false,
            return_action_matrix = false,
            name_tag = "",
            ifplot = true,
            ifsave = true)

        Path_Save = string(Path_Save,"Action_prob_Epi",string(Epi),"/",name_tag)
        Sub_Num = length(Data)
        Action_matrix = zeros(6,3,Sub_Num)      # a1: good, a2: neutral, a3: bad
        for n = 1:Sub_Num
            TM = Data[n].TM
            if fix_TM
                Func_fix_TM!(Data[n])
            end
            if length(Epi) == 1
                states  = Data[n].states[Epi]
                actions = Data[n].actions[Epi]
            else
                states  = vcat(Data[n].states[Epi]...)
                actions = vcat(Data[n].actions[Epi]...)
            end
            inds = 1:length(states)
            for s = 1:6
                if sum(states .== s) > 0
                    a = actions[inds[states .== s]]
                    a = Func_ActType.(a, s, TM=TM, scores = [3,2,2,1])
                    for i = 1:3
                        Action_matrix[s,i,n] = mean(a .== i)
                    end
                else
                    Action_matrix[s,:,n] .= NaN
                end
            end
        end
        if return_action_matrix
                return Action_matrix
        end
        mA = zeros(6,3,3)
        dA = zeros(6,3,3)
        pvals = zeros(6,3,3)
        logBFs = zeros(6,3,3,2)
        for i = 1:3
            if Epi == 1
                Act_mat_temp = Action_matrix[:,:,:]
            else
                Act_mat_temp = Action_matrix[:,:,Goal_type_Set .== (i-1)]
            end
            for s = 1:6
                inds = 1:size(Act_mat_temp)[3]
                inds = inds[isnan.(Act_mat_temp[s,1,:]).== 0]
                mA[s,:,i] = mean(Act_mat_temp[s,:,inds], dims = 2)[:]
                if Sub_testing & Scale_std
                        dA[s,:,i] = std(Act_mat_temp[s,:,inds], dims = 2)[:] ./
                                        sqrt(Sample_size_for_test)
                else
                        dA[s,:,i] = std(Act_mat_temp[s,:,inds], dims = 2)[:] ./
                                        sqrt(length(inds))
                end

                if Sub_testing
                        Y1_test = Act_mat_temp[s,1,inds]; Y2_test = Act_mat_temp[s,2,inds];
                        temp = Func_subsamp_test(Y1_test,Y2_test,
                                             Sample_size_for_test,
                                             Sample_size_for_test,
                                             Repeating_test,
                                             Test_function, same_ind = true)
                        pvals[s,1,i] = mean(temp[isnan.(temp) .== 0])
                        Y1_test = Act_mat_temp[s,1,inds]; Y2_test = Act_mat_temp[s,3,inds];
                        temp = Func_subsamp_test(Y1_test,Y2_test,
                                             Sample_size_for_test,
                                             Sample_size_for_test,
                                             Repeating_test,
                                             Test_function, same_ind = true)
                        pvals[s,2,i] = mean(temp[isnan.(temp) .== 0])
                        Y1_test = Act_mat_temp[s,2,inds]; Y2_test = Act_mat_temp[s,3,inds];
                        temp = Func_subsamp_test(Y1_test,Y2_test,
                                             Sample_size_for_test,
                                             Sample_size_for_test,
                                             Repeating_test,
                                             Test_function, same_ind = true)
                        pvals[s,3,i] = mean(temp[isnan.(temp) .== 0])

                        Y1_test = Act_mat_temp[s,1,inds]; Y2_test = Act_mat_temp[s,2,inds];
                        temp = Func_subsamp_test(Y1_test,Y2_test,
                                     Sample_size_for_test, Sample_size_for_test,
                                     Repeating_test, BayTest_function,
                                     BayesianTest = true, same_ind = true)
                        logBFs[s,1,i,1],logBFs[s,1,i,2] = mean.(temp)

                        Y1_test = Act_mat_temp[s,1,inds]; Y2_test = Act_mat_temp[s,3,inds];
                        temp = Func_subsamp_test(Y1_test,Y2_test,
                                     Sample_size_for_test, Sample_size_for_test,
                                     Repeating_test, BayTest_function,
                                     BayesianTest = true, same_ind = true)
                        logBFs[s,2,i,1],logBFs[s,2,i,2] = mean.(temp)

                        Y1_test = Act_mat_temp[s,2,inds]; Y2_test = Act_mat_temp[s,3,inds];
                        temp = Func_subsamp_test(Y1_test,Y2_test,
                                     Sample_size_for_test, Sample_size_for_test,
                                     Repeating_test, BayTest_function,
                                     BayesianTest = true, same_ind = true)
                        logBFs[s,3,i,1],logBFs[s,3,i,2] = mean.(temp)
                else
                        Test_result = Test_function(Act_mat_temp[s,1,inds],
                                                    Act_mat_temp[s,2,inds])
                        @show Test_result
                        pvals[s,1,i] = pvalue(Test_result)
                        Test_result = Test_function(Act_mat_temp[s,1,inds],
                                                    Act_mat_temp[s,3,inds])
                        @show Test_result
                        pvals[s,2,i] = pvalue(Test_result)
                        Test_result = Test_function(Act_mat_temp[s,2,inds],
                                                    Act_mat_temp[s,3,inds])
                        @show Test_result
                        pvals[s,3,i] = pvalue(Test_result)

                        logBFs[s,1,i,1] = BayTest_function(Act_mat_temp[s,1,inds],
                                                         Act_mat_temp[s,2,inds])
                        logBFs[s,2,i,1] = BayTest_function(Act_mat_temp[s,1,inds],
                                                         Act_mat_temp[s,3,inds])
                        logBFs[s,3,i,1] = BayTest_function(Act_mat_temp[s,2,inds],
                                                         Act_mat_temp[s,3,inds])
                        logBFs[s,:,i,2] = - logBFs[s,:,i,1]
                end
            end
        end
        # --------------------------------------------------------------------------
        # Figures
        # --------------------------------------------------------------------------
        σ = 0.2
        if points_to_plot == -1
                points_to_plot = Sub_Num
        end
        if ifplot
        for i = 1:3
            y = mA[:,:,i]
            dy = dA[:,:,i]
            fig = figure(figsize=(10,6));
            for s = 1:6
                ax = subplot(2,3,s)
                ax.bar(1:3,y[s,:],color = Colors[i])
                ax.errorbar(1:3,y[s,:],yerr=dy[s,:],color="k",
                            linewidth=1,drawstyle="steps",linestyle="",capsize=3)
                ax.set_xlim([0.5,3.5])
                ax.set_ylim([0.0,1.0])
                ax.set_title(string("state ", string(s)))

                ax.plot([1,2],[0.8,0.8], "k")
                ax.text(1.5,0.82,
                string("p:", Func_pval_string(pvals[s,1,i]),
                        ", lBF:", Func_logBF_string(logBFs[s,1,i,1]),
                             ",", Func_logBF_string(logBFs[s,1,i,2])),
                        fontsize=4, horizontalalignment="center", rotation=0)
                ax.plot([1,3],[0.9,0.9], "k")
                ax.text(2,0.92,
                string("p:", Func_pval_string(pvals[s,2,i]),
                        ", lBF:", Func_logBF_string(logBFs[s,2,i,1]),
                             ",", Func_logBF_string(logBFs[s,2,i,2])),
                        fontsize=4, horizontalalignment="center", rotation=0)
                ax.plot([2,3],[0.5,0.5], "k")
                ax.text(2.5,0.52,
                string("p:", Func_pval_string(pvals[s,3,i]),
                       ", lBF:", Func_logBF_string(logBFs[s,3,i,1]),
                            ",", Func_logBF_string(logBFs[s,3,i,2])),
                        fontsize=4, horizontalalignment="center", rotation=0)

                if Epi == 1
                    Act_mat_temp = Action_matrix[:,:,:]
                else
                    Act_mat_temp = Action_matrix[:,:,Goal_type_Set .== (i-1)]
                end
                inds = 1:size(Act_mat_temp)[3]
                inds = inds[isnan.(Act_mat_temp[s,1,:]).== 0]
                As = Act_mat_temp[s,:,inds]
                for j = 1:min(size(As)[2],points_to_plot)
                        x_plot = Array((1:3) .+ 2*σ*(rand() - 0.5))
                        y_plot = (As[:,j])[:]
                        ax.plot(x_plot,y_plot,".k",alpha = 0.5)
                        ax.plot(x_plot,y_plot,"k",alpha = 0.1)
                end
            end
            tight_layout()
            if ifsave
                savefig(string(Path_Save, "Action_prob_", Legends[i], ".pdf" ))
                savefig(string(Path_Save, "Action_prob_", Legends[i], ".svg" ))
            else
                display(fig)
            end


            fig = figure(figsize=(6,6));
            for s = 1:6
                ax = subplot(2,3,s)
                ax.bar(1:2,y[s,1:2],color = Colors[i])
                ax.errorbar(1:2,y[s,1:2],yerr=dy[s,1:2],color="k",
                            linewidth=1,drawstyle="steps",linestyle="",capsize=3)
                ax.set_xlim([0.5,2.5])
                ax.set_ylim([0.0,1.0])
                ax.set_title(string("state ", string(s)))

                ax.plot([1,2],[0.8,0.8], "k")
                ax.text(1.5,0.82,
                        string("p:", Func_pval_string(pvals[s,1,i]),
                        ", lBF:", Func_logBF_string(logBFs[s,1,i,1]),
                             ",", Func_logBF_string(logBFs[s,1,i,2])),
                        fontsize=4, horizontalalignment="center", rotation=0)

                if Epi == 1
                    Act_mat_temp = Action_matrix[:,1:2,:]
                else
                    Act_mat_temp = Action_matrix[:,1:2,Goal_type_Set .== (i-1)]
                end
                inds = 1:size(Act_mat_temp)[3]
                inds = inds[isnan.(Act_mat_temp[s,1,:]).== 0]
                As = Act_mat_temp[s,:,inds]
                for j = 1:min(size(As)[2],points_to_plot)
                        x_plot = Array((1:2) .+ 2*σ*(rand() - 0.5))
                        y_plot = (As[:,j])[:]
                        ax.plot(x_plot,y_plot,".k",alpha = 0.5)
                        ax.plot(x_plot,y_plot,"k",alpha = 0.1)
                end
            end
            tight_layout()
            if ifsave
                savefig(string(Path_Save, "Action_prob_", Legends[i], "_2choices.pdf" ))
                savefig(string(Path_Save, "Action_prob_", Legends[i], "_2choices.svg" ))
            else
                display(fig)
            end
        end
        end

        return (; μ = mA, dμ = dA)
end
export Func_plot_action_probability
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_plot_state_ratio_Epi25(Data, Goal_type_Set;
                Colors = ["#004D66","#0350B5","#00CCF5"],
                Path_Save = "src/Data_Analysis/Figures/StateRatio_Epi25/",
                Traps = [8,9], Stoch = [7],
                All_states = [1,2,3,4,5,6,7,8,9],
                points_to_plot = -1,
                Sub_testing = false,
                Sample_size_for_test = 20,
                Repeating_test = 1000,
                return_Y_set = false,
                ifsave = true)
    Sub_Num = length(Data)
    Epi_sum_latency_traps = zeros(Sub_Num,5)
    Epi_sum_latency_stoch = zeros(Sub_Num,5)
    Epi_sum_latency_alls = zeros(Sub_Num,5)
    for Epi = 1:5
        temp = Func_desired_states_visit(Data, DesiredStates = Traps,
                                         Epi= Epi)
        Epi_sum_latency_traps[:,Epi] = sum.(temp[4])
        temp = Func_desired_states_visit(Data, DesiredStates = Stoch,
                                         Epi= Epi)
        Epi_sum_latency_stoch[:,Epi] = sum.(temp[4])
        temp = Func_desired_states_visit(Data, DesiredStates = All_states,
                                         Epi= Epi)
        Epi_sum_latency_alls[:,Epi] = sum.(temp[4])
    end

    Y_traps = Epi_sum_latency_traps ./ Epi_sum_latency_alls
    Y_stoch = Epi_sum_latency_stoch ./ Epi_sum_latency_alls

    Y_set = [Y_traps, Y_stoch]
    if return_Y_set
        return Y_set
    end
    Legend = ["trap", "stochastic"]
    Y_lims = [[0,1.0],[0,1.0]]
    μs = Vector{Matrix{Float64}}([]); dμs = Vector{Matrix{Float64}}([])
    ρs = Vector{Float64}([]); dρs = Vector{Float64}([]);
    ρGs = Vector{Vector{Tuple{Float64, Float64}}}([])
    for i_Y = 1:2
        mY, dY, MedY, dMedY, Q25_Y, Q75_Y, Y =
            Func_group_based_information(Y_set[i_Y],Goal_type_Set;two_parts = true)
        push!(μs, MedY); push!(dμs, dMedY)
        Title_specifier = string(Legend[i_Y],"_2")
        Title = string("Average ratio of time in ",
                        Title_specifier,
                        " states")
        Func_Bar_Plots_specific(mY, dY, Y, Goal_type_Set,
                              Path_Save, Title_specifier, Title;
                              Colors = Colors,
                              Testing = true,fig_size = (6,7),fig_close = false,
                              y_lims = Y_lims[i_Y],
                              points_to_plot = points_to_plot,
                              Sub_testing = Sub_testing,
                              Sample_size_for_test = Sample_size_for_test,
                              Repeating_test = Repeating_test,
                              ifsave = ifsave)

        ρ, dρ = Func_Bar_Plots_linear_specific(mY[:,2], dY[:,2], Y[:,2],
                                       Goal_type_Set,
                                       Path_Save, Title_specifier, Title;
                                       Colors = Colors,
                                       Spearman_test = false, α = 0.8, δα = 0.1,
                                       fig_size = (3.5,6), fig_close = false,
                                       y_lims = Y_lims[i_Y],
                                       x_lims = [0.25,3.75],
                                       x_0ticks = ["Average Epi 2:5"],
                                       points_to_plot = points_to_plot,
                                       Sub_testing = Sub_testing,
                                       Sample_size_for_test = Sample_size_for_test,
                                       Repeating_test = Repeating_test,
                                       pass_stats = true,
                                       ifsave = ifsave)
        push!(ρs, ρ); push!(dρs, dρ)
        mY, dY, MedY, dMedY, Q25_Y, Q75_Y, Y =
            Func_group_based_information(Y_set[i_Y],Goal_type_Set;two_parts = false)
        Title_specifier = string(Legend[i_Y],"_2_Epi25")
        Title = string("Average ratio of time in ",
                        Title_specifier,
                        " states -- in Epi 2-5")
        ρGs_temp = Vector{Tuple{Float64, Float64}}([])
        for i_group = 1:3
                temp = Func_Bar_Plots_specific_oneGroup(i_group, mY[:,2:5], dY[:,2:5], Y[:,2:5],
                                Goal_type_Set,
                                Path_Save, Title_specifier, Title;
                                Colors = Colors,
                                Testing = true, α = 1.1, δα = 0.1,
                                fig_size = (3.5,6), fig_close = false,
                                y_lims = Y_lims[i_Y],
                                points_to_plot = points_to_plot,
                                Sub_testing = Sub_testing,
                                Sample_size_for_test = Sample_size_for_test,
                                Repeating_test = Repeating_test, 
                                pass_stats = true,
                                ifsave = ifsave)
                #
                push!(ρGs_temp, temp)
        end
        push!(ρGs, ρGs_temp)
    end
    return (; μs = μs, dμs = dμs, ρs = ρs, dρs = dρs, ρGs = ρGs)
end
export Func_plot_state_ratio_Epi25


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_plot_state_ratio_Epi1(Data, Goal_type_Set;
                Epi = 1,
                Colors = ["#004D66","#00CCF5"],
                Path_Save = "src/Data_Analysis/Figures/StateRatio_Epi1/",
                Traps = [8,9], Stoch = [7],
                All_states = [1,2,3,4,5,6,7,8,9],
                points_to_plot = -1,
                Sub_testing = false, Scale_std = false,
                Sample_size_for_test = 20,
                Repeating_test = 1000,
                Test_function = OneSampleTTest,
                Bay_Test_function = BIC_OneSampleTTest,
                ifsave = true)
        Sub_Num = length(Data)
        if points_to_plot == -1
            points_to_plot = Sub_Num
        end
        Epi_sum_latency_traps = zeros(Sub_Num,2)
        Epi_sum_latency_stoch = zeros(Sub_Num,2)
        Epi_sum_latency_alls = zeros(Sub_Num,2)

        Options = [[ true, false],
                   [false, true ]]
        for i_opt = 1:2
                temp = Func_desired_states_visit(Data, DesiredStates = Traps,
                                         Epi= Epi,
                                         first_half = Options[i_opt][1],
                                         second_half = Options[i_opt][2])
                Epi_sum_latency_traps[:,i_opt] = sum.(temp[4])

                temp = Func_desired_states_visit(Data, DesiredStates = Stoch,
                                         Epi= Epi,
                                         first_half = Options[i_opt][1],
                                         second_half = Options[i_opt][2])
                Epi_sum_latency_stoch[:,i_opt] = sum.(temp[4])

                temp = Func_desired_states_visit(Data, DesiredStates = All_states,
                                         Epi= Epi,
                                         first_half = Options[i_opt][1],
                                         second_half = Options[i_opt][2])
                Epi_sum_latency_alls[:,i_opt] = sum.(temp[4])
        end

        Y_traps = Epi_sum_latency_traps ./ Epi_sum_latency_alls
        Y_stoch = Epi_sum_latency_stoch ./ Epi_sum_latency_alls
        Y = cat(Y_traps,Y_stoch,dims = 3)

        mY = mean(Y,dims=1)[1,:,:]

        if Sub_testing & Scale_std
                dY = std(Y,dims=1)[1,:,:] ./ sqrt(Sample_size_for_test)
        else
                dY = std(Y,dims=1)[1,:,:] ./ sqrt(Sub_Num)
        end

        # X-axis information
        x_0 = [1,4]
        x_0ticks = ["Traps", "Stoch"]
        σ = 0.2

        fig = figure(figsize=(4,7)); ax = gca()
        for i=1:2
                x = x_0 .+ (i-1)
                ax.bar(x,mY[i,:], color = Colors[i])
        end
        ax.legend(["1st half", "2nd half"])
        for i=1:2
                x = x_0 .+ (i-1)
                ax.errorbar(x,mY[i,:],yerr=dY[i,:],color="k",
                            linewidth=1,drawstyle="steps",linestyle="",capsize=3)
                for j = 1:points_to_plot
                        x_plot = x .+ 2*σ*(rand() - 0.5)
                        ax.plot(x_plot,Y[j,i,:],".k",alpha = 0.5)
                        #ax.plot(x_plot,Y[j,i,:],"k", alpha = 0.1)
                end
        end
        y_min = 0
        y_max = 1.1
        y_pval = 1.0
        y_pval2 = 1.02
        for i = 1:2
        if Sub_testing
                Y_temp = hcat(Y[:,1,i],Y[:,2,i])
                pval = mean(Func_subsamp_test(Y_temp,
                                   Sample_size_for_test,
                                   Repeating_test,
                                   Test_function))

                logBF = mean.(Func_subsamp_test(Y_temp,
                                           Sample_size_for_test,
                                           Repeating_test,
                                           Bay_Test_function,
                                           BayesianTest = true))
        else
                Test_result = Test_function(filter(!isnan,Y[:,1,i]),
                                            filter(!isnan,Y[:,2,i]))
                @show Test_result
                pval = pvalue(Test_result)
                logBF = Bay_Test_function(filter(!isnan,Y[:,1,i]),
                                          filter(!isnan,Y[:,2,i]))
                logBF = [logBF, -logBF]
        end
        x_i = x_0[i]
        x_j = x_0[i] + 1
        ax.plot([x_i,x_j],[1,1] .* y_pval, "k")
        ax.text((x_i+x_j)/2,y_pval2,
                string("p:",Func_pval_string(pval),
                        ", lBF:",Func_logBF_string(logBF[1]),
                        ",",Func_logBF_string(logBF[2])),
                fontsize=4, horizontalalignment="center", rotation=0)
        end
        ax.set_xticks(x_0.+0.5)
        ax.set_xticklabels(x_0ticks)
        ax.set_title("Fraction of time in Epi 1")
        ax.set_ylim([y_min,y_max])
        ax.set_xlim([0,6])

        if ifsave
            savefig(string(Path_Save, "TimeFraction.pdf" ))
            savefig(string(Path_Save, "TimeFraction.svg" ))
        else
            display(fig)
        end

        return (; μ = mY, dμ = dY)
end
export Func_plot_state_ratio_Epi1


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_plot_PPC_comparisons(Y, Color_plot; Path_Save = "",
            Test_function = UnequalVarianceTTest,
            BayTest_function = BIC_EqualVarianceTTest,
            Test_function_const = OneSampleTTest,
            BayTest_function_const = BIC_OneSampleTTest,
            ylim = -1, ylabel = " ", Model_names = 1:3, test_constant = NaN, 
            ifsave = true)
    N_model = length(Y) - 1
    mY = mean.(Y); dY = std.(Y) ./ sqrt.(length.(Y));
    pvals = zeros(N_model); logBFs = zeros(N_model)
    for n = 1:N_model
        if isnan(test_constant)
            Y1_test = Y[1]; Y2_test = Y[1 + n]
            Test_result = Test_function(
                            filter(!isnan,Y1_test), filter(!isnan,Y2_test))
            @show Test_result
            pvals[n] = pvalue(Test_result)
            logBFs[n] = BayTest_function(
                            filter(!isnan,Y1_test), filter(!isnan,Y2_test))
        else
            Y_test = Y[1 + n]
            Test_result = Test_function_const(
                                        filter(!isnan,Y_test) .- test_constant)
            @show Test_result
            pvals[n] = pvalue(Test_result)
            logBFs[n] = BayTest_function_const(
                                        filter(!isnan,Y_test) .- test_constant)
        end
    end
    fig = figure(figsize=(4,6)); ax = gca()
    x = 1:N_model
    for n = 1:N_model
            ax.bar(x[n],mY[n+1], color = Color_plot[n+1])
            ax.scatter(x[n], mY[1], color = Color_plot[1])
            ax.text(x[n],mY[n+1] + dY[n+1] * 1.05,
                    string("p:", Func_pval_string(pvals[n]),
                    ", lBF:", Func_logBF_string(logBFs[n]),
                         ",", Func_logBF_string(-logBFs[n])),
                        fontsize=9, horizontalalignment="center", rotation=0)
    end
    ax.errorbar(x,mY[2:end],yerr=dY[2:end],color="k",
                linewidth=1,drawstyle="steps",linestyle="",capsize=3)
    ax.errorbar(x,mY[1] .* ones(N_model) ,yerr=dY[1] .* ones(N_model),
                color=Color_plot[1], linewidth=1,drawstyle="steps",linestyle="",
                capsize=3)
    ax.set_xlim([0.25,N_model+0.75])
    ax.plot([0.25,N_model+0.75],[0,0],"--k")
    if ylim != -1
        ax.set_ylim(ylim)
    end
    ax.set_xticks(1:N_model); ax.set_xticklabels(Model_names)
    ax.set_ylabel(ylabel)

    if ifsave
        savefig(string(Path_Save,".pdf"))
        savefig(string(Path_Save,".svg"))
    else
        display(fig)
    end
end
export Func_plot_PPC_comparisons

function Func_plot_PPC_comparisons_corr(Y, Color_plot; Path_Save = "",
                                ylim = -1, ylabel = " ", Model_names = 1:3,
                                Bootstrap_size = 1000, ifsave = true)
    N_model = length(Y) - 1
    mY = cor_on_tuples.(Y);
    dY = [stderror(bootstrap(cor_on_tuples, Y[n], 
                  BasicSampling(Bootstrap_size)))[1] for n = 1:length(Y)]
    pvals = zeros(N_model)
    for n = 1:N_model
        Y1_test = Y[1]; Y2_test = Y[1 + n]
        xy, rx, ry = HypothesisTests.ptstats(Y1_test, Y2_test)
        perm_samples = [(shuffle!(xy); 
                        cor_on_tuples(view(xy,rx)) - 
                        cor_on_tuples(view(xy,ry))) for i = 1:Bootstrap_size]
        Test_result = HypothesisTests.PermutationTest(
                        cor_on_tuples(Y1_test) - cor_on_tuples(Y2_test), 
                        perm_samples)
        @show Test_result
        pvals[n] = pvalue(Test_result)
    end
    fig = figure(figsize=(4,6)); ax = gca()
    x = 1:N_model
    for n = 1:N_model
        ax.bar(x[n],mY[n+1], color = Color_plot[n+1])
        ax.scatter(x[n], mY[1], color = Color_plot[1])
        ax.text(x[n],mY[n+1] + dY[n+1] * 1.05,
                string("p:", Func_pval_string(pvals[n])),
                    fontsize=9, horizontalalignment="center", rotation=0)
    end
    ax.errorbar(x,mY[2:end],yerr=dY[2:end],color="k",
            linewidth=1,drawstyle="steps",linestyle="",capsize=3)
    ax.errorbar(x,mY[1] .* ones(N_model) ,yerr=dY[1] .* ones(N_model),
            color=Color_plot[1], linewidth=1,drawstyle="steps",linestyle="",
            capsize=3)
    ax.set_xlim([0.25,N_model+0.75])
    ax.plot([0.25,N_model+0.75],[0,0],"--k")
    if ylim != -1
        ax.set_ylim(ylim)
    end
    ax.set_xticks(1:N_model); ax.set_xticklabels(Model_names)
    ax.set_ylabel(ylabel)

    if ifsave
        savefig(string(Path_Save,".pdf"))
        savefig(string(Path_Save,".svg"))
    else
        display(fig)
    end
end
export Func_plot_PPC_comparisons_corr

function cor_on_tuples(Y)
    cor([Y[i][1] for i = 1:length(Y)], [Y[i][2] for i = 1:length(Y)])
end
export cor_on_tuples

function Func_plot_PPC_comparisons_med(Y, Color_plot; Path_Save = "",
    ylim = -1, ylabel = " ", Model_names = 1:3, Bootstrap_size = 1000, 
    ifsave = true)
    N_model = length(Y) - 1
    mY = median.(Y); 
    dY = [stderror(bootstrap(median, Y[n], 
            BasicSampling(Bootstrap_size)))[1] for n = 1:length(Y)]
    pvals = zeros(N_model);
    for n = 1:N_model
        Y1_test = Y[1]; Y2_test = Y[1 + n]        
        Test_result = ApproximatePermutationTest(
                filter(!isnan,Y1_test), filter(!isnan,Y2_test),
                median, Bootstrap_size)
        @show Test_result
        pvals[n] = pvalue(Test_result)
    end
    fig = figure(figsize=(4,6)); ax = gca()
    x = 1:N_model
    for n = 1:N_model
        ax.bar(x[n],mY[n+1], color = Color_plot[n+1])
        ax.scatter(x[n], mY[1], color = Color_plot[1])
        ax.text(x[n],mY[n+1] + dY[n+1] * 1.05,
                string("p:", Func_pval_string(pvals[n])),
                fontsize=9, horizontalalignment="center", rotation=0)
    end
    ax.errorbar(x,mY[2:end],yerr=dY[2:end],color="k",
            linewidth=1,drawstyle="steps",linestyle="",capsize=3)
    ax.errorbar(x,mY[1] .* ones(N_model) ,yerr=dY[1] .* ones(N_model),
            color=Color_plot[1], linewidth=1,drawstyle="steps",linestyle="",
            capsize=3)
    ax.set_xlim([0.25,N_model+0.75])
    ax.plot([0.25,N_model+0.75],[0,0],"--k")
    if ylim != -1
        ax.set_ylim(ylim)
    end
    ax.set_xticks(1:N_model); ax.set_xticklabels(Model_names)
    ax.set_ylabel(ylabel)

    if ifsave
        savefig(string(Path_Save,".pdf"))
        savefig(string(Path_Save,".svg"))
    else
        display(fig)
    end
end
export Func_plot_PPC_comparisons_med
