# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# pvalue converting
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_pval_string(pval)
    if pval < 1e-4
        return "< 1e-4"
    elseif pval < 1e-3
        return "< 1e-3"
    else
        return string(round(pval, digits = 3))
    end
end
export Func_pval_string


function Func_logBF_string(logBF)
    return string(round(logBF, digits = 2))
end
export Func_logBF_string

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Plotting bar plots for different episodes
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_Bar_Plots_specific(mY, dY, Y, Goal_type_Set,
                                Path_Save, Title_specifier, Title;
                                Colors = ["#004D66","#0350B5","#00CCF5"], σ = 0.2,
                                Test_function_1 = UnequalVarianceTTest,
                                Test_function_2 = OneSampleTTest,
                                Bay_Test_function_1 = BIC_EqualVarianceTTest,
                                Bay_Test_function_2 = BIC_OneSampleTTest,
                                Testing = false, α = 1.1, δα = 0.1,
                                fig_size = (12,7), fig_close = true,
                                y_lims = -1, points_to_plot = -1,
                                Sub_testing = false, Scale_std = false,
                                Sample_size_for_test = 20,
                                Repeating_test = 1000,
                                swap_epis = false, ifsave = true)
    # X-axis information
    x_0 = (1:4:(1 + (size(mY)[2]-1)*4))
    if size(mY)[2] == 2
        x_0ticks = ["Epi 1", "Average Epi 2:5"]
        if swap_epis
            x_0ticks = x_0ticks[[2,1]]
            mY = mY[:,[2,1]]
            dY = dY[:,[2,1]]
            Y = Y[:,[2,1]]
        end
    else
        x_0ticks = ["Epi 1", "Epi 2", "Epi 3", "Epi 4", "Epi 5"]
    end

    # --------------------------------------------------------------------------
    # Figure for averages
    # --------------------------------------------------------------------------
    fig = figure(figsize=fig_size); ax = gca()
    for i=1:3
        x = x_0 .+ (i-2)
        ax.bar(x,mY[i,:],color=Colors[i])
    end
    ax.legend(["CHF2","CHF3","CHF4"])
    if Sub_testing & Scale_std
        dY = dY .* sqrt(length(dY[:,1]) / Sample_size_for_test)
    end
    for i=1:3
        x = x_0 .+ (i-2)
        ax.errorbar(x,mY[i,:],yerr=dY[i,:],color="k",
                    linewidth=1,drawstyle="steps",linestyle="",capsize=3)
        inds = 1:length(Goal_type_Set)
        inds = inds[Goal_type_Set .== (i-1)]
        if points_to_plot != -1
            inds = inds[1:points_to_plot]
        end
        for j = inds
            ax.plot(x .+ 2*σ*(rand() - 0.5),Y[j,:],".k",alpha = 0.5)
        end
    end
    y_bar_max = findmax(filter(!isnan,mY .+ dY))[1]
    y_ax = ax.get_ylim()[2]
    if Testing
        if size(mY)[2] == 2
            # Testing between groups differences for each phase of experiment
            for part = 1:2
                for i = 1:3
                    for j = (i+1):3
                        Y1_test = Y[Goal_type_Set .== (i-1),part]
                        Y2_test = Y[Goal_type_Set .== (j-1),part]
                        if Sub_testing
                            pval = mean(Func_subsamp_test(Y1_test,Y2_test,
                                                       Sample_size_for_test,
                                                       Sample_size_for_test,
                                                       Repeating_test,
                                                       Test_function_1))
                            logBF = mean.(Func_subsamp_test(Y1_test,Y2_test,
                                                    Sample_size_for_test,
                                                    Sample_size_for_test,
                                                    Repeating_test,
                                                    Bay_Test_function_1,
                                                    BayesianTest = true))
                        else
                            Test_result = Test_function_1(
                                            filter(!isnan,Y1_test),
                                            filter(!isnan,Y2_test))
                            @show Test_result
                            pval = pvalue(Test_result)
                            logBF = Bay_Test_function_1(
                                            filter(!isnan,Y1_test),
                                            filter(!isnan,Y2_test))
                            logBF = [logBF, -logBF]
                        end
                        if (j-i) < 2
                            y_p_temp = α*y_bar_max
                        else
                            y_p_temp = α*y_bar_max + δα*y_ax
                        end
                        x_i = x_0[part] + (i-2)
                        x_j = x_0[part] + (j-2)
                        ax.plot([x_i,x_j],[1,1] .* y_p_temp, "k")
                        ax.text((x_i+x_j)/2,y_p_temp+δα*y_ax/4,
                                string("p:",Func_pval_string(pval),
                                    ", lBF:",Func_logBF_string(logBF[1]),
                                         ",",Func_logBF_string(logBF[2])),
                                fontsize=4, horizontalalignment="center", rotation=0)
                    end
                end
            end
            # Testing between phase differences for each group
            y_p2 = α*y_bar_max + 2*δα*y_ax
            for i = 1:3
                Y1_test = Y[Goal_type_Set .== (i-1),1]
                Y2_test = Y[Goal_type_Set .== (i-1),2]
                if Sub_testing
                    pval = mean(Func_subsamp_test(Y1_test,Y2_test,
                                               Sample_size_for_test,
                                               Sample_size_for_test,
                                               Repeating_test,
                                               Test_function_2))
                    logBF = mean.(Func_subsamp_test(Y1_test,Y2_test,
                                            Sample_size_for_test,
                                            Sample_size_for_test,
                                            Repeating_test,
                                            Bay_Test_function_2,
                                            BayesianTest = true))
                else
                    Test_result = Test_function_2(
                                    filter(!isnan,Y1_test),
                                    filter(!isnan,Y2_test))
                    @show Test_result
                    pval = pvalue(Test_result)
                    logBF = Bay_Test_function_2(
                                    filter(!isnan,Y1_test),
                                    filter(!isnan,Y2_test))
                    logBF = [logBF, -logBF]
                end
                y_p_temp = y_p2 + (3-i)*δα*y_ax
                x_i = x_0[1] + (i-2)
                x_j = x_0[2] + (i-2)
                ax.plot([x_i,x_j],[1,1] .* y_p_temp, "k")
                ax.text((x_i+x_j)/2,y_p_temp+δα*y_ax/4,
                        string("p:",Func_pval_string(pval),
                            ", lBF:",Func_logBF_string(logBF[1]),
                                 ",",Func_logBF_string(logBF[2])),
                        fontsize=4, horizontalalignment="center", rotation=0)
            end
        elseif size(mY)[2] > 2
            # Testing correlation for episode 2:5
            y_p1 = α*y_bar_max + 2*δα*y_ax
            for i = 1:3
                x = x_0 .+ (i-2)
                Y_corr = Y[Goal_type_Set .== (i-1),2:end]
                mY_corr = mY[i,2:end]
                x_corr = Float64.(x[2:end])
                X_corr = cat(ones(length(x_corr)),x_corr,dims=2)
                ρs = zeros(size(Y_corr)[1])
                βs = zeros(size(Y_corr)[1],2)
                for n = 1:length(ρs)
                    ρs[n] = cor(x_corr,Y_corr[n,:])
                    βs[n,:] = inv(X_corr' * X_corr) * X_corr' * Y_corr[n,:]
                end
                β_hat = mean(βs,dims=1)[:]
                Y_plot = X_corr * β_hat

                ρs = ρs[isnan.(ρs) .== 0]
                ρ = mean(ρs)
                dρ = std(ρs) / sqrt(length(ρs))
                if Sub_testing
                    pval = mean(Func_subsamp_test(ρs,
                                               Sample_size_for_test,
                                               Repeating_test,
                                               OneSampleTTest))
                    logBF = mean.(Func_subsamp_test(ρs,
                                               Sample_size_for_test,
                                               Repeating_test,
                                               BIC_OneSampleTTest,
                                               BayesianTest = true))
                else
                    Test_result = OneSampleTTest(ρs)
                    @show Test_result
                    pval = pvalue(Test_result)
                    logBF = [BIC_OneSampleTTest(ρs), -BIC_OneSampleTTest(ρs)]
                end
                ax.plot(X_corr[:,2],Y_plot,"--",color=Colors[i])
                ax.text(X_corr[i,2],α*y_bar_max,
                    string("p:",Func_pval_string(pval),
                        ", lBF:",Func_logBF_string(logBF[1]),
                             ",",Func_logBF_string(logBF[2])),
                    fontsize=4, horizontalalignment="center", rotation=0,
                    color=Colors[i])
                ax.text(X_corr[i,2],α*y_bar_max+δα*y_ax/2,
                        string("r=",string(round(ρ,digits=3)),"+-",string(round(dρ,digits=3))),
                        fontsize=4, horizontalalignment="center", rotation=0,
                        color=Colors[i])
            end
        end
    end
    ax.set_xticks(x_0)
    ax.set_xticklabels(x_0ticks)
    ax.set_title(Title)
    if y_lims != -1
        ax.set_ylim(y_lims)
    end
    if ifsave
        savefig(string(Path_Save, "Epi_", Title_specifier, "_barplot.pdf" ))
        savefig(string(Path_Save, "Epi_", Title_specifier, "_barplot.svg" ))
    else
        display(fig)
    end
    if fig_close
        close(fig)
    else
        return fig, ax
    end
end
export Func_Bar_Plots_specific


function Func_Bar_Plots_specific_oneGroup(Group_id, mY, dY, Y, Goal_type_Set,
                                   Path_Save, Title_specifier, Title;
                                   Colors = ["#004D66","#0350B5","#00CCF5"], σ = 0.2,
                                   Testing = false, α = 1.1, δα = 0.1,
                                   fig_size = (12,7), fig_close = true,
                                   y_lims = -1, points_to_plot = -1,
                                   Sub_testing = false, Scale_std=false,
                                   Sample_size_for_test = 20,
                                   Repeating_test = 1000,
                                   swap_epis = false,
                                   x_0 = 1:4,
                                   x_0ticks = ["Epi 2", "Epi 3", "Epi 4", "Epi 5"],
                                   Legends = ["CHF2","CHF3","CHF4"],
                                   pass_stats = false, ifsave = true)
    # --------------------------------------------------------------------------
    # Figure for averages
    # --------------------------------------------------------------------------
    fig = figure(figsize=fig_size); ax = gca()
    x = x_0 .+ 0
    ax.bar(x,mY[Group_id,:],color=Colors[Group_id])
    if Sub_testing & Scale_std
        dY = dY .* sqrt(length(dY[:,1]) / Sample_size_for_test)
    end

    x = x_0 .+ 0
    ax.errorbar(x,mY[Group_id,:],yerr=dY[Group_id,:],color="k",
                linewidth=1,drawstyle="steps",linestyle="",capsize=3)
    inds = 1:length(Goal_type_Set)
    inds = inds[Goal_type_Set .== (Group_id-1)]
    if points_to_plot != -1
        inds = inds[1:points_to_plot]
    end
    for j = inds
        x_plot = x .+ 2*σ*(rand() - 0.5)
        ax.plot(x_plot,Y[j,:],".k",alpha = 0.5)
        ax.plot(x_plot,Y[j,:],"k" ,alpha = 0.1)
    end
    y_bar_max = findmax(filter(!isnan,mY .+ dY))[1]
    y_ax = ax.get_ylim()[2]
    if Testing
        # Testing correlation for episode 2:5
        x = x_0 .+ 0
        Y_corr = Y[Goal_type_Set .== (Group_id-1),:]
        x_corr = Float64.(x[:])
        X_corr = cat(ones(length(x_corr)),x_corr,dims=2)
        ρs = zeros(size(Y_corr)[1])
        βs = zeros(size(Y_corr)[1],2)
        for n = eachindex(ρs)
            ρs[n] = cor(x_corr,Y_corr[n,:])
            βs[n,:] = inv(X_corr' * X_corr) * X_corr' * Y_corr[n,:]
        end
        β_hat = mean(βs,dims=1)[:]
        Y_plot = X_corr * β_hat

        ρs = ρs[isnan.(ρs) .== 0]
        ρ = mean(ρs)
        dρ = std(ρs) / sqrt(length(ρs))
        if Sub_testing
            pval = mean(Func_subsamp_test(ρs,
                                        Sample_size_for_test,
                                        Repeating_test,
                                        OneSampleTTest))
            logBF = mean.(Func_subsamp_test(ρs,
                                    Sample_size_for_test,
                                    Repeating_test,
                                    BIC_OneSampleTTest,
                                    BayesianTest = true))
        else
            Test_result = OneSampleTTest(ρs)
            @show Test_result
            pval = pvalue(Test_result)
            logBF = [BIC_OneSampleTTest(ρs), -BIC_OneSampleTTest(ρs)]
        end
        ax.plot(X_corr[:,2],Y_plot,"--",color="k")
        ax.text(X_corr[Group_id,2],α*y_bar_max,
            string("p:",Func_pval_string(pval),
                ", lBF:",Func_logBF_string(logBF[1]),
                        ",",Func_logBF_string(logBF[2])),
            fontsize=4, horizontalalignment="center", rotation=0,
            color="k")
        ax.text(X_corr[Group_id,2],α*y_bar_max+δα*y_ax/2,
                string("r=",string(round(ρ,digits=3)),"+-",string(round(dρ,digits=3))),
                fontsize=4, horizontalalignment="center", rotation=0,
                color="k")
    end
    ax.set_xticks(x_0)
    ax.set_xticklabels(x_0ticks)
    ax.set_title(string(Title, Legends[Group_id]))
    if y_lims != -1
        ax.set_ylim(y_lims)
    end
    if ifsave
        savefig(string(Path_Save, "Epi_", Title_specifier, "_barplot_G", string(Group_id), ".pdf" ))
        savefig(string(Path_Save, "Epi_", Title_specifier, "_barplot_G", string(Group_id), ".svg" ))
    else
        display(fig)
    end
    if fig_close
        close(fig)
    elseif pass_stats
        return ρ, dρ
    else
        return fig, ax
    end
end
export Func_Bar_Plots_specific_oneGroup

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Plotting bar plots for different episodes
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_Bar_Plots_linear_specific(mY, dY, Y, Goal_type_Set,
                                        Path_Save, Title_specifier, Title;
                                        Colors = ["#004D66","#0350B5","#00CCF5"], σ = 0.2,
                                        Testing = true,
                                        α = 1.1, δα = 0.1,
                                        fig_size = (4,6), fig_close = true,
                                        y_lims = -1, x_lims = [0.5,3.5],
                                        points_to_plot = -1,
                                        x_0ticks = ["Average Epi 2:5"],
                                        Spearman_test=false,
                                        Sub_testing = false, Scale_std=false,
                                        Sample_size_for_test = 20,
                                        Repeating_test = 1000,
                                        Testing_compare_const = false,
                                        Testing_constant = 0.5,
                                        Bootstrap_size = 1000,
                                        pass_stats = false, ifsave = true)
    # --------------------------------------------------------------------------
    # X-axis information
    # --------------------------------------------------------------------------
    x_0 = [2]

    # --------------------------------------------------------------------------
    # Figure for averages
    # --------------------------------------------------------------------------
    fig = figure(figsize=fig_size); ax = gca()
    for i=1:3
        x = x_0 .+ (i-2)
        ax.bar(x,mY[[i]],color=Colors[i])
    end
    y_bar_max = findmax(filter(!isnan,mY .+ dY))[1]
    y_ax = ax.get_ylim()[2]

    ax.legend(["CHF2","CHF3","CHF4"])
    for i=1:3
        x = x_0 .+ (i-2)
        if Sub_testing & Scale_std
            dY[[i]] = dY[[i]] .* sqrt(length(dY[[i]]) / Sample_size_for_test)
        end
        ax.errorbar(x,mY[[i]],yerr=dY[[i]],color="k",
                    linewidth=1,drawstyle="steps",linestyle="",capsize=3)
        inds = 1:length(Goal_type_Set)
        inds = inds[Goal_type_Set .== (i-1)]
        if points_to_plot != -1
            inds = inds[1:points_to_plot]
        end
        for j = inds
            ax.plot(x .+ 2*σ*(rand() - 0.5),Y[[j]],".k",alpha = 0.5)
        end
        if Testing_compare_const
            y_test_temp = Y[inds]
            if Sub_testing
                Test_function_temp(x_arg) = OneSampleTTest(x_arg, Testing_constant)
                pval = mean(Func_subsamp_test(y_test_temp,
                                        Sample_size_for_test,
                                        Repeating_test,
                                        Test_function_temp))
                #
                logBF = mean.(Func_subsamp_test(y_test_temp .- Testing_constant,
                                        Sample_size_for_test,
                                        Repeating_test,
                                        BIC_OneSampleTTest,
                                        BayesianTest = true))
            else
                Test_result = OneSampleTTest(y_test_temp, Testing_constant)
                @show Test_result
                pval = pvalue(Test_result)
                logBF = [BIC_OneSampleTTest(y_test_temp .- Testing_constant),
                        -BIC_OneSampleTTest(y_test_temp .- Testing_constant)]

            end
            ax.text(x, α*y_bar_max + 2*δα*y_ax,
                string("p:",Func_pval_string(pval),
                    ", lBF:",Func_logBF_string(logBF[1]),
                         ",",Func_logBF_string(logBF[2])),
                fontsize=4, horizontalalignment="center", rotation=0)

        end
    end

    if Testing
        y_corr = Y[:]
        x_corr = Float64.(Goal_type_Set[:] .+ 1)
        if Spearman_test
            # mat"""
            # [$rho_spear,$pval_spear] = corr($x_corr,$y_corr,'Type','Spearman')
            # """
            error("No Spearman correlation availabe")
            ρ = rho_spear
            pval = pval_spear
        else
            ρ = cor(x_corr,y_corr)
            dρ = stderror(bootstrap(cor_on_tuples, 
                            [(x_corr[n],y_corr[n]) for n = 1:length(x_corr)], 
                            BasicSampling(Bootstrap_size)))[1]
            if Sub_testing
                x_corr_temp =  [x_corr[Goal_type_Set[:] .==0],
                                x_corr[Goal_type_Set[:] .==1],
                                x_corr[Goal_type_Set[:] .==2]]
                y_corr_temp =  [y_corr[Goal_type_Set[:] .==0],
                                y_corr[Goal_type_Set[:] .==1],
                                y_corr[Goal_type_Set[:] .==2]]
                pval = mean(Func_subsamp_test(x_corr_temp,
                                              y_corr_temp,
                                              Sample_size_for_test,
                                              Sample_size_for_test,
                                              Repeating_test,
                                              CorrelationTest))

                logBF = mean.(Func_subsamp_test(x_corr_temp,
                                      y_corr_temp,
                                      Sample_size_for_test,
                                      Sample_size_for_test,
                                      Repeating_test,
                                      BIC_CorrelationTest,
                                      BayesianTest = true))
            else
                Test_result = CorrelationTest(x_corr,y_corr)
                @show Test_result
                pval = pvalue(Test_result)
                logBF = [BIC_CorrelationTest(x_corr,y_corr),
                        -BIC_CorrelationTest(x_corr,y_corr)]
            end
        end
        X_corr = cat(ones(length(x_corr)),x_corr,dims=2)
        β_hat = inv(X_corr' * X_corr) * X_corr' * Y
        X_plot = cat(ones(5),0:4,dims=2)
        Y_plot = X_plot * β_hat
        ax.plot(X_plot[:,2],Y_plot,"--k")
        ax.text(3,α*y_bar_max,
            string("p:",Func_pval_string(pval),
                ", lBF:",Func_logBF_string(logBF[1]),
                     ",",Func_logBF_string(logBF[2])),
            fontsize=4, horizontalalignment="center", rotation=0)

        ax.text(3,α*y_bar_max+δα*y_ax,
                string("r=",string(round(ρ,digits=3)), 
                       " +- ", string(round(dρ,digits=3))),
                fontsize=4, horizontalalignment="center", rotation=0)
    end

    ax.set_xticks(x_0)
    ax.set_xticklabels(x_0ticks)
    ax.set_title(Title)
    ax.set_xlim(x_lims)
    if y_lims != -1
        ax.set_ylim(y_lims)
    end
    if ifsave
        savefig(string(Path_Save, Title_specifier, "_barplot_linear.pdf" ))
        savefig(string(Path_Save, Title_specifier, "_barplot_linear.svg" ))
    else
        display(fig)
    end
    if fig_close
        close(fig)
    elseif pass_stats
        return ρ, dρ
    else
        return fig, ax
    end
end
export Func_Bar_Plots_linear_specific

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Plotting parameter plots
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_Parameter_dots(X, Names, Sets;
                             figsize = (5,5),
                             digit_plot=3, Save_path = "", Save_path2 = "",
                             lower_bounds = -1, upper_bounds = -1,
                             Log_scale_ind = -1, Set_Legends = -1,
                             Set_unique = -1,σ_plot = 0.05)
    N_var = size(X)[2]
    # default values
    if lower_bounds == -1
        lower_bounds = zeros(N_var)
    end
    if upper_bounds == -1
        upper_bounds = zeros(N_var)
        for i = 1:N_var
            upper_bounds[i] = findmax(X[:,i])[1]
        end
    end
    if Log_scale_ind == -1
        Log_scale_ind = (ones(20) .== 0)
    end
    if Set_Legends == -1
        Set_unique = unique(Sets)
        Set_Legends = string.(Set_unique)
    end

    # plotting
    for i = 1:N_var
        #ax = subplot(floor(sqrt(N_var)),ceil(N_var / floor(sqrt(N_var))), i)
        fig = figure(figsize=figsize)
        ax = subplot(1,1,1)
        y = X[:,i]
        μ = string(round(mean(y),digits=digit_plot))
        σ = string(round(std(y),digits=digit_plot))
        if Log_scale_ind[i]
            ax.set_xscale("log")
            if findmin(y)[1]>0
                lower_bounds[i] = min(lower_bounds[i], findmin(y)[1])
            end
        end
        ax.set_xlim(lower_bounds[i],upper_bounds[i])
        ax.plot(y, Sets .+ (σ_plot .* randn(length(y))), ".k", alpha=0.5)
        ax.legend([string("μ=", μ, ", σ=",σ)])
        ax.set_title(Names[i])
        ax.set_yticks(Set_unique)
        ax.set_yticklabels(Set_Legends)
        savefig(string(Save_path, "dot_plots_", string(i), Save_path2, ".pdf"))
        savefig(string(Save_path, "dot_plots_", string(i), Save_path2, ".svg"))
        close(fig)
    end
end
export Func_Parameter_dots

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Plotting parameter plots
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_Parameter_boxplot(X, Names, Sets;
                                 figsize = (5,5),
                                 Save_path = "", Save_path2 = "",
                                 lower_bounds = -1, upper_bounds = -1,
                                 Set_unique = -1, Set_Legends = -1,
                                 All_in_one = false,
                                 Log_scale_ind = -1)
    N_var = size(X)[2]
    # default values
    if lower_bounds == -1
        lower_bounds = zeros(N_var)
    end
    if upper_bounds == -1
        upper_bounds = zeros(N_var)
        for i = 1:N_var
            upper_bounds[i] = findmax(X[:,i])[1]
        end
    end
    if Set_Legends == -1
        Set_unique = unique(Sets)
        Set_Legends = string.(Set_unique)
    end
    if Log_scale_ind == -1
        Log_scale_ind = (ones(22) .== 0)
    end

    # plotting
    for i = 1:N_var
        if All_in_one
            if i == 1
                fig = figure(figsize=figsize)
            end
            ax = subplot(Int64(floor(sqrt(N_var))),
                         Int64(ceil(N_var / floor(sqrt(N_var)))),i)
        else
            fig = figure(figsize=figsize)
            ax = subplot(1,1,1)
        end
        y = []
        for j = 1:length(Set_unique)
            push!(y, X[Sets .== Set_unique[j] ,i])
        end
        ax.boxplot(y)
        ax.scatter(1:length(y), mean.(y))
        ax.set_title(Names[i])
        ax.set_xticks(Set_unique .+ 1)
        ax.set_xticklabels(Set_Legends)
        if Log_scale_ind[i]
            ax.set_yscale("log")
            y_min = findmin(X[:,i])[1]
            if y_min>0
                lower_bounds[i] = min(lower_bounds[i], y_min)
            end
        end
        ax.set_ylim(lower_bounds[i],upper_bounds[i])
        if All_in_one
        else
            savefig(string(Save_path, "box_plots_", string(i), Save_path2, ".pdf"))
            savefig(string(Save_path, "box_plots_", string(i), Save_path2, ".svg"))
            close(fig)
        end
    end
    if All_in_one
        tight_layout()
        savefig(string(Save_path, "box_plots_All_in_one", Save_path2, ".pdf"))
        savefig(string(Save_path, "box_plots_All_in_one", Save_path2, ".svg"))
    end

end
export Func_Parameter_boxplot


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Plotting heatmaps
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Data_PlotHeatMap(Y_plot, ax; X_names = "", TextVal = true,
                                cmap="RdBu",vmin=-1,vmax=1.,
                                rotationX=45, rotationY=45)
        if isnan(vmin) || isnan(vmax)
                cp = ax.imshow(Y_plot,cmap=cmap)
        else
                cp = ax.imshow(Y_plot,cmap=cmap, vmin=vmin,vmax=vmax)
        end
        if TextVal
                for i = 1:size(Y_plot)[1]
                for j = 1:size(Y_plot)[2]
                        ax.text(j - 1, i - 1, 
                        string(round(Y_plot[i,j],digits=2)),
                        fontsize=5, horizontalalignment="center", color = "w")
                end
                end
        end
        if X_names != ""
                ax.set_xticks(0:(size(Y_plot)[1]-1)); 
                ax.set_xticklabels(X_names,fontsize=9,rotation=rotationX)
                ax.set_yticks(0:(size(Y_plot)[1]-1)); 
                ax.set_yticklabels(X_names,fontsize=9,rotation=rotationY)
        end
        return cp                  
end
export Data_PlotHeatMap

