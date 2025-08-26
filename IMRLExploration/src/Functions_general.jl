# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Moving Average
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_movmean(x,k;ZeroPadding = false, med = false,
                            quant = false, quant_α = 0.5)
    n = length(x)
    y = zeros(n)
    if ZeroPadding
        x = append!(zeros(k),x)
        x = append!(x,zeros(k))
        inds = 0:(2*k)
        for i = 1:n
            y[i] = mean(x[i .+ inds])
        end
    else
        for i = 1:n
            if i <= k
                ind_beg = 1
            else
                ind_beg = i - k
            end
            if n <= i+k
                ind_end = n
            else
                ind_end = i + k
            end
            if med
                y[i] = median(x[ind_beg:ind_end])
            elseif quant
                y[i] = quantile(x[ind_beg:ind_end], quant_α)
            else
                y[i] = mean(x[ind_beg:ind_end])
            end
        end
    end
    return y
end
export Func_movmean


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Moving Average
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_disc_hist_seq(x,S)
    counts = zeros(length(S))
    for i = eachindex(S)
        counts[i] = sum(x .== S[i])
    end
    return counts
end
export Func_disc_hist_seq


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# FDR control
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function FDR_control_pval(p_values;FDR=0.1)
    temp = sort(p_values)
    inds = Array(1:length(temp))
    pval_thresh = inds .* FDR / length(temp)
    if sum(temp .< pval_thresh) > 0
        pval_thresh = pval_thresh[findmax(inds[temp .< pval_thresh])[1]]
    else
        pval_thresh = 0
    end
    R0 = p_values .< pval_thresh
    if sum(R0)>0
        argR0 = inds[R0]
    else
        argR0 = []
    end
    return R0, argR0, pval_thresh
end
export FDR_control_pval

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Q values to action probabilities with softmax
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_agent_Q_to_π(Q::Array{Float64,2})
    π_A = zeros(size(Q))

    for s = 1:size(Q)[1]
        Q_temp = Q[s,:] .- findmax(@view Q[s,:])[1]
        π_A_temp = exp.(Q_temp)
        π_A_temp = π_A_temp ./ sum(π_A_temp)
        if sum(isnan.(π_A_temp))==1
            π_A_temp = isnan.(π_A_temp) .* 1.0
        end
        π_A[s,:] .= π_A_temp
    end
    return π_A
end
export Func_agent_Q_to_π


# # ------------------------------------------------------------------------------
# # ------------------------------------------------------------------------------
# # T to trial numbers
# # ------------------------------------------------------------------------------
# # ------------------------------------------------------------------------------
# function Func_Ts2trial_time(Ts::Array{Int64,1})
#     N_epi = length(Ts)
#     trial_time = Array{Array{Int64,1},1}(undef,N_epi)
#     for n = 1:N_epi
#         trial_time[n] = 1:Ts[n]
#     end
#     return trial_time
# end
# export Func_Ts2trial_time


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function BIC_OneSampleTTest(Y::Array{Float64,1})
    N = length(Y)

    σ_0 = sqrt(mean(Y.^2))
    K_0 = 1; P_0 = Normal(0,σ_0);

    μ_1 = mean(Y); σ_1 = sqrt(mean((Y .- μ_1).^2))
    K_1 = 2; P_1 = Normal(μ_1,σ_1)

    log_p0 = sum(logpdf.(P_0,Y)) - (K_0 * log(N) / 2)
    log_p1 = sum(logpdf.(P_1,Y)) - (K_1 * log(N) / 2)

    logBF = log_p1 - log_p0
end
function BIC_OneSampleTTest(Y1::Array{Float64,1},Y2::Array{Float64,1})
    BIC_OneSampleTTest(Y1 .- Y2)
end
export BIC_OneSampleTTest

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function BIC_EqualVarianceTTest(X::Array{Float64,1},Y::Array{Float64,1})
    N_X = length(X)
    N_Y = length(Y)
    N = N_X + N_Y
    
    μ_0 = mean(vcat(X,Y));
    σ_0 = sqrt(mean((vcat(X,Y) .- μ_0).^2));
    K_XY0 = 2; K_X0 = 0; K_Y0 = 0;
    P_X0 = Normal(μ_0,σ_0); P_Y0 = Normal(μ_0,σ_0);

    μ_X1 = mean(X); μ_Y1 = mean(Y);
    σ_1 = sqrt(mean(vcat((X .- μ_X1), (Y .- μ_Y1)).^2));
    K_XY1 = 1; K_X1 = 1; K_Y1 = 1;
    P_X1 = Normal(μ_X1,σ_1); P_Y1 = Normal(μ_Y1,σ_1);

    log_p0 = sum(logpdf.(P_X0,X)) + sum(logpdf.(P_Y0,Y)) -
            (K_X0 * log(N_X) / 2) - (K_Y0 * log(N_Y) / 2) -
            (K_XY0 * log(N_Y + N_X) / 2)

    log_p1 = sum(logpdf.(P_X1,X)) + sum(logpdf.(P_Y1,Y)) -
            (K_X1 * log(N_X) / 2) - (K_Y1 * log(N_Y) / 2) -
            (K_XY1 * log(N_Y + N_X) / 2)

    logBF = log_p1 - log_p0
end
export BIC_EqualVarianceTTest
BIC_UnequalVarianceTTest(X::Array{Float64,1},Y::Array{Float64,1}) = 
    BIC_EqualVarianceTTest(X::Array{Float64,1},Y::Array{Float64,1})
export BIC_UnequalVarianceTTest
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function BIC_CorrelationTest(X::Array{Float64,1},Y::Array{Float64,1})
    if length(X) != length(Y)
        error("X must have the same dimension as Y.")
    end
    N = length(Y)

    #X_0 = ones(N); Y_hat_0 = X_0 * inv(X_0' * X_0) * X_0' * Y;
    Y_hat_0 = mean(Y); Δ_0 = Y .- Y_hat_0
    σ_0 = sqrt(mean(Δ_0.^2))
    K_0 = 2; P_0 = Normal(0,σ_0);

    X_1 = hcat(ones(N),X); Y_hat_1 = X_1 * inv(X_1' * X_1) * X_1' * Y;
    Δ_1 = Y .- Y_hat_1; σ_1 = sqrt(mean(Δ_1.^2))
    K_1 = 3; P_1 = Normal(0,σ_1)

    log_p0 = sum(logpdf.(P_0,Δ_0)) - (K_0 * log(N) / 2)
    log_p1 = sum(logpdf.(P_1,Δ_1)) - (K_1 * log(N) / 2)

    logBF = log_p1 - log_p0
end
export BIC_CorrelationTest

# # ------------------------------------------------------------------------------
# # ------------------------------------------------------------------------------
# # ------------------------------------------------------------------------------
# # ------------------------------------------------------------------------------
# function Func_logit(u::Float64)
#     log(u / (1-u))
# end
# export Func_logit

# function Func_inv_logit(u::Float64)
#     1 / (1 + exp(-u))
# end
# export Func_inv_logit


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# WARNING: To-be-checked :WARNING
# ------------------------------------------------------------------------------
function Func_subsamp_test(x::Array{Float64,1}, N_subsamp::Int64,
                           N_repeat::Int64, Test_function;
                           p_thresh = 0.05, logBF_thresh = log(3),
                           pass_values = false,
                           BayesianTest = false)
    N_samp = length(x)
    if (N_subsamp > N_samp)
        error("Number of subsamples > number of real samples")
    end
    stat_vec = zeros(N_repeat)
    for i = 1:N_repeat
        inds = shuffle(1:N_samp)
        x_temp = x[inds[1:N_subsamp]]
        if BayesianTest
            stat_vec[i] = Test_function(filter(!isnan,x_temp))
        else
            stat_vec[i] = pvalue(Test_function(filter(!isnan,x_temp)))
        end
    end
    if pass_values
        return stat_vec
    else
        if BayesianTest
            return [1 .- (stat_vec .> logBF_thresh),
                    1 .- (stat_vec .< -logBF_thresh)]
        else
            return 1 .- (stat_vec .< p_thresh)
        end
    end
end
function Func_subsamp_test(x1::Array{Float64,1}, x2::Array{Float64,1},
                           N1_subsamp::Int64, N2_subsamp::Int64,
                           N_repeat::Int64, Test_function;
                           same_ind = false,
                           p_thresh = 0.05, logBF_thresh = log(3),
                           pass_values = false,
                           BayesianTest = false)
    N1_samp = length(x1)
    N2_samp = length(x2)
    if (N1_subsamp > N1_samp)|(N2_subsamp > N2_samp)
        error("Number of subsamples > number of real samples")
    end
    stat_vec = zeros(N_repeat)
    for i = 1:N_repeat
        inds1 = shuffle(1:N1_samp)
        if same_ind
            if (N1_samp == N2_samp) & (N1_subsamp == N2_subsamp)
                inds2 = inds1
            else
                error("Same index does not work.")
            end
        else
            inds2 = shuffle(1:N2_samp)
        end
        x1_temp = x1[inds1[1:N1_subsamp]]
        x2_temp = x2[inds2[1:N2_subsamp]]
        if BayesianTest
            stat_vec[i] = Test_function(filter(!isnan,x1_temp),
                                        filter(!isnan,x2_temp))
        else
            stat_vec[i] = pvalue(Test_function(
                                filter(!isnan,x1_temp),
                                filter(!isnan,x2_temp)))
        end
    end
    if pass_values
        return stat_vec
    else
        if BayesianTest
            return [1 .- (stat_vec .> logBF_thresh),
                    1 .- (stat_vec .< -logBF_thresh)]
        else
            return 1 .- (stat_vec .< p_thresh)
        end
    end
end
function Func_subsamp_test(x::Array{Float64,2}, N_subsamp::Int64,
                           N_repeat::Int64, Test_function;
                           p_thresh = 0.05, logBF_thresh = log(3),
                           pass_values = false,
                           BayesianTest = false)
    Func_subsamp_test(x[:,1],x[:,2],N_subsamp,N_subsamp,N_repeat,Test_function;
                      p_thresh = p_thresh, logBF_thresh = logBF_thresh,
                      pass_values = pass_values,BayesianTest = BayesianTest)
end
function Func_subsamp_test(x1::Array{Array{Float64,1},1},
                           x2::Array{Array{Float64,1},1},
                           N1_subsamp::Int64, N2_subsamp::Int64,
                           N_repeat::Int64, Test_function;
                           same_ind = false,
                           p_thresh = 0.05, logBF_thresh = log(3),
                           pass_values = false, BayesianTest = false)
    N1_samp = findmin(length.(x1))[1]
    N2_samp = findmin(length.(x2))[1]
    if (N1_subsamp > N1_samp)|(N2_subsamp > N2_samp)
        error("Number of subsamples > number of real samples.")
    end
    if length(x1) != length(x2)
        error("x1 and x2 do not have the same length.")
    end
    stat_vec = zeros(N_repeat)
    for i = 1:N_repeat
        inds1 = shuffle(1:N1_samp)
        if same_ind
            if (N1_samp == N2_samp) & (N1_subsamp == N2_subsamp)
                inds2 = inds1
            else
                error("Same index does not work.")
            end
        else
            inds2 = shuffle(1:N2_samp)
        end
        x1_temp = []
        x2_temp = []
        for i = 1:length(x1)
            push!(x1_temp, x1[i][inds1[1:N1_subsamp]])
            push!(x2_temp, x2[i][inds2[1:N2_subsamp]])
        end
        x1_temp = reduce(vcat,x1_temp)
        x2_temp = reduce(vcat,x2_temp)
        if BayesianTest
            stat_vec[i] = Test_function(filter(!isnan,x1_temp),
                                        filter(!isnan,x2_temp))
        else
            stat_vec[i] = pvalue(Test_function(
                                filter(!isnan,x1_temp),
                                filter(!isnan,x2_temp)))
        end
    end
    if pass_values
        return stat_vec
    else
        if BayesianTest
            return [1 .- (stat_vec .> logBF_thresh),
                    1 .- (stat_vec .< -logBF_thresh)]
        else
            return 1 .- (stat_vec .< p_thresh)
        end
    end
end
export Func_subsamp_test

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_subsamp_Gbased_participants(X,S,N; Groups = 0:2, rng = -1)
    if rng == -1
        rng = Random.GLOBAL_RNG
    end
    if length(X) !== length(S)
        error("X and S must have the same size.")
    end
    inds = [(1:length(X))[S .== g] for g = Groups]
    if N > findmin(length.(inds))[1]
        error("N is too big.")
    end
    X_sampled = [X[shuffle(rng,inds[i])[1:N]] for i = 1:length(Groups)]
    return vcat(X_sampled...)
end
export Func_subsamp_Gbased_participants

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# cross validation sampler
function stratified_cv_indices(df, fields, num_folds, fold_num; idxs = :both)
    if fold_num < 1 || fold_num > num_folds
        error("Fold number must be between 1 and the number of folds.")
    end
    gidxs = groupby(df, fields) |> groupindices
    idxs = [findall(==(i), gidxs) for i in union(gidxs)]
    fold_sizes = [length(idx) ÷ num_folds + 1 for idx in idxs]
    fold_indices = vcat([idx[(fold_num-1)*N+1:min(length(idx), fold_num*N)]
                         for (idx, N) in zip(idxs, fold_sizes)]...)
    rest = setdiff(1:nrow(df), fold_indices)
    if idxs === :train
        rest
    elseif idxs === :test
        fold_indices
    else
        rest, fold_indices
    end
end
