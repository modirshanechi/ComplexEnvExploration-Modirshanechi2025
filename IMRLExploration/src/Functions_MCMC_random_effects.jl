# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# functions for MCMC based on Klass Stephan paper
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Prior_MCMC_sampler(ϵ::Float64;n_r=3,n_M=57,H0=false,r=-1)
        if H0
                r = ones(n_r) ./ n_r
        elseif r == -1
                P_r = Dirichlet(n_r, ϵ)
                r = rand(P_r)
        end
        M = rand(Categorical(r),n_M)
        return r,M
end
export Prior_MCMC_sampler

function Base_Chain_MCMC_sampler(r,M;N_scale=10,ϵ=1.,N_change = 1)
        M_new = deepcopy(M)
        P_unif = DiscreteUniform(1, length(M))
        for i = shuffle(1:length(M))[1:N_change]
                j = rand(P_unif)
                M_new[i] = M[j]
        end
        r_new = rand(Dirichlet(ϵ .+
                     [count(==(i),M) for i = 1:length(r)] ./ N_scale))
        return r_new, M_new
end
export Base_Chain_MCMC_sampler

function Base_Chain_MCMC_logpdf(r1,M1,r2,M2;N_scale=10,ϵ=1.)
        if sum(abs.(M2 .- M1) .> 0) == 0
                log_Prob_m = 0 # This probability doesn't matter if M2 and M1 are equal
        else
                log_Prob_m = [0.]
                for i = (1:length(M1))[abs.(M2 .- M1) .> 0]
                        log_Prob_m[1] = log_Prob_m[1] +
                                        log(count(==(M2[i]),M1)) -
                                        log(length(M1))
                end
                log_Prob_m = log_Prob_m[1]
        end
        Prob_r = pdf(Dirichlet(ϵ .+
                     [count(==(i),M1) for i = 1:length(r2)] ./ N_scale),
                     r2)
        return log_Prob_m + log(Prob_r)
end
export Base_Chain_MCMC_logpdf

function logL_MCMC(r,M,L_matrix;P_r=Dirichlet(3, 1.))
        Sub_Num = length(M)
        log_prior = log(pdf(P_r,r))
        log_M_r = sum([count(==(i),M) for i = 1:length(r)] .* log.(r))
        log_Y_M = sum([L_matrix[i,Int64(M[i])] for i=1:Sub_Num])
        return log_prior + log_M_r + log_Y_M
end
export logL_MCMC

function MCMC_accept_prob(r1,M1,r2,M2,L_matrix;N_scale=10,ϵ=1.,
                                               P_r=Dirichlet(3, 1))
        log_π1 = logL_MCMC(r1,M1,L_matrix;P_r=P_r)
        log_π2 = logL_MCMC(r2,M2,L_matrix;P_r=P_r)
        log_ϕ12 = Base_Chain_MCMC_logpdf(r1,M1,r2,M2;N_scale=N_scale,ϵ=ϵ)
        log_ϕ21 = Base_Chain_MCMC_logpdf(r2,M2,r1,M1;N_scale=N_scale,ϵ=ϵ)
        log_a12 = log_π2 + log_ϕ21 - log_π1 - log_ϕ12
        return min(1,exp(log_a12))
end
export MCMC_accept_prob

function MCMC_BMS(L_matrix; N_Sampling = Int(1e5), N_Chains = 5, α = 1.,
                            N_scale=1., ϵ=1., N_change=1, uniform_initial=false)
        N_model = size(L_matrix)[2]
        Sub_Num = size(L_matrix)[1]

        R_matrix = zeros(N_Sampling, N_model, N_Chains);
        M_matrix = zeros(N_Sampling, Sub_Num, N_Chains);
        π_matrix = zeros(N_Sampling, N_Chains);

        P_r = Dirichlet(N_model, α)

        M0_star = zeros(size(L_matrix)[1])
        for i = 1:Sub_Num
                M0_star[i] = findmax(L_matrix[i,:])[2]
        end
        r0_star = [count(==(i),M0_star) for i = 1:N_model] ./ Sub_Num

        for i_chain = 1:N_Chains
                if uniform_initial
                        r0,M0 = Prior_MCMC_sampler(1.;n_r=N_model,n_M=Sub_Num)
                else
                        r0,M0 = Base_Chain_MCMC_sampler(r0_star,M0_star;
                                        N_scale=N_scale,ϵ=ϵ,N_change=N_change)
                end
                R_matrix[1,:,i_chain] = r0
                M_matrix[1,:,i_chain] = M0
                π_matrix[1,i_chain] = logL_MCMC(r0,M0,L_matrix;P_r=P_r)
                for i_sample = 2:N_Sampling
                        r1 = R_matrix[i_sample-1,:,i_chain]
                        M1 = M_matrix[i_sample-1,:,i_chain]
                        r2,M2 = Base_Chain_MCMC_sampler(r1,M1;N_scale=N_scale,ϵ=ϵ,
                                                        N_change=N_change)
                        a12 = MCMC_accept_prob(r1,M1,r2,M2,L_matrix;
                                               N_scale=N_scale,ϵ=ϵ,P_r=P_r)
                        if rand() < a12
                                R_matrix[i_sample,:,i_chain] = r2
                                M_matrix[i_sample,:,i_chain] = M2
                                π_matrix[i_sample,i_chain] = logL_MCMC(r2,M2,L_matrix;P_r=P_r)
                        else
                                R_matrix[i_sample,:,i_chain] = r1
                                M_matrix[i_sample,:,i_chain] = M1
                                π_matrix[i_sample,i_chain] = π_matrix[i_sample-1,i_chain]
                        end
                end
        end
        return R_matrix, M_matrix, π_matrix
end
export MCMC_BMS


function Sampling_H0_H1_rOnly(L_matrix,α; N_Sampling = Int(1e5))
        N_model = size(L_matrix)[2]
        Sub_Num = size(L_matrix)[1]

        logP0 = MarginalProb_XgivenR(L_matrix, ones(N_model) ./ N_model)[1]

        R1_matrix = zeros(N_Sampling, N_model);
        π1_matrix = zeros(N_Sampling);
        for i_sample = 1:N_Sampling
                r1,M1 = Prior_MCMC_sampler(α;n_r=N_model,n_M=Sub_Num)
                R1_matrix[i_sample,:] = r1
                π1_matrix[i_sample] = MarginalProb_XgivenR(L_matrix,r1)[1]
        end
        π1_max = findmax(π1_matrix)[1]
        logP1 = π1_max + log(sum(exp.(π1_matrix .- π1_max))) - log(N_Sampling)
        
        BOR = 1 / (1 + exp(logP1 - logP0))

        return BOR, logP0, logP1, R1_matrix, π1_matrix
end
export Sampling_H0_H1_rOnly

function MarginalProb_XgivenR(L_matrix,r)
        Sub_Num = size(L_matrix)[1]
        L_max_vec = [findmax(L_matrix[n,:])[1] for n = 1:Sub_Num]
        L_xr_vec = [(L_max_vec[n] +
                    log(sum(r .* exp.(L_matrix[n,:] .- L_max_vec[n]))))
                    for n = 1:Sub_Num]
        return sum(L_xr_vec), L_xr_vec
end
export MarginalProb_XgivenR


function Sampling_H0_H1(L_matrix,α; N_Sampling = Int(1e5))
        N_model = size(L_matrix)[2]
        Sub_Num = size(L_matrix)[1]

        R0_matrix = zeros(N_Sampling, N_model);
        M0_matrix = zeros(N_Sampling, Sub_Num);
        π0_matrix = zeros(N_Sampling);

        R1_matrix = zeros(N_Sampling, N_model);
        M1_matrix = zeros(N_Sampling, Sub_Num);
        π1_matrix = zeros(N_Sampling);

        for i_sample = 1:N_Sampling
                r0,M0 = Prior_MCMC_sampler(α;n_r=N_model,n_M=Sub_Num,H0=true)
                R0_matrix[i_sample,:] = r0
                M0_matrix[i_sample,:] = M0
                π0_matrix[i_sample] = sum([L_matrix[i,Int64(M0[i])] for i=1:Sub_Num])

                r1,M1 = Prior_MCMC_sampler(α;n_r=N_model,n_M=Sub_Num,H0=false)
                R1_matrix[i_sample,:] = r1
                M1_matrix[i_sample,:] = M1
                π1_matrix[i_sample] = sum([L_matrix[i,Int64(M1[i])] for i=1:Sub_Num])
        end
        π0_max = findmax(π0_matrix)[1]
        logP0 = π0_max + log(sum(exp.(π0_matrix .- π0_max))) - log(N_Sampling)
        π1_max = findmax(π1_matrix)[1]
        logP1 = π1_max + log(sum(exp.(π1_matrix .- π1_max))) - log(N_Sampling)

        BOR = 1 / (1 + exp(logP1 - logP0))

        return BOR, logP0, logP1, R0_matrix, M0_matrix, π0_matrix,
               R1_matrix, M1_matrix, π1_matrix
end
export Sampling_H0_H1


function MCMC_BMS_Statistics(R_matrix_samples,M_matrix_samples,BOR)
        R_samples_all = zeros(size(R_matrix_samples)[1]*size(R_matrix_samples)[3],
                              size(R_matrix_samples)[2]);
        for i = 1:size(R_matrix_samples)[2]
                R_samples_all[:,i] = R_matrix_samples[:,i,:][:]
        end
        exp_r = mean(R_samples_all, dims = 1)[:]
        d_exp_r = std(R_samples_all, dims = 1)[:]

        Best_model_samples = zeros(size(R_samples_all));
        for i = 1:size(R_samples_all)[1]
                Best_model_samples[i,findmax(R_samples_all[i,:])[2]] = 1
        end
        xp = mean(Best_model_samples, dims = 1)[:]

        pxp = xp .* (1 - BOR) .+ ones(length(xp)) ./ length(xp) .* BOR

        M_samples_all = zeros(size(M_matrix_samples)[1]*size(M_matrix_samples)[3],
                              size(M_matrix_samples)[2]);
        for i = 1:size(M_matrix_samples)[2]
                M_samples_all[:,i] = M_matrix_samples[:,i,:][:]
        end
        exp_M = zeros(size(M_samples_all)[2],size(R_matrix_samples)[2])
        for i = 1:size(R_matrix_samples)[2]
                exp_M[:,i] = mean(M_samples_all .== i,dims=1)[:]
        end
        return R_samples_all, M_samples_all, exp_r, d_exp_r, xp, pxp, exp_M
end
function MCMC_BMS_Statistics(L_matrix; N_Sampling = Int(1e5), 
                N_Sampling_BOR = Int(1e5),
                N_Chains = 40, α = 1., N_scale=1., ϵ=1., N_jump = 50,
                uniform_initial=false, test_plotting = true,  N_change = -1)
        if N_change == -1
                Sub_Num = size(L_matrix)[1]
                N_change=max(1,Int64(floor(Sub_Num / 20)))
        end
        R_matrix, M_matrix, π_matrix = MCMC_BMS(L_matrix;
                        N_Sampling=N_Sampling,N_Chains=N_Chains,α=α,
                        N_scale=N_scale,ϵ=ϵ,N_change=N_change,
                        uniform_initial=uniform_initial);

        BOR = Sampling_H0_H1_rOnly(L_matrix,α; N_Sampling = N_Sampling_BOR)[1];

        R_matrix_samples = R_matrix[N_jump:N_jump:N_Sampling,:,:];
        M_matrix_samples = M_matrix[N_jump:N_jump:N_Sampling,:,:];

        # Testing
        if test_plotting
                N_model = size(L_matrix)[2]
                figure()
                ax = subplot(2,2,1)
                ax.plot(π_matrix)
                ax = subplot(2,2,2)
                ax.plot(M_matrix[1:1000,1:10,1])
                for i = 1:N_model
                        ax = subplot(2,2,3)
                        ax.hist(R_matrix_samples[:,i,:][:])
                        ax = subplot(2,2,4)
                        ax.plot(R_matrix_samples[:,i,1])
                end
        end
        R_samples_all, M_samples_all, exp_r, d_exp_r, xp, pxp, exp_M =
                MCMC_BMS_Statistics(R_matrix_samples,M_matrix_samples,BOR)
        return R_matrix_samples, M_matrix_samples, R_samples_all, M_samples_all,
               exp_r, d_exp_r, xp, pxp, exp_M, BOR
                
end
export MCMC_BMS_Statistics


