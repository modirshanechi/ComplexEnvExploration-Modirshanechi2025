# ------------------------------------------------------------------------------
# parameter convertor
# ------------------------------------------------------------------------------
function η2x(η::Number,i)
      if i ∈ η_01index_global
            return logit(η)      # logit = inverse softmax: [0,1] → R
      elseif i ∈ η_Rpindex_global
            return logexpm1(η)   # inv-softplus: R+ → R
      else
            return η
      end
end
function η2x(η::Vector,i)
      η2x(η[i],i)
end
function η2x(η::Vector)
      [η2x(η,i) for i = eachindex(η)]
end
export η2x
function x2η(x::Number,i)
      if i ∈ η_01index_global
            return logistic(x)   # softmax: R → [0,1]
      elseif i ∈ η_Rpindex_global
            return log1pexp(x)   # softplus: R → R+
      else
            return x
      end
end
function x2η(x::Vector,i)
      x2η(x[i],i)
end
function x2η(x::Vector)
      [x2η(x,i) for i = eachindex(x)]
end
export x2η
function param2η(p::ComponentVector)
      x = vcat(
            [p["κ"],p["ϵ_new"], p["ϵ_obs"], p["λ_e"], p["λ_i"]], 
            η2x(η0_global,6),η2x(η0_global,7),
            [p["ρ"], p["μ_e"], p["μ_i"], p["Q_e0"], p["Q_i0"],p["β_MBe_1"], 
             p["β_MBe_2_G0"], p["β_MBe_2_G1"], p["β_MBe_2_G2"]],
            η2x(η0_global,17),
            [p["β_MBi_2_G0"], p["β_MBi_2_G1"], p["β_MBi_2_G2"], p["β_MFe_1"], 
             p["β_MFe_2_G0"], p["β_MFe_2_G1"], p["β_MFe_2_G2"], p["β_MFi_1"], 
             p["β_MFi_2_G0"], p["β_MFi_2_G1"], p["β_MFi_2_G2"],
             p["Q_bias_1"],   p["Q_bias_2"],
             p["r1"], p["r2"], p["wN"], p["wI"], p["wS"]]);
      x = x2η(x)
      return (;κ=x[1], ϵ_new=x[2], ϵ_obs=x[3], λ_e=x[4], λ_i=x[5], T_PS_e=x[6],
      T_PS_i=x[7], ρ=x[8], μ_e=x[9], μ_i=x[10], Q_e0=x[11], Q_i0=x[12],
      β_MBe_1=x[13], β_MBe_2_G0=x[14], β_MBe_2_G1=x[15], β_MBe_2_G2=x[16], 
      β_MBi_1=x[17], β_MBi_2_G0=x[18], β_MBi_2_G1=x[19], β_MBi_2_G2=x[20], 
      β_MFe_1=x[21], β_MFe_2_G0=x[22], β_MFe_2_G1=x[23], β_MFe_2_G2=x[24],
      β_MFi_1=x[25], β_MFi_2_G0=x[26], β_MFi_2_G1=x[27], β_MFi_2_G2=x[28],
      Q_bias_1=x[29],   Q_bias_2=x[30],
      r1=x[31], r2=x[32], wN=x[33], wI=x[34], wS=x[35])
end
function param2η(p::NamedTuple) 
      param2η(ComponentArray(p))
end
export param2η
function paramold2η(p::ComponentVector)
      x = vcat(
            [p["κ"],p["ϵ_new"], p["ϵ_obs"], p["λ_e"], p["λ_i"]], 
            η2x(η0_global,6),η2x(η0_global,7),
            [p["ρ"], p["μ_e"], p["μ_i"], p["Q_e0"], p["Q_i0"],p["β_MBe_1"], 
             p["β_MBe_2"],    p["β_MBe_2"],    p["β_MBe_2"]],
            η2x(η0_global,17),
            [p["β_MBi_2_G0"], p["β_MBi_2_G1"], p["β_MBi_2_G2"], p["β_MFe_1"], 
             p["β_MFe_2"],    p["β_MFe_2"],    p["β_MFe_2"],    p["β_MFi_1"], 
             p["β_MFi_2_G0"], p["β_MFi_2_G1"], p["β_MFi_2_G2"],
             0.0,   0.0,
             p["r1"], p["r2"], p["wN"], p["wI"], p["wS"]]);
      x = x2η(x)
      return (;κ=x[1], ϵ_new=x[2], ϵ_obs=x[3], λ_e=x[4], λ_i=x[5], T_PS_e=x[6],
      T_PS_i=x[7], ρ=x[8], μ_e=x[9], μ_i=x[10], Q_e0=x[11], Q_i0=x[12],
      β_MBe_1=x[13], β_MBe_2_G0=x[14], β_MBe_2_G1=x[15], β_MBe_2_G2=x[16], 
      β_MBi_1=x[17], β_MBi_2_G0=x[18], β_MBi_2_G1=x[19], β_MBi_2_G2=x[20], 
      β_MFe_1=x[21], β_MFe_2_G0=x[22], β_MFe_2_G1=x[23], β_MFe_2_G2=x[24],
      β_MFi_1=x[25], β_MFi_2_G0=x[26], β_MFi_2_G1=x[27], β_MFi_2_G2=x[28],
      Q_bias_1=x[29],   Q_bias_2=x[30],
      r1=x[31], r2=x[32], wN=x[33], wI=x[34], wS=x[35])
end
function paramold2η(p::NamedTuple) 
      paramold2η(ComponentArray(p))
end
export paramold2η
function zero_padd_param(p::ComponentVector)
      x = vcat(
            [p["κ"],p["ϵ_new"], p["ϵ_obs"], p["λ_e"], p["λ_i"]], 
            0.,0.,
            [p["ρ"], p["μ_e"], p["μ_i"], p["Q_e0"], p["Q_i0"],p["β_MBe_1"], 
             p["β_MBe_2_G0"], p["β_MBe_2_G1"], p["β_MBe_2_G2"]],
            0.,
            [p["β_MBi_2_G0"], p["β_MBi_2_G1"], p["β_MBi_2_G2"], p["β_MFe_1"], 
             p["β_MFe_2_G0"], p["β_MFe_2_G1"], p["β_MFe_2_G2"], p["β_MFi_1"], 
             p["β_MFi_2_G0"], p["β_MFi_2_G1"], p["β_MFi_2_G2"],
             p["Q_bias_1"],   p["Q_bias_2"],
             p["r1"], p["r2"], p["wN"], p["wI"], p["wS"]]);
      return (;κ=x[1], ϵ_new=x[2], ϵ_obs=x[3], λ_e=x[4], λ_i=x[5], T_PS_e=x[6],
      T_PS_i=x[7], ρ=x[8], μ_e=x[9], μ_i=x[10], Q_e0=x[11], Q_i0=x[12],
      β_MBe_1=x[13], β_MBe_2_G0=x[14], β_MBe_2_G1=x[15], β_MBe_2_G2=x[16], 
      β_MBi_1=x[17], β_MBi_2_G0=x[18], β_MBi_2_G1=x[19], β_MBi_2_G2=x[20], 
      β_MFe_1=x[21], β_MFe_2_G0=x[22], β_MFe_2_G1=x[23], β_MFe_2_G2=x[24],
      β_MFi_1=x[25], β_MFi_2_G0=x[26], β_MFi_2_G1=x[27], β_MFi_2_G2=x[28],
      Q_bias_1=x[29],   Q_bias_2=x[30],
      r1=x[31], r2=x[32], wN=x[33], wI=x[34], wS=x[35])
end
function zero_padd_param(p::NamedTuple) 
      zero_padd_param(ComponentArray(p))
end
export zero_padd_param
# ------------------------------------------------------------------------------
# parameters
# ------------------------------------------------------------------------------
parameters(A::Str_Agent) = parameters(A,η0_global)
# parameters(::Str_Agent,η) = (; 
#                   κ           = η2x(η,1),
#                   ϵ_new       = η2x(η,2),
#                   ϵ_obs       = η2x(η,3),
#                   λ_e         = η2x(η,4),
#                   λ_i         = η2x(η,5),
#                   # T_PS_e    = η2x(η,6),    # fixed
#                   # T_PS_i    = η2x(η,7),    # fixed
#                   ρ           = η2x(η,8),
#                   μ_e         = η2x(η,9),
#                   μ_i         = η2x(η,10),
#                   Q_e0        = η2x(η,11),
#                   Q_i0        = η2x(η,12),
#                   β_MBe_1     = η2x(η,13),
#                   β_MBe_2     = η2x(η,14),
#                   # β_MBi_1   = η2x(η,15),   # fixed
#                   β_MBi_2_G0  = η2x(η,16),
#                   β_MBi_2_G1  = η2x(η,17),
#                   β_MBi_2_G2  = η2x(η,18),
#                   β_MFe_1     = η2x(η,19),
#                   β_MFe_2     = η2x(η,20),
#                   β_MFi_1     = η2x(η,21),
#                   β_MFi_2_G0  = η2x(η,22),
#                   β_MFi_2_G1  = η2x(η,23),
#                   β_MFi_2_G2  = η2x(η,24),
#                   r1          = η2x(η,25),
#                   r2          = η2x(η,26),
#                   wN          = η2x(η,27),
#                   wI          = η2x(η,28),
#                   wS          = η2x(η,29))
# #
parameters(::Str_Agent,η) = (; 
                  κ           = η2x(η,1),
                  ϵ_new       = η2x(η,2),
                  ϵ_obs       = η2x(η,3),
                  λ_e         = η2x(η,4),
                  λ_i         = η2x(η,5),
                  # T_PS_e    = η2x(η,6),    # fixed
                  # T_PS_i    = η2x(η,7),    # fixed
                  ρ           = η2x(η,8),
                  μ_e         = η2x(η,9),
                  μ_i         = η2x(η,10),
                  Q_e0        = η2x(η,11),
                  Q_i0        = η2x(η,12),
                  β_MBe_1     = η2x(η,13),
                  β_MBe_2_G0  = η2x(η,14),
                  β_MBe_2_G1  = η2x(η,15),
                  β_MBe_2_G2  = η2x(η,16),
                  # β_MBi_1   = η2x(η,17),   # fixed
                  β_MBi_2_G0  = η2x(η,18),
                  β_MBi_2_G1  = η2x(η,19),
                  β_MBi_2_G2  = η2x(η,20),
                  β_MFe_1     = η2x(η,21),
                  β_MFe_2_G0  = η2x(η,22),
                  β_MFe_2_G1  = η2x(η,23),
                  β_MFe_2_G2  = η2x(η,24),
                  β_MFi_1     = η2x(η,25),
                  β_MFi_2_G0  = η2x(η,26),
                  β_MFi_2_G1  = η2x(η,27),
                  β_MFi_2_G2  = η2x(η,28),
                  Q_bias_1    = η2x(η,29),
                  Q_bias_2    = η2x(η,30),
                  r1          = η2x(η,31),
                  r2          = η2x(η,32),
                  wN          = η2x(η,33),
                  wI          = η2x(η,34),
                  wS          = η2x(η,35))
#
parameters(::Str_Agent,η::NamedTuple) = (; 
                  κ           = η2x(η.κ,1),
                  ϵ_new       = η2x(η.ϵ_new,2),
                  ϵ_obs       = η2x(η.ϵ_obs,3),
                  λ_e         = η2x(η.λ_e,4),
                  λ_i         = η2x(η.λ_i,5),
                  # T_PS_e    = η2x(η.T_PS_e,6),    # fixed
                  # T_PS_i    = η2x(η.T_PS_i,7),    # fixed
                  ρ           = η2x(η.ρ,8),
                  μ_e         = η2x(η.μ_e,9),
                  μ_i         = η2x(η.μ_i,10),
                  Q_e0        = η2x(η.Q_e0,11),
                  Q_i0        = η2x(η.Q_i0,12),
                  β_MBe_1     = η2x(η.β_MBe_1,13),
                  β_MBe_2_G0  = η2x(η.β_MBe_2_G0,14),
                  β_MBe_2_G1  = η2x(η.β_MBe_2_G1,15),
                  β_MBe_2_G2  = η2x(η.β_MBe_2_G2,16),
                  # β_MBi_1   = η2x(η.β_MBi_1,17),   # fixed
                  β_MBi_2_G0  = η2x(η.β_MBi_2_G0,18),
                  β_MBi_2_G1  = η2x(η.β_MBi_2_G1,19),
                  β_MBi_2_G2  = η2x(η.β_MBi_2_G2,20),
                  β_MFe_1     = η2x(η.β_MFe_1,21),
                  β_MFe_2_G0  = η2x(η.β_MFe_2_G0,22),
                  β_MFe_2_G1  = η2x(η.β_MFe_2_G1,23),
                  β_MFe_2_G2  = η2x(η.β_MFe_2_G2,24),
                  β_MFi_1     = η2x(η.β_MFi_1,25),
                  β_MFi_2_G0  = η2x(η.β_MFi_2_G0,26),
                  β_MFi_2_G1  = η2x(η.β_MFi_2_G1,27),
                  β_MFi_2_G2  = η2x(η.β_MFi_2_G2,28),
                  Q_bias_1    = η2x(η.Q_bias_1,29),
                  Q_bias_2    = η2x(η.Q_bias_2,30),
                  r1          = η2x(η.r1,31),
                  r2          = η2x(η.r2,32),
                  wN          = η2x(η.wN,33),
                  wI          = η2x(η.wI,34),
                  wS          = η2x(η.wS,35))
function parameters(m::PopulationModel, η)
      params = parameters(m.model, η)
      params_nonshared = FitPopulations.drop(params, m.shared)
      population_parameters = parameters(m.prior, params_nonshared, m.model)
      merge(params_nonshared, (; population_parameters), params[m.shared]) # order matter! see above
end
function parameters(::FitPopulations.DiagonalNormalPrior, params, ::Str_Agent)
      (μ = deepcopy(params), σ = FitPopulations._one(params))
end

# ------------------------------------------------------------------------------
# Agent initialization
# ------------------------------------------------------------------------------                 
function initialize!(A::Str_Agent, p;
                  Goal_type = 0, eR_max_t = 0.,
                  T_PS = 100, β_MBi_1 = 0.1,
                  Goal_states = Goal_states_inf_env)
      A.Param.κ       = x2η(p.κ,        1)
      A.Param.ϵ_new   = x2η(p.ϵ_new,    2)
      A.Param.ϵ_obs   = x2η(p.ϵ_obs,    3)
      A.Param.λ_e     = x2η(p.λ_e,      4)
      A.Param.λ_i     = x2η(p.λ_i,      5)
      A.Param.T_PS_e  = T_PS  # A.Param.T_PS_e  = η0_global[6]
      A.Param.T_PS_i  = T_PS  # A.Param.T_PS_i  = η0_global[7]
      A.Param.ρ       = x2η(p.ρ,        8)
      A.Param.μ_e     = x2η(p.μ_e,      9)
      A.Param.μ_i     = x2η(p.μ_i,      10)
      A.Param.Q_e0    = x2η(p.Q_e0,     11)
      A.Param.Q_i0    = x2η(p.Q_i0,     12)
      A.Param.β_MBe_1 = x2η(p.β_MBe_1,  13)
      if Goal_type == 0
            A.Param.β_MBe_2 = x2η(p.β_MBe_2_G0,  14)
      elseif Goal_type == 1
            A.Param.β_MBe_2 = x2η(p.β_MBe_2_G1,  15)
      else
            A.Param.β_MBe_2 = x2η(p.β_MBe_2_G2,  16)
      end
      A.Param.β_MBi_1 = β_MBi_1     # A.Param.β_MBi_1 = η0_global[15]
      if Goal_type == 0
            A.Param.β_MBi_2 = x2η(p.β_MBi_2_G0,     18)
      elseif Goal_type == 1
            A.Param.β_MBi_2 = x2η(p.β_MBi_2_G1,     19)
      else
            A.Param.β_MBi_2 = x2η(p.β_MBi_2_G2,     20)
      end
      A.Param.β_MFe_1 = x2η(p.β_MFe_1,  21)
      if Goal_type == 0
            A.Param.β_MFe_2 = x2η(p.β_MFe_2_G0,  22)
      elseif Goal_type == 1
            A.Param.β_MFe_2 = x2η(p.β_MFe_2_G1,  23)
      else
            A.Param.β_MFe_2 = x2η(p.β_MFe_2_G2,  24)
      end
      A.Param.β_MFi_1 = x2η(p.β_MFi_1,  25)
      if Goal_type == 0
            A.Param.β_MFi_2 = x2η(p.β_MFi_2_G0, 26)
      elseif Goal_type == 1
            A.Param.β_MFi_2 = x2η(p.β_MFi_2_G1, 27)
      else
            A.Param.β_MFi_2 = x2η(p.β_MFi_2_G2, 28)
      end
      A.Param.Q_bias[1] = x2η(p.Q_bias_1, 29)
      A.Param.Q_bias[2] = x2η(p.Q_bias_2, 30)

      A.Func_eR_sas.Rs[2] = x2η(p.r1,     31)
      A.Func_eR_sas.Rs[3] = x2η(p.r2,     32)

      A.Func_iR_sas.ws[1] = x2η(p.wN,     33)
      A.Func_iR_sas.ws[2] = x2η(p.wI,     34)
      A.Func_iR_sas.ws[3] = x2η(p.wS,     35)

      A.Func_iR_sas.N_sas .= 0.
      A.Func_iR_sas.I_sas .= 0.
      A.Func_iR_sas.S_sas .= 0.
      
      for i = (length(Goal_states)+1):length(A.State_Set)
            Func_set_unknown_value!(A.State_Set[i])
      end
      A.state_num = length(Goal_states)

      A.C_s .= 0; A.C_sa .= 0; A.C_sas .= 0; A.θ_sas .= 0
      A.eR_sas .= 0; A.iR_sas .= 0
      
      A.Q_MBe .= 0; A.Q_MBi .= 0; A.U_e .= 0; A.U_i .= 0
      A.V_dummy .= 0; A.P_dummy .= 0

      A.Q_MFe .= A.Param.Q_e0; A.Q_MFi .= A.Param.Q_i0; A.E_e .= 0; A.E_i .= 0

      Func_set_unknown_value!(A.S_t)
      A.A_t     = Func_unknown_values(A.Act_Set)                              # WARNING: There will be allocation if actions are arrays
      Func_set_unknown_value!(A.S_t_old)
      A.A_t_old = Func_unknown_values(A.Act_Set)                              # WARNING: There will be allocation if actions are arrays

      A.S_ind_t = -1; A.A_ind_t = -1; A.S_ind_t_old = -1; A.A_ind_t_old = -1

      A.Q_MB_t .= -1; A.Q_MF_t .= -1; A.Q_t .= -1; A.π_A_t .= -1;

      A.eR_t = -1; A.eRPE_t = -1; A.iR_t = -1; A.iRPE_t = -1
      A.eR_max_t = eR_max_t
      
      A.Epi = 1; A.t_total = 0; A.t_epi = 0; A.new_state = true
      nothing 
end
# ------------------------------------------------------------------------------
# Agent logp
# ------------------------------------------------------------------------------
function logp(Data, A::Str_Agent, p)
      initialize!(A, p; Goal_type = Data[1].G_type)
      Epi = 1; lp = 0.
      for (S_Seq, A_Seq, Sind_Seq, Aind_Seq, Snum_Seq, Snew_Seq, SSet_Seq) in Data
            if Epi == 1
                  Func_initialize_agent!(A)
            else
                  Func_reset_agent_EpiEnd!(A)
            end
            Epi += 1
            lp += Func_train_to_SandA!(A, S_Seq, A_Seq, Sind_Seq, Aind_Seq, 
                                          Snum_Seq, Snew_Seq, SSet_Seq; 
                                          onlyLogl = true)
      end
      lp
end
function logp_pass_agent(Data, A::Str_Agent, p)
      initialize!(A, p; Goal_type = Data[1].G_type)
      lps = zeros(length(Data)); Agents = Array{Array{Str_Agent,1},1}([])
      Epi = 1
      for (S_Seq, A_Seq, Sind_Seq, Aind_Seq, Snum_Seq, Snew_Seq, SSet_Seq) in Data
            if Epi == 1
                  Func_initialize_agent!(A)
            else
                  Func_reset_agent_EpiEnd!(A)
            end
            lp, As = Func_train_to_SandA!(A, S_Seq, A_Seq, Sind_Seq, Aind_Seq, 
                                          Snum_Seq, Snew_Seq, SSet_Seq; 
                                          onlyLogl = false)
            lps[Epi] = lp; push!(Agents,As)
            Epi += 1
      end
      lps, Agents
end
export logp_pass_agent

# ------------------------------------------------------------------------------
# simulate Agent
# ------------------------------------------------------------------------------
function simulate(A::Str_Agent, p;
                        s0 = [[0,0],[0,0],[0,0],[0,0],[0,0]],
                        G_type_prob = [1.,1.,1.] ./ 3, 
                        TransMatrix = Read_transM(), 
                        S4_Manipulator = true, Stochastic = true,
                        NotRepeatedState = true, N_stoch = 50,
                        tracked = false, rng=Random.GLOBAL_RNG,
                        ifpass_env = false)
      G_type = rand(rng, Categorical(G_type_prob)) - 1
      @show G_type
      Environment = Str_InfEnv(; G_type=G_type, TransMatrix=TransMatrix, 
                        S4_Manipulator=S4_Manipulator, Stochastic=Stochastic,
                        NotRepeatedState=NotRepeatedState, N_stoch=N_stoch)
      initialize!(A, p; Goal_type = G_type)
      sdata = Vector{@NamedTuple{S_Seq::Vector{Vector{Int64}}, A_Seq::Vector{Int64}, G_type::Int64}}([])
      for Epi = eachindex(s0)
            if Epi == 1
                  Func_initialize_agent!(A)
            else
                  Func_reset_agent_EpiEnd!(A)
            end
            push!(sdata, Func_simulate_SandA!(A, Environment; s0 = s0[Epi], 
                              rng = rng,G_type_default = G_type))
      end
      if tracked
            lps, Agents = logp_pass_agent(sdata, A::Str_Agent, p)
            lp = sum(lps)
            sdata = (; data = sdata, history = Agents)
      else
            lp = logp(sdata, A::Str_Agent, p)
      end
      if !ifpass_env
            (; data = sdata, logp = lp)
      else
            (; data = sdata, logp = lp, TM = Environment.TransMatrix)
      end
end
