# ------------------------------------------------------------------------------
# parameters
# ------------------------------------------------------------------------------
parameters(A::Str_Agent_Policy) = parameters(A,η0_global)
# parameters(::Str_Agent_Policy,η) = (; 
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
#
parameters(::Str_Agent_Policy,η) = (; 
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
parameters(::Str_Agent_Policy,η::NamedTuple) = (; 
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
function parameters(::FitPopulations.DiagonalNormalPrior, params, ::Str_Agent_Policy)
      (μ = deepcopy(params), σ = FitPopulations._one(params))
end
function parameters_old(A::Str_Agent_Policy,η)
      η_extended = vcat(η[1:13] , η[14] .* [1.,1.,1.], 
                        η[15:19], η[20] .* [1.,1.,1.], 
                        η[21:24], [0.,0.],
                        η[25:29])
      parameters(A,η_extended)
end
export parameters_old

# ------------------------------------------------------------------------------
# Agent initialization
# ------------------------------------------------------------------------------                 
function initialize!(A::Str_Agent_Policy, p;
                  Goal_type = 0, eR_max_t = 0., T_PS = 100, β_MBi_1 = 0.1)
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

      A.C_s .= 0; A.C_sa .= 0; A.C_sas .= 0; A.θ_sas .= 0
      A.eR_sas .= 0; A.iR_sas .= 0
      
      A.Q_MBe .= 0; A.Q_MBi .= 0; A.U_e .= 0; A.U_i .= 0
      A.V_dummy .= 0; A.P_dummy .= 0

      A.Q_MFe .= A.Param.Q_e0; A.Q_MFi .= A.Param.Q_i0; A.E_e .= 0; A.E_i .= 0

      A.Q_MB_t .= -1; A.Q_MF_t .= -1; A.Q_t .= -1; A.π_A_t .= -1;

      A.eR_t = -1; A.eRPE_t = -1; A.iR_t = -1; A.iRPE_t = -1
      A.eR_max_t = eR_max_t
      
      nothing 
end
# ------------------------------------------------------------------------------
# Agent logp
# ------------------------------------------------------------------------------
function logp(Data, A::Str_Agent_Policy, p)
      initialize!(A, p; Goal_type = Data[1].G_type)
      Epi = 1; lp = 0.
      for (AStates, G_type) in Data
            if Epi == 1
                  Func_initialize_agent!(A, AStates[1])
            else
                  Func_reset_agent_EpiEnd!(A)
            end
            Epi += 1
            lp += Func_train_to_SandA!(A, AStates; onlyLogl = true)
      end
      lp
end
export logp
function logp_pass_agent(Data, A::Str_Agent_Policy, p)
      initialize!(A, p; Goal_type = Data[1].G_type)
      lps = zeros(length(Data)); Agents = Array{Array{Str_Agent_Policy,1},1}([])
      Epi = 1
      for (AStates, G_type) in Data
            if Epi == 1
                  Func_initialize_agent!(A, AStates[1])
            else
                  Func_reset_agent_EpiEnd!(A)
            end
            lp, As = Func_train_to_SandA!(A, AStates; onlyLogl = false)
            lps[Epi] = lp; push!(Agents,As)
            Epi += 1
      end
      lps, Agents
end
export logp_pass_agent

# ------------------------------------------------------------------------------
# simulate Agent
# ------------------------------------------------------------------------------
function simulate(A::Str_Agent_Policy, p;
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
      sdata = Vector{@NamedTuple{AStates::Vector{Str_Agent_State}, G_type::Int64}}([])
      for Epi = eachindex(s0)
            if Epi == 1
                  AState0 = Str_Agent_State()
                  AState0 = Str_Agent_State(AState0, s0[Epi], if_init = true)
                  Func_initialize_agent!(A, AState0)
            else
                  AState0 = deepcopy(sdata[end].AStates[end])
                  AState0 = Str_Agent_State(AState0, s0[Epi])
                  Func_reset_agent_EpiEnd!(A)
            end
            push!(sdata, 
                  Func_simulate_SandA!(A, AState0, Environment; rng = rng,
                                          G_type_default = G_type))
      end
      if tracked
            lps, Agents = logp_pass_agent(sdata, A::Str_Agent_Policy, p)
            lp = sum(lps)
            sdata = (; data = sdata, history = Agents)
      else
            lp = logp(sdata, A::Str_Agent_Policy, p)
      end
      if !ifpass_env
            (; data = sdata, logp = lp)
      else
            (; data = sdata, logp = lp, TM = Environment.TransMatrix)
      end
end
export simulate
