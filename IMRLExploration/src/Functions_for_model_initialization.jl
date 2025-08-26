# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Initialization of an agent using initial values
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_initialize_agent!(Agent::Str_Agent)
    Agent.new_state = true
    Func_update_world_model!(Agent)
    Func_update_reward_matrices!(Agent, Agent.Func_eR_sas, extrinsic=true)
    Func_update_reward_matrices!(Agent, Agent.Func_iR_sas, extrinsic=false)
    Func_initialize_MB_eQ!(Agent)
    Func_initialize_MB_iQ!(Agent)
    Agent.new_state = false
end
function Func_initialize_agent!(Agent::Str_Agent_Policy, AState::Str_Agent_State)
    Func_update_world_model!(Agent, AState)
    Func_update_reward_matrices!(Agent, Agent.Func_eR_sas, AState, 
                                    extrinsic=true, ifinit = true)
    Func_update_reward_matrices!(Agent, Agent.Func_iR_sas, AState, 
                                    extrinsic=false, ifinit = true)
    Func_initialize_MB_eQ!(Agent,  AState)
    Func_initialize_MB_iQ!(Agent,  AState)
end
export Func_initialize_agent!

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Reseting agent after every episode
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_reset_agent_EpiEnd!(Agent::Str_Agent)
    # Eligibility traces are reset
    Agent.E_e .= 0
    Agent.E_i .= 0

    # Values for time > 0 (to be filled in next time-steps)
    Func_set_unknown_value!(Agent.S_t)
    Agent.A_t     = Func_unknown_values(Agent.Act_Set)                        # WARNING: There will be allocation if actions are arrays
    Func_set_unknown_value!(Agent.S_t_old)
    Agent.A_t_old = Func_unknown_values(Agent.Act_Set)                        # WARNING: There will be allocation if actions are arrays
    Agent.S_ind_t = -1
    Agent.A_ind_t = -1
    Agent.S_ind_t_old = -1
    Agent.A_ind_t_old = -1
    Agent.Q_MB_t .= -1
    Agent.Q_MF_t .= -1
    Agent.Q_t .= -1
    Agent.π_A_t  .= -1

    Agent.eR_t   = -1
    Agent.eRPE_t = -1
    Agent.iR_t   = -1
    Agent.iRPE_t = -1

    Agent.Epi = Agent.Epi + 1
    Agent.t_epi = 0

    Agent.new_state = false

    return Agent
end
function Func_reset_agent_EpiEnd!(Agent::Str_Agent_Policy)
    # Eligibility traces are reset
    Agent.E_e .= 0
    Agent.E_i .= 0

    # Values for time > 0 (to be filled in next time-steps)
    Agent.Q_MB_t .= -1
    Agent.Q_MF_t .= -1
    Agent.Q_t .= -1
    Agent.π_A_t  .= -1

    Agent.eR_t   = -1
    Agent.eRPE_t = -1
    Agent.iR_t   = -1
    Agent.iRPE_t = -1
    return Agent
end
export Func_reset_agent_EpiEnd!

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Initialize MB Q values
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_initialize_MB_eQ!(Agent::Str_Agent)
    # state_num = Agent.state_num
    # act_num = Agent.act_num

    # λ = Agent.Param.λ_e
    # ϵ_new = Agent.Param.ϵ_new
    # ϵ_obs = Agent.Param.ϵ_obs

    # W_obs = ϵ_obs / ( (1-λ) * ϵ_new + ϵ_obs * state_num)
    # W_new = ϵ_new / ( (1-λ) * ϵ_new + ϵ_obs * state_num)
    # θ_obs = ϵ_obs / ( ϵ_new + ϵ_obs * state_num)
    # θ_new = ϵ_new / ( ϵ_new + ϵ_obs * state_num)

    # R_sum = sum(Agent.eR_sas[1,1,:])                                          # WARNING: Assumption: there is no reward in unobserved states

    # x = θ_obs + λ * θ_new * W_obs
    # Q_MBe = x / ( 1 - λ * state_num * x ) * R_sum
    ##### WARNING #####
    Q_MBe = 0
    Agent.Q_MBe .= Q_MBe
    Agent.U_e   .= Q_MBe
end
function Func_initialize_MB_eQ!(Agent::Str_Agent_Policy, AState::Str_Agent_State)
    Q_MBe = 0
    Agent.Q_MBe .= Q_MBe
    Agent.U_e   .= Q_MBe
end


function Func_initialize_MB_iQ!(Agent::Str_Agent)
    state_num = Agent.state_num

    λ = Agent.Param.λ_i
    ϵ_new = Agent.Param.ϵ_new
    ϵ_obs = Agent.Param.ϵ_obs

    W_obs = ϵ_obs / ( (1-λ) * ϵ_new + ϵ_obs * state_num)
    W_new = ϵ_new / ( (1-λ) * ϵ_new + ϵ_obs * state_num)
    θ_obs = ϵ_obs / ( ϵ_new + ϵ_obs * state_num)
    θ_new = ϵ_new / ( ϵ_new + ϵ_obs * state_num)

    R_no = Agent.iR_sas[state_num+1,1,1]                                        # WARNING: Assumption: there is symmetry
    R_nn = Agent.iR_sas[state_num+1,1,state_num+1]
    R_oo = Agent.iR_sas[1,1,1]
    R_on = Agent.iR_sas[1,1,state_num+1]

    x0 = θ_obs + λ * θ_new * W_obs
    x0 = ( 1 - λ * state_num * x0 )

    x1 = θ_obs * state_num * R_oo

    x2 = R_on + λ*W_new*R_nn + λ*W_obs*R_no*state_num

    Q_MBi = (x1 + θ_new * x2) / x0

    Agent.Q_MBi .= Q_MBi
    Agent.U_i   .= Q_MBi
end
function Func_initialize_MB_iQ!(Agent::Str_Agent_Policy, AState::Str_Agent_State)
    state_num = AState.state_num - (AState.new_state * 1)

    λ = Agent.Param.λ_i
    ϵ_new = Agent.Param.ϵ_new
    ϵ_obs = Agent.Param.ϵ_obs

    W_obs = ϵ_obs / ( (1-λ) * ϵ_new + ϵ_obs * state_num)
    W_new = ϵ_new / ( (1-λ) * ϵ_new + ϵ_obs * state_num)
    θ_obs = ϵ_obs / ( ϵ_new + ϵ_obs * state_num)
    θ_new = ϵ_new / ( ϵ_new + ϵ_obs * state_num)

    R_no = Agent.iR_sas[state_num+1,1,1]                                        # WARNING: Assumption: there is symmetry
    R_nn = Agent.iR_sas[state_num+1,1,state_num+1]
    R_oo = Agent.iR_sas[1,1,1]
    R_on = Agent.iR_sas[1,1,state_num+1]

    x0 = θ_obs + λ * θ_new * W_obs
    x0 = ( 1 - λ * state_num * x0 )

    x1 = θ_obs * state_num * R_oo

    x2 = R_on + λ*W_new*R_nn + λ*W_obs*R_no*state_num

    Q_MBi = (x1 + θ_new * x2) / x0

    Agent.Q_MBi .= Q_MBi
    Agent.U_i   .= Q_MBi
end

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# 1st observation
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_first_obs!(Agent::Str_Agent;s=[0,0])
    state_num = Agent.state_num
    if s ∈ Agent.State_Set
        State_Set = Agent.State_Set
        s_ind::Int = findfirst(isequal(s),State_Set)
        Agent.S_ind_t = s_ind
        Agent.C_s[s_ind] = Agent.C_s[s_ind] + 1
        Agent.new_state = false
    else
        Agent.State_Set[state_num + 1] .= s                                     # WARNING: It doesn't work if initialized unknown states don't have the same dimension as the true states
        state_num = state_num + 1
        Agent.state_num = state_num
        s_ind = state_num
        Agent.S_ind_t = s_ind
        Agent.C_s[s_ind] = Agent.C_s[s_ind] + 1
        Agent.new_state = true
    end
    Func_update_world_model!(Agent)
    Func_update_reward_matrices!(Agent, Agent.Func_eR_sas, extrinsic=true)
    Func_update_reward_matrices!(Agent, Agent.Func_iR_sas, extrinsic=false)
    Func_update_Q_MB_PS!(Agent, extrinsic=true)
    Func_update_Q_MB_PS!(Agent, extrinsic=false)

    Agent.S_t .= s                                                              # WARNING: It doesn't work if initialized unknown states don't have the same dimension as the true states
    Agent.t_epi = Agent.t_epi + 1
    Agent.t_total = Agent.t_total + 1

    Func_compute_π_act!(Agent)
end
function Func_first_obs!(Agent::Str_Agent,s,s_ind,SSet,s_new,s_num)
    Agent.State_Set = SSet
    Agent.state_num = s_num
    Agent.new_state = s_new

    Agent.S_ind_t = s_ind
    Agent.C_s[s_ind] = Agent.C_s[s_ind] + 1

    Func_update_world_model!(Agent)
    Func_update_reward_matrices!(Agent, Agent.Func_eR_sas, extrinsic=true)
    Func_update_reward_matrices!(Agent, Agent.Func_iR_sas, extrinsic=false)
    Func_update_Q_MB_PS!(Agent, extrinsic=true)
    Func_update_Q_MB_PS!(Agent, extrinsic=false)

    Agent.S_t .= s                                                              # WARNING: It doesn't work if initialized unknown states don't have the same dimension as the true states
    Agent.t_epi   = Agent.t_epi + 1
    Agent.t_total = Agent.t_total + 1

    Func_compute_π_act!(Agent)
end
function Func_first_obs!(Agent::Str_Agent_Policy, AState::Str_Agent_State)
    s_ind = AState.S_ind_t
    Agent.C_s[s_ind] = Agent.C_s[s_ind] + 1

    Func_update_world_model!(Agent, AState)
    Func_update_reward_matrices!(Agent, Agent.Func_eR_sas, AState, extrinsic=true)
    Func_update_reward_matrices!(Agent, Agent.Func_iR_sas, AState, extrinsic=false)
    Func_update_Q_MB_PS!(Agent, AState, extrinsic=true)
    Func_update_Q_MB_PS!(Agent, AState, extrinsic=false)
    Func_compute_π_act!(Agent, AState)
end
export Func_first_obs!


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Agent consistency check
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_size_consistency_check(Agent::Str_Agent;verbose=false)
    an = length(Agent.Act_Set)
    sn = length(Agent.State_Set)

    if (length(Agent.C_s) != sn) ||
       (size(Agent.C_sa) != (sn,an)) ||
       (size(Agent.C_sas) != (sn,an,sn)) ||
       (size(Agent.θ_sas) != (sn,an,sn+1)) ||
       (size(Agent.eR_sas) != (sn+1,an,sn+1)) ||
       (size(Agent.iR_sas) != (sn+1,an,sn+1)) ||
       (size(Agent.Q_MBe) != (sn,an)) ||
       (length(Agent.U_e) != sn) ||
       (size(Agent.Q_MBi) != (sn,an)) ||
       (length(Agent.U_i) != sn) ||
       (size(Agent.Q_MFe) != (sn,an)) ||
       (size(Agent.E_e) != (sn,an)) ||
       (size(Agent.Q_MFi) != (sn,an)) ||
       (size(Agent.E_i) != (sn,an))

       error("There is a problem in size consistency!")
    else
       if verbose
           println("Size consistency test passed.")
       end
    end
end
export Func_size_consistency_check

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Agent consistency check
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_count_consistency_check(Agent::Str_Agent;verbose=false)
    an = Agent.act_num
    sn = Agent.state_num
    if findmax(abs.( Agent.C_sa .- sum(Agent.C_sas, dims=3)[:,:]))[1] > 1e-10
        error("There is a problem in count consistency!")
    else
        if verbose
            println("Count consistency test passed.")
        end
    end
end
export Func_count_consistency_check

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Recalling unknown values
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_unknown_values(x; unknown_val = -1)
    zero(x[1]) .+ unknown_val
end
export Func_unknown_values

function Func_set_unknown_value!(s; unknown_val = -1)
    for i = eachindex(s)
        s[i] = unknown_val
    end
end
export Func_set_unknown_value!

function Func_count_known_values(Set)
    i = 0; s_unknown = Func_unknown_values(Set)
    for j = eachindex(Set)
        i = i + (Set[j] != s_unknown) * 1
    end
    return i
end
export Func_count_known_values

