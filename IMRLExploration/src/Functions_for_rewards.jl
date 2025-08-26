# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# No intrinsic rewards
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_iR_sas_NoC(Agent,s,a,sp)
    return 0
end
export Func_iR_sas_NoC

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Novelty
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_iR_sas_Novelty(Agent,s,a,sp)
    state_num = Agent.state_num

    C_s = Agent.C_s
    SumC_s = sum(C_s)

    if sp != -1
        return (log(1 + 1*state_num +SumC_s) - log(1 + C_s[sp]))
    else
        return (log(1 + 1*state_num +SumC_s) - log(1))
    end
end
function Func_iR_sas_Novelty(Agent::Str_Agent_Policy, AState::Str_Agent_State,
                            s,a,sp; ifinit = false)
    #state_num = AState.state_num
    if ifinit
        state_num = AState.state_num - (AState.new_state * 1)
    else
        state_num = AState.state_num
    end

    C_s = Agent.C_s
    SumC_s = sum(C_s)

    if sp != -1
        return (log(1 + 1*state_num +SumC_s) - log(1 + C_s[sp]))
    else
        return (log(1 + 1*state_num +SumC_s) - log(1))
    end
end
export Func_iR_sas_Novelty
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Shannon surprise
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_iR_sas_Shannon_Surp(Agent,s,a,sp)
    state_num = Agent.state_num

    ϵ_new = Agent.Param.ϵ_new
    ϵ_obs = Agent.Param.ϵ_obs

    θ_sas = Agent.θ_sas

    if s != -1
        if sp != -1
            return -log(θ_sas[s,a,sp])
        else
            return -log(θ_sas[s,a,state_num+1])
        end
    else
        if sp != -1
            return log(ϵ_new + ϵ_obs * state_num) - log(ϵ_obs)
        else
            return log(ϵ_new + ϵ_obs * state_num) - log(ϵ_new)
        end
    end
end
function Func_iR_sas_Shannon_Surp(Agent::Str_Agent_Policy, 
                                AState::Str_Agent_State,s,a,sp; ifinit = false)
    #state_num = AState.state_num
    if ifinit
        state_num = AState.state_num - (AState.new_state * 1)
    else
        state_num = AState.state_num
    end

    ϵ_new = Agent.Param.ϵ_new
    ϵ_obs = Agent.Param.ϵ_obs

    θ_sas = Agent.θ_sas

    if s != -1
        if sp != -1
            return -log(θ_sas[s,a,sp])
        else
            return -log(θ_sas[s,a,state_num+1])
        end
    else
        if sp != -1
            return log(ϵ_new + ϵ_obs * state_num) - log(ϵ_obs)
        else
            return log(ϵ_new + ϵ_obs * state_num) - log(ϵ_new)
        end
    end
end
export Func_iR_sas_Shannon_Surp

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Information Gain
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_iR_sas_InfGain(Agent,s,a,sp)
    state_num = Agent.state_num

    ϵ_new = Agent.Param.ϵ_new
    ϵ_obs = Agent.Param.ϵ_obs

    C_sa = Agent.C_sa
    C_sas = Agent.C_sas
    θ_sas = Agent.θ_sas

    if s != -1
        if sp != -1
            x0 = ϵ_new + state_num*ϵ_obs + C_sa[s,a]
            x1 = ϵ_obs + C_sas[s,a,sp]
            θ = θ_sas[s,a,sp]
            return ( log( 1 + 1/x0 ) - θ * log(1 + 1/x1) )
        else
            x0 = ϵ_new + state_num*ϵ_obs + C_sa[s,a]
            return log( 1 + (1+ϵ_obs)/x0 )
        end

    else
        if sp != -1
            x0 = ϵ_new + state_num*ϵ_obs
            return ( log( 1 + 1/x0 ) - (ϵ_obs/x0) * log(1 + 1/ϵ_obs) )
        else
            x0 = ϵ_new + state_num*ϵ_obs
            return log( 1 + (1+ϵ_obs)/x0 )
        end
    end
end
function Func_iR_sas_InfGain(Agent::Str_Agent_Policy, AState::Str_Agent_State,
                            s,a,sp; ifinit = false)
    #state_num = AState.state_num
    if ifinit
        state_num = AState.state_num - (AState.new_state * 1)
    else
        state_num = AState.state_num
    end

    ϵ_new = Agent.Param.ϵ_new
    ϵ_obs = Agent.Param.ϵ_obs

    C_sa = Agent.C_sa
    C_sas = Agent.C_sas
    θ_sas = Agent.θ_sas

    if s != -1
        if sp != -1
            x0 = ϵ_new + state_num*ϵ_obs + C_sa[s,a]
            x1 = ϵ_obs + C_sas[s,a,sp]
            θ = θ_sas[s,a,sp]
            return ( log( 1 + 1/x0 ) - θ * log(1 + 1/x1) )
        else
            x0 = ϵ_new + state_num*ϵ_obs + C_sa[s,a]
            return log( 1 + (1+ϵ_obs)/x0 )
        end

    else
        if sp != -1
            x0 = ϵ_new + state_num*ϵ_obs
            return ( log( 1 + 1/x0 ) - (ϵ_obs/x0) * log(1 + 1/ϵ_obs) )
        else
            x0 = ϵ_new + state_num*ϵ_obs
            return log( 1 + (1+ϵ_obs)/x0 )
        end
    end
end
export Func_iR_sas_InfGain


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Combining reward functions
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_iR_sas_NIS(Agent,s,a,sp; wN = 1/3, wI = 1/3, wS = 1/3)
    N = Func_iR_sas_Novelty(Agent,s,a,sp)
    I = Func_iR_sas_InfGain(Agent,s,a,sp)
    S = Func_iR_sas_Shannon_Surp(Agent,s,a,sp)
    return wN * N + wI * I + wS * S
end
function Func_iR_sas_NIS(Agent::Str_Agent_Policy, AState::Str_Agent_State,s,a,sp;
                        wN = 1/3, wI = 1/3, wS = 1/3, ifinit = false)
    N = Func_iR_sas_Novelty(Agent,AState,s,a,sp, ifinit = ifinit)
    I = Func_iR_sas_InfGain(Agent,AState,s,a,sp, ifinit = ifinit)
    S = Func_iR_sas_Shannon_Surp(Agent,AState,s,a,sp, ifinit = ifinit)
    return wN * N + wI * I + wS * S
end
export Func_iR_sas_NIS

@concrete struct NIS_object
    ws#::Vector{Float64}
    N_sas#::Array{Float64,3}
    I_sas#::Array{Float64,3}
    S_sas#::Array{Float64,3}
end
function NIS_object(ws; n_state = 61, n_action = 3)
    if length(ws) != 3
        error("ws must have 3 elements.")
    end
    N_sas = zeros(n_state+1, n_action, n_state+1)
    I_sas = zeros(n_state+1, n_action, n_state+1)
    S_sas = zeros(n_state+1, n_action, n_state+1)
    NIS_object(ws,N_sas,I_sas,S_sas)
end
export NIS_object
