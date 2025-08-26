# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Building the world-model
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_update_world_model!(Agent::Str_Agent)
    state_num = Agent.state_num
    act_num = Agent.act_num

    C_sas = Agent.C_sas
    ϵ_new = Agent.Param.ϵ_new
    ϵ_obs = Agent.Param.ϵ_obs

    θ_sas = Agent.θ_sas
    if (!Agent.new_state)&
       (Agent.S_t_old ∈ Agent.State_Set)&
       (Agent.A_t_old ∈ Agent.Act_Set)&
       (!Agent.Param.back_leak)
        s = Agent.S_ind_t_old
        a = Agent.A_ind_t_old
        for sp = 1:state_num
            θ_sas[s,a,sp] = C_sas[s,a,sp] + ϵ_obs
        end
        θ_sas[s,a,state_num+1] = ϵ_new
        θ_sas[s,a,:] .= @views ( θ_sas[s,a,:] ./ sum(θ_sas[s,a,:]) )
    else
        for s = 1:state_num
            for a = 1:act_num
                for sp = 1:state_num
                    θ_sas[s,a,sp] = C_sas[s,a,sp] + ϵ_obs
                end
                θ_sas[s,a,state_num+1] = ϵ_new
                θ_sas[s,a,:] .= @views ( θ_sas[s,a,:] ./ sum(θ_sas[s,a,:]) )
            end
        end
    end
end
function Func_update_world_model!(Agent::Str_Agent_Policy, AState::Str_Agent_State)
    state_num = AState.state_num
    act_num = AState.act_num

    C_sas = Agent.C_sas
    ϵ_new = Agent.Param.ϵ_new
    ϵ_obs = Agent.Param.ϵ_obs

    θ_sas = Agent.θ_sas
    if (!AState.new_state)&
       (AState.S_t_old ∈ AState.State_Set)&
       (AState.A_t_old ∈ AState.Act_Set)&
       (!Agent.Param.back_leak)
        s = AState.S_ind_t_old
        a = AState.A_ind_t_old
        for sp = 1:state_num
            θ_sas[s,a,sp] = C_sas[s,a,sp] + ϵ_obs
        end
        θ_sas[s,a,state_num+1] = ϵ_new
        θ_sas[s,a,:] .= @views ( θ_sas[s,a,:] ./ sum(θ_sas[s,a,:]) )
    else
        for s = 1:state_num
            for a = 1:act_num
                for sp = 1:state_num
                    θ_sas[s,a,sp] = C_sas[s,a,sp] + ϵ_obs
                end
                θ_sas[s,a,state_num+1] = ϵ_new
                θ_sas[s,a,:] .= @views ( θ_sas[s,a,:] ./ sum(θ_sas[s,a,:]) )
            end
        end
    end
end
export Func_update_world_model!

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Update Reward matrices
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# function Func_update_reward_matrices!(Agent::Str_Agent, Func_R_sas; extrinsic::Bool=true)
#     state_num = Agent.state_num
#     act_num   = Agent.act_num
#     if extrinsic
#         R_sas = Agent.eR_sas
#     else
#         R_sas = Agent.iR_sas
#     end

#     for a = 1:act_num
#         # s ∈ S^{(t)} and s' ∈ S^{(t)}
#         for s = 1:state_num
#             for sp = 1:state_num
#                 R_sas[s,a,sp] = Func_R_sas(Agent, s, a, sp)
#             end
#             # s ∈ S^{(t)} but s' ∉ S^{(t)}
#             R_sas[s,a,state_num+1] = Func_R_sas(Agent, s, a, -1)
#         end
#         # s ∉ S^{(t)} but s' ∈ S^{(t)}
#         for sp = 1:state_num
#             R_sas[state_num+1,a,sp] = Func_R_sas(Agent, -1, a, sp)
#         end
#         # s ∉ S^{(t)} and s' ∉ S^{(t)}
#         R_sas[state_num+1,a,state_num+1] = Func_R_sas(Agent, -1, a, -1)
#     end
# end

# # Surprise
# function Func_update_reward_matrices!(Agent::Str_Agent, 
#                             Func_R_sas::typeof(Func_iR_sas_Shannon_Surp); 
#                             extrinsic::Bool=true)
#     state_num = Agent.state_num
#     act_num   = Agent.act_num
#     if extrinsic
#         R_sas = Agent.eR_sas
#     else
#         R_sas = Agent.iR_sas
#     end

#     State_Set = Agent.State_Set
#     Act_Set   = Agent.Act_Set

#     if (!Agent.new_state)&
#        (Agent.A_t_old ∈ Act_Set)&
#        (Agent.S_t_old ∈ State_Set)
#         s = Agent.S_ind_t_old
#         a = Agent.A_ind_t_old
#         for sp = 1:state_num
#             R_sas[s,a,sp] = Func_R_sas(Agent, s, a, sp)
#         end
#         # s' ∉ S^{(t)}
#         R_sas[s,a,state_num+1] = Func_R_sas(Agent, s, a, -1)
#     else
#         for a = 1:act_num
#             # s ∈ S^{(t)} and s' ∈ S^{(t)}
#             for s = 1:state_num
#                 for sp = 1:state_num
#                     R_sas[s,a,sp] = Func_R_sas(Agent, s, a, sp)
#                 end
#                 # s ∈ S^{(t)} but s' ∉ S^{(t)}
#                 R_sas[s,a,state_num+1] = Func_R_sas(Agent, s, a, -1)
#             end
#             # s ∉ S^{(t)} but s' ∈ S^{(t)}
#             for sp = 1:state_num
#                 R_sas[state_num+1,a,sp] =
#                     Func_R_sas(Agent, -1, a, sp)
#             end
#             # s ∉ S^{(t)} and s' ∉ S^{(t)}
#             R_sas[state_num+1,a,state_num+1] =
#                 Func_R_sas(Agent, -1, a, -1)
#         end
#     end
# end
# # Inf Gain
# function Func_update_reward_matrices!(Agent::Str_Agent, 
#                             Func_R_sas::typeof(Func_iR_sas_InfGain); 
#                             extrinsic::Bool=true)
#     state_num = Agent.state_num
#     act_num   = Agent.act_num
#     if extrinsic
#         R_sas = Agent.eR_sas
#     else
#         R_sas = Agent.iR_sas
#     end

#     State_Set = Agent.State_Set
#     Act_Set   = Agent.Act_Set

#     if (!Agent.new_state)&
#        (Agent.A_t_old ∈ Act_Set)&
#        (Agent.S_t_old ∈ State_Set)
#         s = Agent.S_ind_t_old
#         a = Agent.A_ind_t_old
#         for sp = 1:state_num
#             R_sas[s,a,sp] = Func_R_sas(Agent, s, a, sp)
#         end
#         # s' ∉ S^{(t)}
#         R_sas[s,a,state_num+1] = Func_R_sas(Agent, s, a, -1)
#     else
#         for a = 1:act_num
#             # s ∈ S^{(t)} and s' ∈ S^{(t)}
#             for s = 1:state_num
#                 for sp = 1:state_num
#                     R_sas[s,a,sp] = Func_R_sas(Agent, s, a, sp)
#                 end
#                 # s ∈ S^{(t)} but s' ∉ S^{(t)}
#                 R_sas[s,a,state_num+1] = Func_R_sas(Agent, s, a, -1)
#             end
#             # s ∉ S^{(t)} but s' ∈ S^{(t)}
#             for sp = 1:state_num
#                 R_sas[state_num+1,a,sp] =
#                     Func_R_sas(Agent, -1, a, sp)
#             end
#             # s ∉ S^{(t)} and s' ∉ S^{(t)}
#             R_sas[state_num+1,a,state_num+1] =
#                 Func_R_sas(Agent, -1, a, -1)
#         end
#     end
# end
# # Novelty
# function Func_update_reward_matrices!(Agent::Str_Agent, 
#                             Func_R_sas::typeof(Func_iR_sas_Novelty); 
#                             extrinsic::Bool=true)
#     state_num = Agent.state_num
#     if extrinsic
#         R_sas = Agent.eR_sas
#     else
#         R_sas = Agent.iR_sas
#     end

#     for sp = 1:state_num
#         R_sas[:,:,sp] .= Func_R_sas(Agent, 1, 1, sp)
#     end
#     R_sas[:,:,state_num+1] .= Func_R_sas(Agent, 1, 1, -1)
# end
# Novelty + Surprise + Reward
function Func_update_reward_matrices!(Agent::Str_Agent, Func_R_sas::NIS_object; 
                                        extrinsic::Bool=true)
    state_num = Agent.state_num
    act_num   = Agent.act_num
    State_Set = Agent.State_Set
    Act_Set   = Agent.Act_Set
    
    # Novelty computation
    for sp = 1:state_num
        Func_R_sas.N_sas[:,:,sp] .= 
            Func_iR_sas_Novelty(Agent, 1, 1, sp)
    end
    Func_R_sas.N_sas[:,:,state_num+1] .=  
        Func_iR_sas_Novelty(Agent, 1, 1, -1)

    # Inf Gain + Surprise computation
    if (!Agent.new_state)&
       (Agent.A_t_old ∈ Act_Set)&
       (Agent.S_t_old ∈ State_Set)&
       (!Agent.Param.back_leak)
        s = Agent.S_ind_t_old
        a = Agent.A_ind_t_old
        for sp = 1:state_num
            Func_R_sas.I_sas[s,a,sp] = 
                Func_iR_sas_InfGain(Agent, s, a, sp)
            Func_R_sas.S_sas[s,a,sp] = 
                Func_iR_sas_Shannon_Surp(Agent, s, a, sp)
        end
        # s' ∉ S^{(t)}
        Func_R_sas.I_sas[s,a,state_num+1] = 
            Func_iR_sas_InfGain(Agent, s,a, -1)
        Func_R_sas.S_sas[s,a,state_num+1] = 
            Func_iR_sas_Shannon_Surp(Agent, s,a, -1)
    else
        for a = 1:act_num
            # s ∈ S^{(t)} and s' ∈ S^{(t)}
            for s = 1:state_num
                for sp = 1:state_num
                    Func_R_sas.I_sas[s,a,sp] = 
                        Func_iR_sas_InfGain(Agent, s, a,sp)
                    Func_R_sas.S_sas[s,a,sp] = 
                        Func_iR_sas_Shannon_Surp(Agent, s, a,sp)
                end
                # s ∈ S^{(t)} but s' ∉ S^{(t)}
                Func_R_sas.I_sas[s,a,state_num+1] = 
                    Func_iR_sas_InfGain(Agent, s, a, -1)
                Func_R_sas.S_sas[s,a,state_num+1] = 
                    Func_iR_sas_Shannon_Surp(Agent, s, a, -1)
            end
            # s ∉ S^{(t)} but s' ∈ S^{(t)}
            for sp = 1:state_num
                Func_R_sas.I_sas[state_num+1,a,sp] = 
                    Func_iR_sas_InfGain(Agent, -1, a, sp)
                Func_R_sas.S_sas[state_num+1,a,sp] = 
                    Func_iR_sas_Shannon_Surp(Agent, -1, a, sp)
            end
            # s ∉ S^{(t)} and s' ∉ S^{(t)}
            Func_R_sas.I_sas[state_num+1,a,state_num+1] = 
                Func_iR_sas_InfGain(Agent, -1, a, -1)
            Func_R_sas.S_sas[state_num+1,a,state_num+1] = 
                Func_iR_sas_Shannon_Surp(Agent, -1, a, -1)
        end
    end
    Agent.Func_iR_sas = Func_R_sas
    if extrinsic
        Agent.eR_sas .= @views ((Func_R_sas.ws[1] .* Func_R_sas.N_sas) .+
                                (Func_R_sas.ws[2] .* Func_R_sas.I_sas) .+
                                (Func_R_sas.ws[3] .* Func_R_sas.S_sas))
    else
        Agent.iR_sas .= @views ((Func_R_sas.ws[1] .* Func_R_sas.N_sas) .+
                                (Func_R_sas.ws[2] .* Func_R_sas.I_sas) .+
                                (Func_R_sas.ws[3] .* Func_R_sas.S_sas))
    end
end
function Func_update_reward_matrices!(Agent::Str_Agent_Policy, Func_R_sas::NIS_object, 
                                    AState::Str_Agent_State; extrinsic::Bool=true,
                                    ifinit = false)
    if ifinit
        state_num = AState.state_num - (AState.new_state * 1)
    else
        state_num = AState.state_num
    end
    act_num   = AState.act_num
    State_Set = AState.State_Set
    Act_Set   = AState.Act_Set
    
    # Novelty computation
    for sp = 1:state_num
        Func_R_sas.N_sas[:,:,sp] .= Func_iR_sas_Novelty(Agent, AState, 1, 1, sp, 
                                                        ifinit = ifinit)
    end
    Func_R_sas.N_sas[:,:,state_num+1] .= Func_iR_sas_Novelty(Agent, AState, 1, 1, -1,
                                                        ifinit = ifinit)

    # Inf Gain + Surprise computation
    if (!AState.new_state)&
       (AState.A_t_old ∈ Act_Set)&
       (AState.S_t_old ∈ State_Set)&
       (!Agent.Param.back_leak)
        s = AState.S_ind_t_old
        a = AState.A_ind_t_old
        for sp = 1:state_num
            Func_R_sas.I_sas[s,a,sp] = 
                Func_iR_sas_InfGain(Agent, AState, s, a, sp, ifinit = ifinit)
            Func_R_sas.S_sas[s,a,sp] = 
                Func_iR_sas_Shannon_Surp(Agent, AState, s, a, sp, ifinit = ifinit)
        end
        # s' ∉ S^{(t)}
        Func_R_sas.I_sas[s,a,state_num+1] = 
            Func_iR_sas_InfGain(Agent, AState, s,a, -1, ifinit = ifinit)
        Func_R_sas.S_sas[s,a,state_num+1] = 
            Func_iR_sas_Shannon_Surp(Agent, AState, s,a, -1, ifinit = ifinit)
    else
        for a = 1:act_num
            # s ∈ S^{(t)} and s' ∈ S^{(t)}
            for s = 1:state_num
                for sp = 1:state_num
                    Func_R_sas.I_sas[s,a,sp] = 
                        Func_iR_sas_InfGain(Agent, AState, s, a,sp, ifinit = ifinit)
                    Func_R_sas.S_sas[s,a,sp] = 
                        Func_iR_sas_Shannon_Surp(Agent, AState, s, a,sp, ifinit = ifinit)
                end
                # s ∈ S^{(t)} but s' ∉ S^{(t)}
                Func_R_sas.I_sas[s,a,state_num+1] = 
                    Func_iR_sas_InfGain(Agent, AState, s, a, -1, ifinit = ifinit)
                Func_R_sas.S_sas[s,a,state_num+1] = 
                    Func_iR_sas_Shannon_Surp(Agent, AState, s, a, -1, ifinit = ifinit)
            end
            # s ∉ S^{(t)} but s' ∈ S^{(t)}
            for sp = 1:state_num
                Func_R_sas.I_sas[state_num+1,a,sp] = 
                    Func_iR_sas_InfGain(Agent, AState, -1, a, sp, ifinit = ifinit)
                Func_R_sas.S_sas[state_num+1,a,sp] = 
                    Func_iR_sas_Shannon_Surp(Agent, AState, -1, a, sp, ifinit = ifinit)
            end
            # s ∉ S^{(t)} and s' ∉ S^{(t)}
            Func_R_sas.I_sas[state_num+1,a,state_num+1] = 
                Func_iR_sas_InfGain(Agent, AState, -1, a, -1, ifinit = ifinit)
            Func_R_sas.S_sas[state_num+1,a,state_num+1] = 
                Func_iR_sas_Shannon_Surp(Agent, AState, -1, a, -1, ifinit = ifinit)
        end
    end
    Agent.Func_iR_sas = Func_R_sas
    if extrinsic
        Agent.eR_sas .= @views ((Func_R_sas.ws[1] .* Func_R_sas.N_sas) .+
                                (Func_R_sas.ws[2] .* Func_R_sas.I_sas) .+
                                (Func_R_sas.ws[3] .* Func_R_sas.S_sas))
    else
        Agent.iR_sas .= @views ((Func_R_sas.ws[1] .* Func_R_sas.N_sas) .+
                                (Func_R_sas.ws[2] .* Func_R_sas.I_sas) .+
                                (Func_R_sas.ws[3] .* Func_R_sas.S_sas))
    end
end
# Exstrinsic Reward
function Func_update_reward_matrices!(Agent::Str_Agent, Func_R_sas::InfEnv_Rewards; 
                                    extrinsic::Bool=true)
    state_num = Agent.state_num
    if extrinsic
        R_sas = Agent.eR_sas
    else
        R_sas = Agent.iR_sas
    end
    
    # State_Set = Agent.State_Set
    # for sp = 1:state_num
    #     if State_Set[sp] ∈ Func_R_sas.Goal_states
    #         R_sas[:,:,sp] .= 
    #                 Func_R_sas.Rs[findfirst(isequal(State_Set[sp]),
    #                                                 Func_R_sas.Goal_states)]
    #     end
    # end
    # Assuming that the goal states are the firs states of R_sas
    for sp = eachindex(Func_R_sas.Goal_states)
        R_sas[:,:,sp] .= Func_R_sas.Rs[sp]
    end
    R_sas[:,:,state_num+1] .= 0.
end
function Func_update_reward_matrices!(Agent::Str_Agent_Policy, 
                            Func_R_sas::InfEnv_Rewards, AState::Str_Agent_State; 
                            extrinsic::Bool=true, ifinit = false)
    state_num = AState.state_num
    if extrinsic
        R_sas = Agent.eR_sas
    else
        R_sas = Agent.iR_sas
    end
    # Assuming that the goal states are the firs states of R_sas
    for sp = eachindex(Func_R_sas.Rs)
        R_sas[:,:,sp] .= Func_R_sas.Rs[sp]
    end
    R_sas[:,:,state_num+1] .= 0.
end
export Func_update_reward_matrices!
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Prioritized sweeping for updating MB Q-values
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_update_Q_MB_PS!(Agent::Str_Agent; extrinsic::Bool=true,
                                    ΔV_thresh = 1e-2, θ_thresh = 1e-3)
    ϵ_new = Agent.Param.ϵ_new
    ϵ_obs = Agent.Param.ϵ_obs
    state_num = Agent.state_num
    act_num   = Agent.act_num
    θ_sas = Agent.θ_sas

    if extrinsic
        U = Agent.U_e; Q = Agent.Q_MBe
        R_sas = Agent.eR_sas
        λ = Agent.Param.λ_e; T_PS = Agent.Param.T_PS_e
    else
        U = Agent.U_i; Q = Agent.Q_MBi
        R_sas = Agent.iR_sas
        λ = Agent.Param.λ_i; T_PS = Agent.Param.T_PS_i
    end

    W_obs = ϵ_obs / ( (1-λ) * ϵ_new + ϵ_obs * state_num)
    W_new = ϵ_new / ( (1-λ) * ϵ_new + ϵ_obs * state_num)

    if Agent.new_state
        U_new = @views (
                        W_obs * (sum(R_sas[state_num+1,1,1:(state_num-1)]) + 
                                    λ * sum(U[1:(state_num-1)])) +
                        W_new * R_sas[state_num+1,1,state_num+1] )
        U[state_num] = U_new
        Q[state_num,:] .= U_new
    end

    # Calculating Q
    for s = 1:state_num
        for a = 1:act_num
            Q[s,a] = Func_QUpdate(s,a,λ,θ_sas,U,R_sas,W_obs,W_new,state_num)
        end
    end

    V = Agent.V_dummy
    Prior = Agent.P_dummy
    V .= 0; Prior .= 0
    # Calculating V
    for s = 1:state_num
        V[s] = @views (findmax(Q[s,:])[1])
        Prior[s] = abs(V[s] - U[s])
    end

    for i_cycle = 1:T_PS
        sp = @views (findmax(Prior[1:state_num])[2])
        ΔV = V[sp] - U[sp]
        if (abs(ΔV)/abs(findmax(@view V[1:state_num])[1] - findmin(@view V[1:state_num])[1])) <= ΔV_thresh
            break
        else
            U[sp] = V[sp]
            for s = 1:state_num
                if (θ_thresh==0)||
                   (Agent.new_state)||
                    ((sum(@view θ_sas[s,:,sp ])+
                      sum(@view θ_sas[s,:,state_num+1]))*abs(ΔV) > θ_thresh)
                    for a = 1:act_num
                        Q[s,a] += λ*(θ_sas[s,a,sp] + 
                                     λ*W_obs*θ_sas[s,a,state_num+1])*ΔV
                    end
                    V[s] = @views (findmax(Q[s,:])[1])
                    Prior[s] = abs(V[s] - U[s])
                end
            end
        end
    end
    return
end
function Func_update_Q_MB_PS!(Agent::Str_Agent_Policy, AState::Str_Agent_State; 
                extrinsic::Bool=true, ΔV_thresh = 1e-2, θ_thresh = 1e-3)
    state_num = AState.state_num
    act_num   = AState.act_num
    
    ϵ_new = Agent.Param.ϵ_new
    ϵ_obs = Agent.Param.ϵ_obs

    θ_sas = Agent.θ_sas

    if extrinsic
        U = Agent.U_e; Q = Agent.Q_MBe
        R_sas = Agent.eR_sas
        λ = Agent.Param.λ_e; T_PS = Agent.Param.T_PS_e
    else
        U = Agent.U_i; Q = Agent.Q_MBi
        R_sas = Agent.iR_sas
        λ = Agent.Param.λ_i; T_PS = Agent.Param.T_PS_i
    end

    W_obs = ϵ_obs / ( (1-λ) * ϵ_new + ϵ_obs * state_num)
    W_new = ϵ_new / ( (1-λ) * ϵ_new + ϵ_obs * state_num)

    if AState.new_state
        U_new = @views (
                        W_obs * (sum(R_sas[state_num+1,1,1:(state_num-1)]) + 
                                    λ * sum(U[1:(state_num-1)])) +
                        W_new * R_sas[state_num+1,1,state_num+1] )
        U[state_num] = U_new
        Q[state_num,:] .= U_new
    end

    # Calculating Q
    for s = 1:state_num
        for a = 1:act_num
            Q[s,a] = Func_QUpdate(s,a,λ,θ_sas,U,R_sas,W_obs,W_new,state_num)
        end
    end

    V = Agent.V_dummy
    Prior = Agent.P_dummy
    V .= 0; Prior .= 0
    # Calculating V
    for s = 1:state_num
        V[s] = @views (findmax(Q[s,:])[1])
        Prior[s] = abs(V[s] - U[s])
    end

    for i_cycle = 1:T_PS
        sp = @views (findmax(Prior[1:state_num])[2])
        ΔV = V[sp] - U[sp]
        if (abs(ΔV)/abs(findmax(@view V[1:state_num])[1] - findmin(@view V[1:state_num])[1])) <= ΔV_thresh
            break
        else
            U[sp] = V[sp]
            for s = 1:state_num
                if (θ_thresh==0)||
                   (AState.new_state)||
                    ((sum(@view θ_sas[s,:,sp ])+
                      sum(@view θ_sas[s,:,state_num+1]))*abs(ΔV) > θ_thresh)
                    for a = 1:act_num
                        Q[s,a] += λ*(θ_sas[s,a,sp] + 
                                     λ*W_obs*θ_sas[s,a,state_num+1])*ΔV
                    end
                    V[s] = @views (findmax(Q[s,:])[1])
                    Prior[s] = abs(V[s] - U[s])
                end
            end
        end
    end
    return
end
export Func_update_Q_MB_PS!


function Func_QUpdate(s,a,λ,θ_sas,U,R_sas,W_obs,W_new,state_num)
    θ_sa     = @view θ_sas[s,a,1:state_num]
    θ_sa_new = θ_sas[s,a,state_num+1]

    R_sa     = @view R_sas[s,a,1:state_num]
    R_sa_new = R_sas[s,a,state_num+1]

    R_new_s   = @view R_sas[state_num+1, 1, 1:state_num]
    R_new_new = R_sas[state_num+1, 1, state_num+1]

    x0 = dot(θ_sa, R_sa)
    x1 = λ * ((dot(θ_sa, @view U[1:state_num])) + 
              λ*θ_sa_new*W_obs*sum(@view U[1:state_num]))
    x2 = @views θ_sa_new * ( R_sa_new + λ*W_new*R_new_new + λ*W_obs*sum(R_new_s) )
    return (x0+x1+x2)
end

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Computing Action Probability
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_compute_π_act!(Agent::Str_Agent)
    π_A_t = Agent.π_A_t
    if Agent.Epi == 1
        β_MBe = Agent.Param.β_MBe_1; β_MBi = Agent.Param.β_MBi_1
        β_MFe = Agent.Param.β_MFe_1; β_MFi = Agent.Param.β_MFi_1
    else
        β_MBe = Agent.Param.β_MBe_2; β_MBi = Agent.Param.β_MBi_2
        β_MFe = Agent.Param.β_MFe_2; β_MFi = Agent.Param.β_MFi_2
    end

    s_ind = Agent.S_ind_t

    Agent.Q_MB_t .= @views (β_MBe .* Agent.Q_MBe[s_ind,:] .+ β_MBi .* Agent.Q_MBi[s_ind,:])
    Agent.Q_MF_t .= @views (β_MFe .* Agent.Q_MFe[s_ind,:] .+ β_MFi .* Agent.Q_MFi[s_ind,:])
    Agent.Q_t .= Agent.Q_MB_t .+ Agent.Q_MF_t
    Agent.Q_t[2:end] .+= Agent.Param.Q_bias
    Q = Agent.Q_t

    π_A_t .= exp.(Q .- findmax(Q)[1])
    π_A_t .= π_A_t ./ sum(π_A_t)
end
function Func_compute_π_act!(Agent::Str_Agent_Policy, AState::Str_Agent_State)
    π_A_t = Agent.π_A_t
    if AState.Epi == 1
        β_MBe = Agent.Param.β_MBe_1; β_MBi = Agent.Param.β_MBi_1
        β_MFe = Agent.Param.β_MFe_1; β_MFi = Agent.Param.β_MFi_1
    else
        β_MBe = Agent.Param.β_MBe_2; β_MBi = Agent.Param.β_MBi_2
        β_MFe = Agent.Param.β_MFe_2; β_MFi = Agent.Param.β_MFi_2
    end

    s_ind = AState.S_ind_t

    Agent.Q_MB_t .= @views (β_MBe .* Agent.Q_MBe[s_ind,:] .+ β_MBi .* Agent.Q_MBi[s_ind,:])
    Agent.Q_MF_t .= @views (β_MFe .* Agent.Q_MFe[s_ind,:] .+ β_MFi .* Agent.Q_MFi[s_ind,:])
    Agent.Q_t .= Agent.Q_MB_t .+ Agent.Q_MF_t
    Agent.Q_t[2:end] .+= Agent.Param.Q_bias
    Q = Agent.Q_t

    π_A_t .= exp.(Q .- findmax(Q)[1])
    π_A_t .= π_A_t ./ sum(π_A_t)
end
export Func_compute_π_act!

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Sampling or loading actions
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_sample_action!(Agent::Str_Agent; rng=Random.GLOBAL_RNG)
    π_A_t = Agent.π_A_t
    a_ind = rand(rng,Categorical(π_A_t))
    Agent.A_t = Agent.Act_Set[a_ind]
    Agent.A_ind_t = a_ind
end
export Func_sample_action!

function Func_sample_action(Agent::Str_Agent_Policy, AState::Str_Agent_State; 
                            rng=Random.GLOBAL_RNG)
    π_A_t = Agent.π_A_t
    A_ind_t = rand(rng,Categorical(π_A_t))
    A_t = AState.Act_Set[A_ind_t]
    return A_ind_t, A_t
end
export Func_sample_action

function Func_load_action!(Agent::Str_Agent;a=1)
    Agent.A_t = a
    a_ind::Int = findfirst(isequal(a),Agent.Act_Set)
    Agent.A_ind_t = a_ind
    return log(Agent.π_A_t[a_ind])
end
function Func_load_action!(Agent::Str_Agent,a,a_ind)
    Agent.A_t = a
    Agent.A_ind_t = a_ind
    return log(Agent.π_A_t[a_ind])
end
function Func_load_action!(Agent::Str_Agent_Policy, AState::Str_Agent_State)
    a_ind = AState.A_ind_t
    return log(Agent.π_A_t[a_ind])
end
export Func_load_action!

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Observing the new state
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_observe_state!(Agent::Str_Agent;sp=[0,0])
    state_num = Agent.state_num
    State_Set = Agent.State_Set

    s_ind = Agent.S_ind_t
    a_ind = Agent.A_ind_t

    if sp ∈ Agent.State_Set
        sp_ind::Int = findfirst(isequal(sp),State_Set)
        Agent.new_state = false
    else
        Agent.State_Set[state_num + 1] .= sp
        state_num = state_num + 1
        sp_ind = state_num
        Agent.new_state = true
    end
    eR_t = Agent.eR_sas[s_ind,a_ind,sp_ind]
    iR_t = Agent.iR_sas[s_ind,a_ind,sp_ind]

    Agent.eR_t = eR_t
    Agent.iR_t = iR_t
    
    Agent.S_t_old .= Agent.S_t                                                  # WARNING: It doesn't work if initialized unknown states don't have the same dimension as the true states
    Agent.A_t_old  = Agent.A_t                                                  # WARNING: There will be allocation if actions are arrays
    Agent.A_t = Func_unknown_values(Agent.Act_Set)                              # WARNING: There will be allocation if actions are arrays
    Agent.S_t .= sp                                                             # WARNING: It doesn't work if initialized unknown states don't have the same dimension as the true states
    
    Agent.S_ind_t_old = s_ind
    Agent.A_ind_t_old = a_ind
    Agent.A_ind_t = -1
    Agent.S_ind_t = sp_ind
    
    Agent.state_num = state_num
    

    Agent.t_epi   = Agent.t_epi + 1
    Agent.t_total = Agent.t_total + 1

    Agent.eR_max_t = max(Agent.eR_max_t, Agent.eR_t)
end
function Func_observe_state!(Agent::Str_Agent,sp,sp_ind,SSet,s_new,s_num)
    Agent.State_Set = SSet
    Agent.state_num = s_num
    Agent.new_state = s_new

    s_ind = Agent.S_ind_t
    a_ind = Agent.A_ind_t

    eR_t = Agent.eR_sas[s_ind,a_ind,sp_ind]
    iR_t = Agent.iR_sas[s_ind,a_ind,sp_ind]

    Agent.eR_t = eR_t
    Agent.iR_t = iR_t
    
    Agent.S_t_old .= Agent.S_t                                                  # WARNING: It doesn't work if initialized unknown states don't have the same dimension as the true states
    Agent.A_t_old  = Agent.A_t                                                  # WARNING: There will be allocation if actions are arrays
    Agent.A_t = Func_unknown_values(Agent.Act_Set)                              # WARNING: There will be allocation if actions are arrays
    Agent.S_t .= sp                                                             # WARNING: It doesn't work if initialized unknown states don't have the same dimension as the true states
    
    Agent.S_ind_t_old = s_ind
    Agent.A_ind_t_old = a_ind
    Agent.A_ind_t = -1
    Agent.S_ind_t = sp_ind
    

    Agent.t_epi   = Agent.t_epi + 1
    Agent.t_total = Agent.t_total + 1

    Agent.eR_max_t = max(Agent.eR_max_t, Agent.eR_t)
end
function Func_observe_state!(Agent::Str_Agent_Policy, AState::Str_Agent_State)
    s_ind  = AState.S_ind_t_old
    a_ind  = AState.A_ind_t_old
    sp_ind = AState.S_ind_t

    eR_t = Agent.eR_sas[s_ind,a_ind,sp_ind]
    iR_t = Agent.iR_sas[s_ind,a_ind,sp_ind]

    Agent.eR_t = eR_t
    Agent.iR_t = iR_t
    Agent.eR_max_t = max(Agent.eR_max_t, Agent.eR_t)
end
export Func_observe_state!

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Updating counts
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_update_counts!(Agent::Str_Agent)
    κ = Agent.Param.κ

    a_ind  = Agent.A_ind_t_old
    s_ind  = Agent.S_ind_t_old
    sp_ind = Agent.S_ind_t

    if Agent.Param.total_leak
        Agent.C_s .= κ .* Agent.C_s
    else
        Agent.C_s .= Agent.C_s
    end
    Agent.C_s[sp_ind] = Agent.C_s[sp_ind] + 1

    if Agent.Param.back_leak
        Agent.C_sa  .= @views (κ .* Agent.C_sa)
        Agent.C_sas .= @views (κ .* Agent.C_sas)
    else
        Agent.C_sa[s_ind,a_ind] = κ * Agent.C_sa[s_ind,a_ind]
        Agent.C_sas[s_ind,a_ind,:] .= @views (κ .* Agent.C_sas[s_ind,a_ind,:])
    end
    Agent.C_sa[s_ind,a_ind] = Agent.C_sa[s_ind,a_ind] + 1
    Agent.C_sas[s_ind,a_ind,sp_ind] = Agent.C_sas[s_ind,a_ind,sp_ind] + 1
end
function Func_update_counts!(Agent::Str_Agent_Policy, AState::Str_Agent_State)
    κ = Agent.Param.κ

    a_ind  = AState.A_ind_t_old
    s_ind  = AState.S_ind_t_old
    sp_ind = AState.S_ind_t

    if Agent.Param.total_leak
        Agent.C_s .= κ .* Agent.C_s
    else
        Agent.C_s .= Agent.C_s
    end
    Agent.C_s[sp_ind] = Agent.C_s[sp_ind] + 1

    if Agent.Param.back_leak
        Agent.C_sa  .= @views (κ .* Agent.C_sa)
        Agent.C_sas .= @views (κ .* Agent.C_sas)
    else
        Agent.C_sa[s_ind,a_ind] = κ * Agent.C_sa[s_ind,a_ind]
        Agent.C_sas[s_ind,a_ind,:] .= @views (κ .* Agent.C_sas[s_ind,a_ind,:])
    end
    Agent.C_sa[s_ind,a_ind] = Agent.C_sa[s_ind,a_ind] + 1
    Agent.C_sas[s_ind,a_ind,sp_ind] = Agent.C_sas[s_ind,a_ind,sp_ind] + 1
end
export Func_update_counts!

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Compute RPEs
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_compute_RPE!(Agent::Str_Agent; extrinsic::Bool=true)
    a_ind  = Agent.A_ind_t_old
    s_ind  = Agent.S_ind_t_old
    sp_ind = Agent.S_ind_t
    if extrinsic
        Q = Agent.Q_MFe; R = Agent.eR_t; λ = Agent.Param.λ_e
    else
        Q = Agent.Q_MFi; R = Agent.iR_t; λ = Agent.Param.λ_i
    end

    Q_old = Q[s_ind,a_ind]                              # Q(s_{t-1},a_{t-1})
    V_new = findmax(@view Q[sp_ind,:] )[1]              # max_{a'} Q(s_{t},a')

    δ = R + λ*V_new - Q_old

    if extrinsic
        Agent.eRPE_t = δ
    else
        Agent.iRPE_t = δ
    end
end
function Func_compute_RPE!(Agent::Str_Agent_Policy, AState::Str_Agent_State; 
                            extrinsic::Bool=true)
    a_ind  = AState.A_ind_t_old
    s_ind  = AState.S_ind_t_old
    sp_ind = AState.S_ind_t
    if extrinsic
        Q = Agent.Q_MFe; R = Agent.eR_t; λ = Agent.Param.λ_e
    else
        Q = Agent.Q_MFi; R = Agent.iR_t; λ = Agent.Param.λ_i
    end

    Q_old = Q[s_ind,a_ind]                              # Q(s_{t-1},a_{t-1})
    V_new = findmax(@view Q[sp_ind,:] )[1]              # max_{a'} Q(s_{t},a')

    δ = R + λ*V_new - Q_old

    if extrinsic
        Agent.eRPE_t = δ
    else
        Agent.iRPE_t = δ
    end
end
export Func_compute_RPE!

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Update Eligibility Trace
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_update_ElTrace!(Agent::Str_Agent; extrinsic::Bool=true)
    a_ind  = Agent.A_ind_t_old
    s_ind  = Agent.S_ind_t_old
    if extrinsic
        E = Agent.E_e; λ = Agent.Param.λ_e; μ = Agent.Param.μ_e
    else
        E = Agent.E_i; λ = Agent.Param.λ_i; μ = Agent.Param.μ_i
    end

    E .= (λ*μ) .* E
    E[s_ind,a_ind] = 1
end
function Func_update_ElTrace!(Agent::Str_Agent_Policy, AState::Str_Agent_State; 
                                extrinsic::Bool=true)
    a_ind  = AState.A_ind_t_old
    s_ind  = AState.S_ind_t_old
    if extrinsic
        E = Agent.E_e; λ = Agent.Param.λ_e; μ = Agent.Param.μ_e
    else
        E = Agent.E_i; λ = Agent.Param.λ_i; μ = Agent.Param.μ_i
    end

    E .= (λ*μ) .* E
    E[s_ind,a_ind] = 1
end
export Func_update_ElTrace!

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Update Q values for model-free
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_update_Q_MF!(Agent::Str_Agent; extrinsic::Bool=true)
    if extrinsic
        Q = Agent.Q_MFe; E = Agent.E_e; δ = Agent.eRPE_t
    else
        Q = Agent.Q_MFi; E = Agent.E_i; δ = Agent.iRPE_t
    end

    ρ = Agent.Param.ρ
    Q .= Q .+ ((ρ*δ) .* E)
end
function Func_update_Q_MF!(Agent::Str_Agent_Policy, AState::Str_Agent_State;
                            extrinsic::Bool=true)
    if extrinsic
        Q = Agent.Q_MFe; E = Agent.E_e; δ = Agent.eRPE_t
    else
        Q = Agent.Q_MFi; E = Agent.E_i; δ = Agent.iRPE_t
    end

    ρ = Agent.Param.ρ
    Q .= Q .+ ((ρ*δ) .* E)
end
export Func_update_Q_MF!

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Update from observing S_t to taking action A_t
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_update_fromS2A!(Agent::Str_Agent)
    Func_update_counts!(Agent)
    Func_update_world_model!(Agent)
    Func_update_reward_matrices!(Agent, Agent.Func_eR_sas, extrinsic=true)
    Func_update_reward_matrices!(Agent, Agent.Func_iR_sas, extrinsic=false)
    Func_update_Q_MB_PS!(Agent, extrinsic=true)
    Func_update_Q_MB_PS!(Agent, extrinsic=false)
    Func_compute_RPE!(Agent, extrinsic=true)
    Func_compute_RPE!(Agent, extrinsic=false)
    Func_update_ElTrace!(Agent, extrinsic=true)
    Func_update_ElTrace!(Agent, extrinsic=false)
    Func_update_Q_MF!(Agent, extrinsic=true)
    Func_update_Q_MF!(Agent, extrinsic=false)
    Func_compute_π_act!(Agent)
end
function Func_update_fromS2A!(Agent::Str_Agent_Policy, AState::Str_Agent_State)
    Func_update_counts!(Agent,AState)
    Func_update_world_model!(Agent,AState)
    Func_update_reward_matrices!(Agent, Agent.Func_eR_sas, AState, extrinsic=true)
    Func_update_reward_matrices!(Agent, Agent.Func_iR_sas, AState, extrinsic=false)
    Func_update_Q_MB_PS!(Agent, AState, extrinsic=true)
    Func_update_Q_MB_PS!(Agent, AState, extrinsic=false)
    Func_compute_RPE!(Agent, AState, extrinsic=true)
    Func_compute_RPE!(Agent, AState, extrinsic=false)
    Func_update_ElTrace!(Agent, AState, extrinsic=true)
    Func_update_ElTrace!(Agent, AState, extrinsic=false)
    Func_update_Q_MF!(Agent, AState, extrinsic=true)
    Func_update_Q_MF!(Agent, AState, extrinsic=false)
    Func_compute_π_act!(Agent, AState)
end
export Func_update_fromS2A!
