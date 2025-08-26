# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Inf environment structure
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
@concrete struct Str_InfEnv
    G_type
    TransMatrix
    Stochastic
    NotRepeatedState
    S4_Manipulator
    N_stoch
end
function  Str_InfEnv(; G_type = 0, TransMatrix = Read_transM(), 
                        S4_Manipulator = true, Stochastic = true,
                        NotRepeatedState = true, N_stoch = 50)
    if S4_Manipulator
        s_ind = 4; TransMatrix[s_ind+1,:] .= -1
    end
    Str_InfEnv(Ref(G_type), TransMatrix, Stochastic, NotRepeatedState, 
                S4_Manipulator, Ref(N_stoch))
end
export Str_InfEnv

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Image to state convertor
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_Image2State(s)
    if s[1] == 2
        return 0
    elseif s[1] == 0
        return s[2]+1
    elseif s[1] == 1
        return  7
    else
        @show s
        error("The state does not exist.")
    end
end
export Func_Image2State


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Sampler
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_env_sampler!(Environment::Str_InfEnv, s, a; 
                            rng=Random.GLOBAL_RNG)
    TransMatrix = Environment.TransMatrix
    G = Environment.G_type[]
    N_stoch = Environment.N_stoch[]

    if s[1] == 2        # if s is a goal state
        # reset to initial state [0,0]
        return [0,0]    
    elseif s[1] == 0    # if s is a progrissing state 
        s_ind = s[2]+1
        # read the next state from transition matrix
        sp = TransMatrix[s_ind+1,a+1]
        # if it's equal to -1, it means it's the first time visiting steat 4
        if sp == -1
            # manipulate state 4, so that the 1st visit is to the stochastic part
            TransMatrix[s_ind+1, (1 .+ [mod(a,3), mod(a+1,3),mod(a+2,3)])] .=
                [7,5,8]
            sp = 7
        end
    elseif s[1] == 1    # if s is a stochastic state
        s_ind = 7
        # read the next state from transition matrix
        sp = TransMatrix[s_ind+1,a+1]
    else
        @show s
        error("The state does not exist.")
    end

    if sp == 0          # if sp is a goal state
        return [2,G]
    elseif sp == 7      # if sp is a stochastic state
        if Environment.Stochastic     # if we *have* it stochastic
            if Environment.NotRepeatedState # if sp ≠ s
                spp = rand(rng,0:(N_stoch-2))
                if spp >= s[2]
                    spp = spp + 1
                end
            else                         # if no constraint
                spp = rand(rng,0:(N_stoch-1))
            end
            return [1,spp]
        else                         
            return [1,0]    # if we have only 1 stochastic state
        end
    else        # if sp is progressing
        return [0,sp-1]
    end
end


function Func_env_sampler!(Environment::Str_InfEnv, AState0::Str_Agent_State; 
                            rng=Random.GLOBAL_RNG)
    sp = Func_env_sampler!(Environment, AState0.S_t, AState0.A_t; rng=rng)
    if sp ∈ AState0.State_Set     
        new_state = false
        state_num = AState0.state_num

        State_Set = AState0.State_Set
        Act_Set = AState0.Act_Set

        S_t = sp
        A_t = Func_unknown_values(Act_Set)
        S_t_old = AState0.S_t
        A_t_old = AState0.A_t
        S_ind_t = findfirst(isequal(sp),State_Set)
        A_ind_t = -1
        S_ind_t_old = AState0.S_ind_t
        A_ind_t_old = AState0.A_ind_t

        act_num = AState0.act_num

        Epi = AState0.Epi
        t_epi = AState0.t_epi + 1
    else
        new_state = true
        state_num = AState0.state_num + 1

        State_Set = AState0.State_Set; State_Set[state_num] .= sp
        Act_Set = AState0.Act_Set

        S_t = sp
        A_t = Func_unknown_values(Act_Set)
        S_t_old = AState0.S_t
        A_t_old = AState0.A_t
        S_ind_t = state_num
        A_ind_t = -1
        S_ind_t_old = AState0.S_ind_t
        A_ind_t_old = AState0.A_ind_t

        act_num = AState0.act_num

        Epi = AState0.Epi
        t_epi = AState0.t_epi + 1
    end
    return deepcopy(Str_Agent_State(State_Set, Act_Set, S_t, A_t, 
                S_t_old, A_t_old, S_ind_t, A_ind_t, S_ind_t_old, A_ind_t_old,
                state_num, act_num, Epi, t_epi, new_state))
end


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Reward function
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
@concrete struct InfEnv_Rewards
    Rs#::Vector{Float64}
    Goal_states#::Vector{Vector{Int64}}
end
function InfEnv_Rewards(Rs; Goal_states = Goal_states_inf_env)
    if length(Rs) != length(Goal_states)
        error("we need to have as many reward values as goal states!")
    end
    InfEnv_Rewards(Rs, Goal_states)
end
export InfEnv_Rewards

function Func_fix_TM!(Data; Epi = 1, s = 4, sps = [7,5,8])
    if sum(Data.states[Epi] .== s) > 0
        a = Data.actions[Epi][Data.states[Epi] .== s][1]
        Data.TM[s+1, (1 .+ [mod(a,3), mod(a+1,3),mod(a+2,3)])] .= sps
    end
end
export Func_fix_TM!
