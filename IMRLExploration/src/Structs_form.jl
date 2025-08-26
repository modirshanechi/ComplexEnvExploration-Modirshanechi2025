# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Structure for Input (Data)
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
@concrete struct Str_Input
    Sub
    Gender             # 0: Male, 1: Female

    states
    actions
    images             # P:0, N:1, G:2

    trial_time
    resp_time

    TM
end
function Str_Input(; Sub=1,Gender=0,states=[zeros(1)],actions=[zeros(1)],
                     images=[zeros(2,1)],trial_time=[zeros(1)],
                     resp_time=[zeros(1)],TM=zeros(2,2))
    Str_Input(Sub,Gender,states,actions,images,trial_time,resp_time,TM)
end
export Str_Input

@concrete struct Str_Input_Seqs
    S_Seqs
    A_Seqs
end
function Str_Input_Seqs(Data::Str_Input)
    Str_Input_Seqs(Func_Image_Arr2Vec.(Data.images),Data.actions)
end
export Str_Input_Seqs

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Extracting state-action information for modeling
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# putting data of each episode in one tuple
function Str_Input2Tuple(Data::Str_Input)
    S_Seqs = Func_Image_Arr2Vec.(Data.images)
    A_Seqs = Data.actions
    [(  S_Seq  = Array{Array{Int64,1},1}(S_Seqs[i]), 
        A_Seq  = Array{Int64,1}(A_Seqs[i]),
        G_type = Func_GoalType(Data)) for i = eachindex(S_Seqs)]
end
export Str_Input2Tuple

# processing data to detailed vectors:
#   S_Seq     : Images
#   A_Seq     : Actions
#   Sind_Seq  : The index of the image within the images observed so far
#   Aind_Seq  : The index of the actions
#   Snum_Seq  : Number of known images so far
#   Snew_Seq  : Whether they observed a new image or now
#   SSet_Seq  : The vectors of observed images
function Str_Input2SASeq(Data::Str_Input; Act_Set = Array(0:2),
                            n_state = 61, Goal_states = Goal_states_inf_env)
    # default state set 
    SSet00 = deepcopy(Goal_states); Snum00 = Int64(length(Goal_states))
    for i = (length(Goal_states)+1):n_state
        push!(SSet00,Func_unknown_values(Goal_states))
    end    

    # sequence of state and actions
    S_Seqs = Func_Image_Arr2Vec.(Data.images)
    A_Seqs = Data.actions
    epi_num = length(S_Seqs)

    # sequence of state-set information
    Snum_Seqs = [zeros(Int64, length(S_Seqs[epi])) for epi = 1:epi_num]
    Snew_Seqs = [zeros(length(S_Seqs[epi])) .== 0 for epi = 1:epi_num]
    SSet_Seqs = [Vector{typeof(SSet00)}(undef, length(S_Seqs[epi])) for epi = 1:epi_num]

    # sequence of state and action indeces
    Sind_Seqs = [zeros(Int64, length(S_Seqs[epi])) for epi = 1:epi_num]
    Aind_Seqs = [zeros(Int64, length(S_Seqs[epi])) for epi = 1:epi_num]

    for epi = 1:epi_num
        #S_Seq = S_Seqs[epi]; A_Seq = A_Seqs[epi]
        if epi == 1
                SSet0 = deepcopy(SSet00)
                Snum0 = Snum00
        else
                SSet0 = deepcopy(SSet_Seqs[epi - 1][end])
                Snum0 = Snum_Seqs[epi - 1][end]
        end

        # first observation
        s = S_Seqs[epi][1]; a = A_Seqs[epi][1]
        if s ∈ SSet0
                Snum_Seqs[epi][1] = Snum0
                Snew_Seqs[epi][1] = false
                SSet_Seqs[epi][1] = deepcopy(SSet0)
        else
                Snum_Seqs[epi][1] = Snum0 + 1
                Snew_Seqs[epi][1] = true
                SSet_Seqs[epi][1] = deepcopy(SSet0); 
                SSet_Seqs[epi][1][Snum_Seqs[epi][1]] .= s
        end
        Sind_Seqs[epi][1] = findfirst(isequal(s),SSet_Seqs[epi][1])
        Aind_Seqs[epi][1] = findfirst(isequal(a),Act_Set)

        # next observations
        for i = 2:length(S_Seqs[epi])
                s = S_Seqs[epi][i]; a = A_Seqs[epi][i]
                if s ∈ SSet_Seqs[epi][i-1]
                    Snum_Seqs[epi][i] = Snum_Seqs[epi][i-1]
                    Snew_Seqs[epi][i] = false
                    SSet_Seqs[epi][i] = deepcopy(SSet_Seqs[epi][i-1])
                else
                    Snum_Seqs[epi][i] = Snum_Seqs[epi][i-1] + 1
                    Snew_Seqs[epi][i] = true
                    SSet_Seqs[epi][i] = deepcopy(SSet_Seqs[epi][i-1]); 
                    SSet_Seqs[epi][i][Snum_Seqs[epi][i]] .= s
                end
                Sind_Seqs[epi][i] = findfirst(isequal(s),SSet_Seqs[epi][i])
                if i != length(S_Seqs[epi])
                    Aind_Seqs[epi][i] = findfirst(isequal(a),Act_Set)
                else
                    Aind_Seqs[epi][i] = -1
                end
        end
    end
    [(; S_Seq     = Vector{Vector{Int64}}(S_Seqs[i]), 
        A_Seq     = Vector{Int64}(A_Seqs[i]),
        Sind_Seq  = Vector{Int64}(Sind_Seqs[i]),
        Aind_Seq  = Vector{Int64}(Aind_Seqs[i]),
        Snum_Seq  = Vector{Int64}(Snum_Seqs[i]),
        Snew_Seq  = Vector{Bool}(Snew_Seqs[i]),
        SSet_Seq  = Vector{Vector{Vector{Int64}}}(SSet_Seqs[i]),
        G_type = Func_GoalType(Data)) for i = eachindex(S_Seqs)]
end
export Str_Input2SASeq

# for transforming simulated data to input structure
function Str_SASeq2Input(SData; Sub = -1, Gender = -1, TM = Read_transM())
    images  = [[SData[epi].AStates[t].S_t for t = eachindex(SData[epi].AStates)] 
                                          for epi = eachindex(SData)]
    actions = [[SData[epi].AStates[t].A_t for t = eachindex(SData[epi].AStates)] 
                                          for epi = eachindex(SData)]
    states  = [Func_Image2State.(images[epi]) for epi = eachindex(images)]                                    
    trial_time = [zero(states[epi]) for epi = eachindex(states)]
    resp_time  = [zero(states[epi]) for epi = eachindex(states)]
    images = [hcat(images[epi]...) for epi = eachindex(states)]
    Str_Input(Sub,Gender,states,actions,images,trial_time,resp_time,TM)
end
function Str_SASeq2Input(SData::NamedTuple; Sub = -1, Gender = -1)
    Str_SASeq2Input(SData.data; Sub=Sub, Gender=Gender, TM = SData.TM)
end
export Str_SASeq2Input

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Structure for Parameters
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
@concrete mutable struct Str_Param
    # Model-building parameters
    κ#::Float64                      # 1 - Count-decay
    ϵ_new#::Float64                  # 2 - World-model parameter for new obs.
    ϵ_obs#::Float64                  # 3 - World-model parameter for obseverd obs.
    # discount factors
    λ_e#::Float64                    # 4 - extrinsic discount factor
    λ_i#::Float64                    # 5 - intrinsic discount factor
    # MB prioritized sweeping
    T_PS_e#::Int                     # 6 - Number of cycle of PS for extrinsic reward
    T_PS_i#::Int                     # 7 - Number of cycle of PS for intrinsic reward
    # MF TD-learner
    ρ#::Float64                      # 8 - MF learning rate
    μ_e#::Float64                    # 9 - decay rate for extrinsic eligibility trace
    μ_i#::Float64                    # 10 - decay rate for intrinsic eligibility trace
    Q_e0#::Float64                   # 11
    Q_i0#::Float64                   # 12
    # Policy parameters
    β_MBe_1#::Float64                # 13 - MB weight (extrinsic) in Epi = 1
    β_MBe_2#::Float64                # 14 - MB weight (extrinsic) in Epi > 1
    β_MBi_1#::Float64                # 15 - MB weight (intrinsic) in Epi = 1
    β_MBi_2#::Float64                # 16 - MB weight (intrinsic) in Epi > 1
    β_MFe_1#::Float64                # 17 - MF weight (extrinsic) in Epi = 1
    β_MFe_2#::Float64                # 18 - MF weight (extrinsic) in Epi > 1
    β_MFi_1#::Float64                # 19 - MF weight (intrinsic) in Epi = 1
    β_MFi_2#::Float64                # 20 - MF weight (intrinsic) in Epi > 1
    Q_bias#::Vector{Float64}         # 21:22 - action biases
    # Whether to leak total counts for novelty
    total_leak#::Bool
    # Whether to leak background counts for all state-action pairs
    back_leak#::Bool
end
function Str_Param(; κ=1., ϵ_new=1., ϵ_obs=1., λ_e=0.9, λ_i=0.9,
                    T_PS_e=η0_global[6], T_PS_i=η0_global[7], ρ=0.1,
                    μ_e = 0.9, μ_i = 0.9, Q_e0=0., Q_i0=0.,
                    β_MBe_1=0., β_MBe_2=1., β_MBi_1=η0_global[15], β_MBi_2=1.,
                    β_MFe_1=0., β_MFe_2=1., β_MFi_1=1., β_MFi_2=1.,
                    Q_bias = [0.,0.], total_leak = true, back_leak = false)
    Str_Param((κ), (ϵ_new), (ϵ_obs), (λ_e), (λ_i), 
              (T_PS_e), (T_PS_i), (ρ),
              (μ_e), (μ_i), (Q_e0), (Q_i0),
              (β_MBe_1), (β_MBe_2), (β_MBi_1), (β_MBi_2),
              (β_MFe_1), (β_MFe_2), (β_MFi_1), (β_MFi_2),
              (Q_bias),
              total_leak, back_leak)
end
function Str_Param(η::Array{Float64,1}; total_leak = true, back_leak = false)
    if length(η) == 22
        Str_Param((η[1]),  (η[2]),  (η[3]),  (η[4]),  (η[5]),
                  (Int(round(η[6]))),  (Int(round(η[7]))),
                  (η[8]),  (η[9]),  (η[10]), (η[11]), (η[12]),
                  (η[13]), (η[14]), (η[15]), (η[16]),
                  (η[17]), (η[18]), (η[19]), (η[20]), 
                  (η[21:22]), total_leak, back_leak)
    else
        error("Wrong number of parameters")
    end
end
export Str_Param

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Structure for Agent
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
@concrete mutable struct Str_Agent#{Stype,Atype,F1,F2}
    Param#::Str_Param                    # Learning parameters

    # Reward functions
    Func_eR_sas#::F1                     # WARNING: Reward functions should get (Agent, s, a, s';...) where s and s' should also take the vale Symbol("New")
    Func_iR_sas#::F2

    # Sets
    State_Set#::Array{Stype,1}           # The set of all visited states
    Act_Set#::Array{Atype,1}             # The set of all possible actions

    C_s#::Array{Float64,1}               # Counts for states
    C_sa#::Array{Float64,2}              # Counts for state-actions
    C_sas#::Array{Float64,3}             # Counts for state-action-states

    θ_sas#::Array{Float64,3}             # Transition probabilities

    eR_sas#::Array{Float64,3}            # The matrix of extrinsic rewards
    iR_sas#::Array{Float64,3}            # The matrix of intrinsic rewards

    Q_MBe#::Array{Float64,2}             # MB Q-values for extrinsic reward
    Q_MBi#::Array{Float64,2}             # MB Q-values for intrinsic reward
    U_e#::Array{Float64,1}               # U-values for PS for extrinsic reward
    U_i#::Array{Float64,1}               # U-values for PS for intrinsic reward

    V_dummy#::Array{Float64,1}           # dummy values for V in PS
    P_dummy#::Array{Float64,1}           # dummy values for Priority in PS

    Q_MFe#::Array{Float64,2}             # MF Q-values for extrinsic reward
    Q_MFi#::Array{Float64,2}             # MF Q-values for intrinsic reward
    E_e#::Array{Float64,2}               # Eligibility trace for extrinsic reward
    E_i#::Array{Float64,2}               # Eligibility trace for intrinsic reward

    # Variables
    S_t#::Stype                          # Current state
    A_t#::Atype                          # Current action
    S_t_old#::Stype                      # Last state
    A_t_old#::Atype                      # Last action
    S_ind_t#::Int64                      # Current state index
    A_ind_t#::Int64                      # Current action index
    S_ind_t_old#::Int64                  # Last state index
    A_ind_t_old#::Int64                  # Last action index
    Q_MB_t#::Array{Float64,1}            # Current MB Qs
    Q_MF_t#::Array{Float64,1}            # Current MF Qs
    Q_t#::Array{Float64,1}               # Current Qs
    π_A_t#::Array{Float64,1}             # Current action probabilities

    eR_t#::Float64                       # Current extrinsic reward
    eRPE_t#::Float64                     # Current extrinsic RPE
    iR_t#::Float64                       # Current intrinsic reward
    iRPE_t#::Float64                     # Current intrinsic RPE

    eR_max_t#::Float64                   # Maximum extrinsic reward found so far

    state_num#::Int64                    # Number of states
    act_num#::Int64                      # Number of actions

    Epi#::Int64                          # Episode number
    t_total#::Int64                      # Time since the beginning of the task
    t_epi#::Int64                        # Time since the beginning of the episode (t_epi <= t_total)

    new_state#::Bool                     # Indicator of whether S_t is a new state
end
function Str_Agent(Param::Str_Param, Rs, Func_iR_sas; 
                   Goal_states = Goal_states_inf_env, Act_Set = Array(0:2),
                   eR_max_t = 0., n_state = 61)
    Param = deepcopy(Param)
    Func_eR_sas = InfEnv_Rewards(Rs; Goal_states = Goal_states)

    State_Set = deepcopy(Goal_states)
    for i = (length(Goal_states)+1):n_state
        push!(State_Set,Func_unknown_values(Goal_states))
    end

    Act_Set = Act_Set
    
    C_s = zeros(length(State_Set))
    C_sa = zeros(length(State_Set),length(Act_Set))
    C_sas = zeros(length(State_Set),length(Act_Set),length(State_Set));
    
    θ_sas = zeros(length(State_Set),length(Act_Set),length(State_Set)+1);       # +1 is for the possiblity of moving to a new state
    
    eR_sas = zeros(length(State_Set)+1,length(Act_Set),length(State_Set)+1);    # +1 is for the potential new states
    iR_sas = zeros(length(State_Set)+1,length(Act_Set),length(State_Set)+1);    # +1 is for the potential new states
    
    Q_MBe = zeros(length(State_Set),length(Act_Set));
    Q_MBi = zeros(length(State_Set),length(Act_Set));
    U_e = zeros(length(State_Set));
    U_i = zeros(length(State_Set));

    V_dummy = zeros(length(State_Set));
    P_dummy = zeros(length(State_Set));
    
    Q_MFe = ones(length(State_Set),length(Act_Set)) .* Param.Q_e0;
    Q_MFi = ones(length(State_Set),length(Act_Set)) .* Param.Q_i0;
    E_e = zeros(length(State_Set),length(Act_Set));
    E_i = zeros(length(State_Set),length(Act_Set));
    
    # Variables
    S_t = Func_unknown_values(State_Set)
    A_t = (Func_unknown_values(Act_Set))         # Ref: Act assumed to be int
    S_t_old = Func_unknown_values(State_Set)
    A_t_old = (Func_unknown_values(Act_Set))      # Ref: Act assumed to be int
    S_ind_t = -1
    A_ind_t = -1
    S_ind_t_old = -1
    A_ind_t_old = -1
    Q_MB_t = ones(length(Act_Set)) .* (-1.)
    Q_MF_t = ones(length(Act_Set)) .* (-1.)
    Q_t    = ones(length(Act_Set)) .* (-1.)
    π_A_t  = ones(length(Act_Set)) .* (-1.)
    
    eR_t = (-1.)
    eRPE_t = (-1.)
    iR_t = (-1.)
    iRPE_t = (-1.)
    
    eR_max_t = (eR_max_t)
    
    state_num = (Func_count_known_values(State_Set))
    act_num   = (length(Act_Set))
    
    Epi = (1)
    t_total = (0)
    t_epi = (0)
    
    new_state = (true)

    Str_Agent(Param, Func_eR_sas, Func_iR_sas, State_Set, Act_Set,
              C_s, C_sa, C_sas, θ_sas, eR_sas, iR_sas,
              Q_MBe, Q_MBi, U_e, U_i, V_dummy , P_dummy, Q_MFe, Q_MFi, E_e, E_i,
              S_t, A_t, S_t_old, A_t_old, 
              S_ind_t, A_ind_t, S_ind_t_old, A_ind_t_old, 
              Q_MB_t, Q_MF_t, Q_t, π_A_t, eR_t, eRPE_t, iR_t, iRPE_t,
              eR_max_t, state_num, act_num, Epi, t_total, t_epi, new_state)
end
export Str_Agent


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Structure for the agent's current state (pre-computed and non-mutable)
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
@concrete struct Str_Agent_State
    # Sets
    State_Set#::Array{Stype,1}           # The set of all visited states
    Act_Set#::Array{Atype,1}             # The set of all possible actions

    # Variables
    S_t#::Stype                          # Current state
    A_t#::Atype                          # Current action
    S_t_old#::Stype                      # Last state
    A_t_old#::Atype                      # Last action
    S_ind_t#::Int64                      # Current state index
    A_ind_t#::Int64                      # Current action index
    S_ind_t_old#::Int64                  # Last state index
    A_ind_t_old#::Int64                  # Last action index

    state_num#::Int64                    # Number of states
    act_num#::Int64                      # Number of actions

    Epi#::Int64                          # Episode number
    t_epi#::Int64                        # Time since the beginning of the episode (t_epi <= t_total)

    new_state#::Bool                     # Indicator of whether S_t is a new state
end
export Str_Agent_State
function Str_Agent_State(A::Str_Agent_State, A_ind_t::Int, A_t::Int)            # to set the new action
    Str_Agent_State(A.State_Set, A.Act_Set, A.S_t, A_t, A.S_t_old, A.A_t_old,
                    A.S_ind_t, A_ind_t, A.S_ind_t_old, A.A_ind_t_old,
                    A.state_num, A.act_num, A.Epi, A.t_epi, A.new_state)
end
function Str_Agent_State(; Act_Set = Array(0:2),
                        Goal_states = Goal_states_inf_env, n_state = 61)
    State_Set = deepcopy(Goal_states)
    for i = (length(Goal_states)+1):n_state
        push!(State_Set,Func_unknown_values(Goal_states))
    end

    Act_Set = Act_Set
    
    # Variables
    S_t = Func_unknown_values(State_Set)
    A_t = (Func_unknown_values(Act_Set))         # Ref: Act assumed to be int
    S_t_old = Func_unknown_values(State_Set)
    A_t_old = (Func_unknown_values(Act_Set))      # Ref: Act assumed to be int
    S_ind_t = -1
    A_ind_t = -1
    S_ind_t_old = -1
    A_ind_t_old = -1
    
    state_num = length(Goal_states)
    act_num   = length(Act_Set)
    
    Epi = 1
    t_epi = 0
    
    new_state = true

    Str_Agent_State(State_Set, Act_Set, S_t, A_t, S_t_old, A_t_old,
                    S_ind_t, A_ind_t, S_ind_t_old, A_ind_t_old,
                    state_num, act_num, Epi, t_epi, new_state)
end
function Str_Agent_State(A::Str_Agent_State, S_t::Vector{Int}; if_init = false) # to initialize the agent and set the current state
    State_Set = A.State_Set
    Act_Set = A.Act_Set
    act_num = A.act_num
    S_t = S_t
    if S_t ∈ State_Set
        state_num = A.state_num
        S_ind_t::Int = findfirst(isequal(S_t),State_Set)
        new_state = false
    else
        state_num = A.state_num + 1
        State_Set[state_num] .= S_t
        S_ind_t = state_num
        new_state = true
    end
    A_t     = Func_unknown_values(Act_Set)
    S_t_old = Func_unknown_values(State_Set)
    A_t_old = Func_unknown_values(Act_Set)
    A_ind_t = -1
    S_ind_t_old = -1
    A_ind_t_old = -1
    
    if if_init
        Epi = A.Epi
    else
        Epi = A.Epi + 1
    end
    t_epi = 0

    Str_Agent_State(State_Set, Act_Set, S_t, A_t, S_t_old, A_t_old,
                            S_ind_t, A_ind_t, S_ind_t_old, A_ind_t_old,
                            state_num, act_num, Epi, t_epi, new_state)
end

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Structure for the agent's current policy variables (mutable)
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

@concrete mutable struct Str_Agent_Policy#{Stype,Atype,F1,F2}
    Param#::Str_Param                    # Learning parameters

    # Reward functions
    Func_eR_sas#::F1                     # WARNING: Reward functions should get (Agent, s, a, s';...) where s and s' should also take the vale Symbol("New")
    Func_iR_sas#::F2

    C_s#::Array{Float64,1}               # Counts for states
    C_sa#::Array{Float64,2}              # Counts for state-actions
    C_sas#::Array{Float64,3}             # Counts for state-action-states

    θ_sas#::Array{Float64,3}             # Transition probabilities

    eR_sas#::Array{Float64,3}            # The matrix of extrinsic rewards
    iR_sas#::Array{Float64,3}            # The matrix of intrinsic rewards

    Q_MBe#::Array{Float64,2}             # MB Q-values for extrinsic reward
    Q_MBi#::Array{Float64,2}             # MB Q-values for intrinsic reward
    U_e#::Array{Float64,1}               # U-values for PS for extrinsic reward
    U_i#::Array{Float64,1}               # U-values for PS for intrinsic reward

    V_dummy#::Array{Float64,1}           # dummy values for V in PS
    P_dummy#::Array{Float64,1}           # dummy values for Priority in PS

    Q_MFe#::Array{Float64,2}             # MF Q-values for extrinsic reward
    Q_MFi#::Array{Float64,2}             # MF Q-values for intrinsic reward
    E_e#::Array{Float64,2}               # Eligibility trace for extrinsic reward
    E_i#::Array{Float64,2}               # Eligibility trace for intrinsic reward

    # Variables
    Q_MB_t#::Array{Float64,1}            # Current MB Qs
    Q_MF_t#::Array{Float64,1}            # Current MF Qs
    Q_t#::Array{Float64,1}               # Current Qs
    π_A_t#::Array{Float64,1}             # Current action probabilities

    eR_t#::Float64                       # Current extrinsic reward
    eRPE_t#::Float64                     # Current extrinsic RPE
    iR_t#::Float64                       # Current intrinsic reward
    iRPE_t#::Float64                     # Current intrinsic RPE

    eR_max_t#::Float64                   # Maximum extrinsic reward found so far
end
function Str_Agent_Policy(Param::Str_Param, Rs, Func_iR_sas; 
                   Goal_states = Goal_states_inf_env, Act_Set = Array(0:2),
                   eR_max_t = 0., n_state = 61)
    n_act = length(Act_Set)

    Param = deepcopy(Param)
    Func_eR_sas = InfEnv_Rewards(Rs; Goal_states = Goal_states)
    
    C_s = zeros(n_state)
    C_sa = zeros(n_state,n_act)
    C_sas = zeros(n_state,n_act,n_state);
    
    θ_sas = zeros(n_state,n_act,n_state+1);       # +1 is for the possiblity of moving to a new state
    
    eR_sas = zeros(n_state+1,n_act,n_state+1);    # +1 is for the potential new states
    iR_sas = zeros(n_state+1,n_act,n_state+1);    # +1 is for the potential new states
    
    Q_MBe = zeros(n_state,n_act);
    Q_MBi = zeros(n_state,n_act);
    U_e = zeros(n_state);
    U_i = zeros(n_state);

    V_dummy = zeros(n_state);
    P_dummy = zeros(n_state);
    
    Q_MFe = ones(n_state,n_act) .* Param.Q_e0;
    Q_MFi = ones(n_state,n_act) .* Param.Q_i0;
    E_e = zeros(n_state,n_act);
    E_i = zeros(n_state,n_act);
    
    # Variables
    Q_MB_t = ones(n_act) .* (-1.)
    Q_MF_t = ones(n_act) .* (-1.)
    Q_t    = ones(n_act) .* (-1.)
    π_A_t  = ones(n_act) .* (-1.)
    
    eR_t = (-1.)
    eRPE_t = (-1.)
    iR_t = (-1.)
    iRPE_t = (-1.)
    
    eR_max_t = (eR_max_t)

    Str_Agent_Policy(Param, Func_eR_sas, Func_iR_sas,
              C_s, C_sa, C_sas, θ_sas, eR_sas, iR_sas,
              Q_MBe, Q_MBi, U_e, U_i, V_dummy , P_dummy, Q_MFe, Q_MFi, E_e, E_i,
              Q_MB_t, Q_MF_t, Q_t, π_A_t, eR_t, eRPE_t, iR_t, iRPE_t, eR_max_t)
end
export Str_Agent_Policy


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Input data to Agent structs
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# transforming the input data to a sequency of state agents
function Str_Input2Agents(Data::Str_Input; Act_Set = Array(0:2),
                        n_state = 61, Goal_states = Goal_states_inf_env)
    DataProc = Str_Input2SASeq(Data; Act_Set = Act_Set,
                        n_state = n_state, Goal_states = Goal_states)
    DataAgent = [(; AStates = Vector{Str_Agent_State}(undef, length(DataProc[i].S_Seq)),
                    G_type = DataProc[i].G_type) 
                    for i = eachindex(DataProc)]
    for epi = eachindex(DataProc)
        for t = eachindex(DataProc[epi].S_Seq)
            State_Set = DataProc[epi].SSet_Seq[t]
            S_t = DataProc[epi].S_Seq[t]
            A_t = DataProc[epi].A_Seq[t]
            S_ind_t = DataProc[epi].Sind_Seq[t]
            A_ind_t = DataProc[epi].Aind_Seq[t]
            
            if t == 1
                S_t_old = Func_unknown_values(Goal_states)
                A_t_old = Func_unknown_values(Act_Set)
                S_ind_t_old = -1
                A_ind_t_old = -1
            else
                S_t_old = DataProc[epi].S_Seq[t-1]
                A_t_old = DataProc[epi].A_Seq[t-1]
                S_ind_t_old = DataProc[epi].Sind_Seq[t-1]
                A_ind_t_old = DataProc[epi].Aind_Seq[t-1]
            end

            state_num = DataProc[epi].Snum_Seq[t]
            act_num = length(Act_Set)

            new_state = DataProc[epi].Snew_Seq[t]
            DataAgent[epi].AStates[t] = Str_Agent_State(
                            State_Set, Act_Set, S_t, A_t, S_t_old, A_t_old,
                            S_ind_t, A_ind_t, S_ind_t_old, A_ind_t_old,
                            state_num, act_num, epi, t, new_state)
        end
    end
    return DataAgent
end
export Str_Input2Agents


