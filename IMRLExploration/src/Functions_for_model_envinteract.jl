# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# For likelihood
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_train_to_SandA!(A::Str_Agent,S_Seq,A_Seq; onlyLogl = false)
  Tsteps = length(S_Seq)
  Func_first_obs!(A,s=S_Seq[1])
  Logl = Func_load_action!(A;a=A_Seq[1])
  if !onlyLogl
    Agents = Array{Str_Agent,1}()
    push!(Agents,deepcopy(A))
  end

  for t = 2:(Tsteps)
    sp = S_Seq[t]
    Func_observe_state!(A;sp=sp)
    Func_update_fromS2A!(A)
    if t<Tsteps
      Logl = Logl + Func_load_action!(A;a=A_Seq[t])
    end
    if !onlyLogl
      push!(Agents,deepcopy(A))
    end
  end

  if onlyLogl
    return Logl
  else
    return Logl, Agents
  end
end
export Func_train_to_SandA!


function Func_train_to_SandA!(A::Str_Agent,
                  S_Seq, A_Seq, Sind_Seq, Aind_Seq,
                  Snum_Seq, Snew_Seq, SSet_Seq; onlyLogl = false)
  Tsteps = length(S_Seq)
  Func_first_obs!(A,S_Seq[1],Sind_Seq[1], SSet_Seq[1], Snew_Seq[1], Snum_Seq[1])

  Logl = Func_load_action!(A,A_Seq[1],Aind_Seq[1])
  if !onlyLogl
    Agents = Array{Str_Agent,1}()
    push!(Agents,deepcopy(A))
  end

  for t = 2:(Tsteps)
    Func_observe_state!(A,  S_Seq[t],Sind_Seq[t], SSet_Seq[t], 
                            Snew_Seq[t], Snum_Seq[t])
    Func_update_fromS2A!(A)
    if t<Tsteps
      Logl = Logl + Func_load_action!(A,A_Seq[t],Aind_Seq[t])
    end
    if !onlyLogl
      push!(Agents,deepcopy(A))
    end
  end

  if onlyLogl
    return Logl
  else
    return Logl, Agents
  end
end

function Func_train_to_SandA!(A::Str_Agent_Policy, ASates; onlyLogl = false)
  Tsteps = length(ASates)
  Func_first_obs!(A,ASates[1])

  Logl = Func_load_action!(A,ASates[1])
  if !onlyLogl
    Agents = Array{Str_Agent_Policy,1}()
    push!(Agents,deepcopy(A))
  end

  for t = 2:(Tsteps)
    Func_observe_state!(A, ASates[t])
    Func_update_fromS2A!(A, ASates[t])
    if t<Tsteps
      Logl = Logl + Func_load_action!(A, ASates[t])
    end
    if !onlyLogl
      push!(Agents,deepcopy(A))
    end
  end

  if onlyLogl
    return Logl
  else
    return Logl, Agents
  end
end
export Func_train_to_SandA!

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# For simulation
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_simulate_SandA!(A, Environment;
                            s0 = [0,0], t_max = 3000, rng=Random.GLOBAL_RNG,
                            G_type_default = -1)
    Func_first_obs!(A,s=s0)
    Func_sample_action!(A;rng=rng)

    t = 1
    S_Seq = [copy(A.S_t)]
    A_Seq = [copy(A.A_t)]
    while (A.eR_t<=0)&&(t<t_max)
        sp = Func_env_sampler!(Environment, A.S_t, A.A_t; rng=rng)
        Func_observe_state!(A;sp=sp)
        Func_update_fromS2A!(A)
        Func_sample_action!(A;rng=rng)
        push!(S_Seq,copy(A.S_t))
        push!(A_Seq,copy(A.A_t))
        t = t + 1
    end
    if S_Seq[end] ∈ Goal_states_inf_env
      G_type = S_Seq[end][2]
    else
      G_type = G_type_default
    end
    return (;S_Seq = S_Seq, A_Seq = A_Seq, G_type = G_type)
end
function Func_simulate_SandA!(A::Str_Agent_Policy, AState0::Str_Agent_State,
                Environment; t_max = 3000, rng=Random.GLOBAL_RNG, 
                G_type_default = -1)
    Func_first_obs!(A, AState0)
    A_ind_t, A_t = Func_sample_action(A, AState0; rng=rng)

    t = 1
    AStates = [Str_Agent_State(AState0, A_ind_t, A_t)]
    while (A.eR_t<=0)&&(t<t_max)
        AStatet = Func_env_sampler!(Environment, deepcopy(AStates[end]); rng=rng)
        Func_observe_state!(A, AStatet)
        Func_update_fromS2A!(A, AStatet)
        if A.eR_t<=0
          A_ind_t, A_t = Func_sample_action(A, AStatet; rng=rng)
        else
          A_ind_t = -1; A_t = -1;
        end
        push!(AStates, Str_Agent_State(AStatet, A_ind_t, A_t))
        t = t + 1
    end
    if AStates[end].S_t ∈ Goal_states_inf_env
      G_type = AStates[end].S_t[2]
    else
      G_type = G_type_default
    end
    return (;AStates = AStates, G_type = G_type)
end
export Func_simulate_SandA!