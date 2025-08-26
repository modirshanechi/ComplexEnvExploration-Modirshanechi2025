# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Agent access functions
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_agent_act(Agent::Str_Agent)
    Agent.A_t
end
export Func_agent_act

function Func_agent_state(Agent::Str_Agent)
    Agent.S_t
end
export Func_agent_state

function Func_agent_logL(Agent::Str_Agent)
    a_ind = findfirst(isequal(Agent.A_t), Agent.Act_Set)
    return log(Agent.π_A_t[a_ind])
end
export Func_agent_logL

function Func_agent_π_A(Agent::Str_Agent)
    return Agent.π_A_t
end
export Func_agent_π_A

function Func_agent_π_A_chosen(Agent::Str_Agent)
    a_ind = findfirst(isequal(Agent.A_t), Agent.Act_Set)
    return Agent.π_A_t[a_ind]
end
export Func_agent_π_A_chosen

function Func_agent_accuracy(Agent::Str_Agent)
    a_ind = findfirst(isequal(Agent.A_t), Agent.Act_Set)
    return (findmax(Agent.π_A_t)[1] == Agent.π_A_t[a_ind]) /
            sum(Agent.π_A_t .== Agent.π_A_t[a_ind])
end
export Func_agent_accuracy

function Func_agent_ent_π_A(Agent::Str_Agent)
    return -sum(Agent.π_A_t .* log.(Agent.π_A_t))
end
export Func_agent_ent_π_A

function Func_agent_Q_MB(Agent::Str_Agent; extrinsic::Bool=true)
    if extrinsic
        return Agent.Q_MBe
    else
        return Agent.Q_MBi
    end
end
export Func_agent_Q_MB

function Func_agent_Q_MF(Agent::Str_Agent; extrinsic::Bool=true)
    if extrinsic
        return Agent.Q_MFe
    else
        return Agent.Q_MFi
    end
end
export Func_agent_Q_MF

function Func_agent_Q_state(Agent::Str_Agent;
                            extrinsic::Bool=true, MB::Bool=true)
    s_ind = findfirst(isequal(Agent.S_t),Agent.State_Set)
    if extrinsic
        if MB
            return Agent.Q_MBe[s_ind,:]
        else
            return Agent.Q_MFe[s_ind,:]
        end
    else
        if MB
            return Agent.Q_MBi[s_ind,:]
        else
            return Agent.Q_MFi[s_ind,:]
        end
    end
end
export Func_agent_Q_state

function Func_agent_ΔQei_state(Agent::Str_Agent)
    if Agent.Epi == 1
        β_MBe = Agent.Param.β_MBe_1; β_MBi = Agent.Param.β_MBi_1
        β_MFe = Agent.Param.β_MFe_1; β_MFi = Agent.Param.β_MFi_1
    else
        β_MBe = Agent.Param.β_MBe_2; β_MBi = Agent.Param.β_MBi_2
        β_MFe = Agent.Param.β_MFe_2; β_MFi = Agent.Param.β_MFi_2
    end

    Qs_MBe = Func_agent_Q_state(Agent, extrinsic = true, MB = true)
    Qs_MFe = Func_agent_Q_state(Agent, extrinsic = true, MB = false)
    Qs_MBi = Func_agent_Q_state(Agent, extrinsic = false, MB = true)
    Qs_MFi = Func_agent_Q_state(Agent, extrinsic = false, MB = false)

    Qs_e = β_MBe .* Qs_MBe .+ β_MFe .* Qs_MFe
    Qs_i = β_MBi .* Qs_MBi .+ β_MFi .* Qs_MFi

    ΔQs_e = findmax(Qs_e)[1] - findmin(Qs_e)[1]
    ΔQs_i = findmax(Qs_i)[1] - findmin(Qs_i)[1]

    return [ΔQs_e, ΔQs_i]
end
export Func_agent_ΔQei_state


function Func_agent_ΔQMBMF_state(Agent::Str_Agent)
    if Agent.Epi == 1
        β_MBe = Agent.Param.β_MBe_1; β_MBi = Agent.Param.β_MBi_1
        β_MFe = Agent.Param.β_MFe_1; β_MFi = Agent.Param.β_MFi_1
    else
        β_MBe = Agent.Param.β_MBe_2; β_MBi = Agent.Param.β_MBi_2
        β_MFe = Agent.Param.β_MFe_2; β_MFi = Agent.Param.β_MFi_2
    end

    Qs_MBe = Func_agent_Q_state(Agent, extrinsic = true, MB = true)
    Qs_MFe = Func_agent_Q_state(Agent, extrinsic = true, MB = false)
    Qs_MBi = Func_agent_Q_state(Agent, extrinsic = false, MB = true)
    Qs_MFi = Func_agent_Q_state(Agent, extrinsic = false, MB = false)

    Qs_MB = β_MBe .* Qs_MBe .+ β_MBi .* Qs_MBi
    Qs_MF = β_MFe .* Qs_MFe .+ β_MFi .* Qs_MFi

    ΔQs_MB = findmax(Qs_MB)[1] - findmin(Qs_MB)[1]
    ΔQs_MF = findmax(Qs_MF)[1] - findmin(Qs_MF)[1]

    return [ΔQs_MB, ΔQs_MF]
end
export Func_agent_ΔQMBMF_state

function Func_agent_ΔQ_state(Agent::Str_Agent)
    if Agent.Epi == 1
        β_MBe = Agent.Param.β_MBe_1; β_MBi = Agent.Param.β_MBi_1
        β_MFe = Agent.Param.β_MFe_1; β_MFi = Agent.Param.β_MFi_1
    else
        β_MBe = Agent.Param.β_MBe_2; β_MBi = Agent.Param.β_MBi_2
        β_MFe = Agent.Param.β_MFe_2; β_MFi = Agent.Param.β_MFi_2
    end

    Qs_MBe = Func_agent_Q_state(Agent, extrinsic = true, MB = true)
    Qs_MFe = Func_agent_Q_state(Agent, extrinsic = true, MB = false)
    Qs_MBi = Func_agent_Q_state(Agent, extrinsic = false, MB = true)
    Qs_MFi = Func_agent_Q_state(Agent, extrinsic = false, MB = false)

    Qs_MB = β_MBe .* Qs_MBe .+ β_MBi .* Qs_MBi
    Qs_MF = β_MFe .* Qs_MFe .+ β_MFi .* Qs_MFi
    Qs = Qs_MB .+ Qs_MF

    ΔQ = findmax(Qs)[1] - findmin(Qs)[1]
    return ΔQ
end
export Func_agent_ΔQ_state



function Func_agent_π_A_all(Agent::Str_Agent)
    if Agent.Epi == 1
        β_MBe = Agent.Param.β_MBe_1
        β_MBi = Agent.Param.β_MBi_1
        β_MFe = Agent.Param.β_MFe_1
        β_MFi = Agent.Param.β_MFi_1
    else
        β_MBe = Agent.Param.β_MBe_2
        β_MBi = Agent.Param.β_MBi_2
        β_MFe = Agent.Param.β_MFe_2
        β_MFi = Agent.Param.β_MFi_2
    end

    Q_MB = β_MBe .* Agent.Q_MBe .+ β_MBi .* Agent.Q_MBi
    Q_MF = β_MFe .* Agent.Q_MFe .+ β_MFi .* Agent.Q_MFi
    Q = Q_MB .+ Q_MF

    π_A = zeros(size(Q))

    for s = 1:Agent.state_num
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
export Func_agent_π_A_all

function Func_agent_eR_t(Agent::Str_Agent)
    Agent.eR_t
end
export Func_agent_eR_t

function Func_agent_eRPE_t(Agent::Str_Agent)
    Agent.eRPE_t
end
export Func_agent_eRPE_t

function Func_agent_iR_t(Agent::Str_Agent)
    Agent.iR_t
end
export Func_agent_iR_t

function Func_agent_iRPE_t(Agent::Str_Agent)
    Agent.iRPE_t
end
export Func_agent_iRPE_t

function Func_agent_summarize_policy(Agent::Str_Agent; X=-1, TM = Read_transM())
    states = Func_Image2State.(Agent.State_Set)
    if X == -1
        X = Func_agent_π_A_all(Agent)
    end
    Information_matrix = cat(states, X, zeros(length(states),3), dims=2)
    for i = eachindex(states)
        Information_matrix[i,5:7] = TM[states[i]+1,:]
    end
    Information_matrix = Information_matrix[sortperm(states),:]
end
export Func_agent_summarize_policy

function Func_agent_summarize_Q(Agent::Str_Agent, Q::Array{Float64,2}; TM = Read_transM(), β=1.)
    states = Func_Image2State.(Agent.State_Set)
    ΔQ = zeros(size(Q))
    for i = 1:size(Q)[1]
        ΔQ[i,:] = (Q[i,:] .- findmax(Q[i,:])[1]) .* β
    end
    Information_matrix = cat(states, ΔQ, zeros(length(states),3), dims=2)
    for i = eachindex(states)
        Information_matrix[i,5:7] = TM[states[i]+1,:]
    end
    Information_matrix = Information_matrix[sortperm(states),:]
end
export Func_agent_summarize_Q


function Func_agent_summarize_θ(Agent::Str_Agent; TM = Read_transM())
    states = Func_Image2State.(Agent.State_Set)
    Information_matrix = cat(states, zeros(length(states),3), zeros(length(states),3), dims=2)
    for i = eachindex(states)
        for j = 1:3
            inds = 1:length(states)
            inds = inds[states .== TM[states[i]+1,j]]
            Information_matrix[i,1+j] = findmax(Agent.θ_sas[i,j,inds])[1]
        end
        Information_matrix[i,5:7] = TM[states[i]+1,:]
    end
    Information_matrix = Information_matrix[sortperm(states),:]
end
export Func_agent_summarize_θ
