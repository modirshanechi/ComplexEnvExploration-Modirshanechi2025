# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Agent access functions
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_agent_act(AgentP::Str_Agent_Policy, AgentS::Str_Agent_State)
    AgentS.A_t, AgentS.A_ind_t
end
export Func_agent_act

function Func_agent_state(AgentP::Str_Agent_Policy, AgentS::Str_Agent_State)
    AgentS.S_t, AgentS.S_ind_t
end
export Func_agent_state

function Func_agent_logL(AgentP::Str_Agent_Policy, AgentS::Str_Agent_State)
    a_ind = AgentS.A_ind_t
    return log(AgentP.π_A_t[a_ind])
end
export Func_agent_logL

function Func_agent_π_A(AgentP::Str_Agent_Policy, AgentS::Str_Agent_State)
    return AgentP.π_A_t
end
export Func_agent_π_A

function Func_agent_π_A_chosen(AgentP::Str_Agent_Policy, AgentS::Str_Agent_State)
    a_ind = AgentS.A_ind_t
    return AgentP.π_A_t[a_ind]
end
export Func_agent_π_A_chosen

function Func_agent_accuracy(AgentP::Str_Agent_Policy, AgentS::Str_Agent_State)
    a_ind = AgentS.A_ind_t
    return (findmax(AgentP.π_A_t)[1] == AgentP.π_A_t[a_ind]) /
            sum(AgentP.π_A_t .== AgentP.π_A_t[a_ind])
end
export Func_agent_accuracy

function Func_agent_ent_π_A(AgentP::Str_Agent_Policy, AgentS::Str_Agent_State)
    return -sum(AgentP.π_A_t .* log.(AgentP.π_A_t))
end
export Func_agent_ent_π_A

function Func_agent_Q_MB(AgentP::Str_Agent_Policy, AgentS::Str_Agent_State; extrinsic::Bool=true)
    if extrinsic
        return AgentP.Q_MBe
    else
        return AgentP.Q_MBi
    end
end
export Func_agent_Q_MB

function Func_agent_Q_MF(AgentP::Str_Agent_Policy, AgentS::Str_Agent_State; extrinsic::Bool=true)
    if extrinsic
        return AgentP.Q_MFe
    else
        return AgentP.Q_MFi
    end
end
export Func_agent_Q_MF

function Func_agent_Q_state(AgentP::Str_Agent_Policy, AgentS::Str_Agent_State;
                            extrinsic::Bool=true, MB::Bool=true)
    s_ind = AgentS.S_ind_t
    if extrinsic
        if MB
            return AgentP.Q_MBe[s_ind,:]
        else
            return AgentP.Q_MFe[s_ind,:]
        end
    else
        if MB
            return AgentP.Q_MBi[s_ind,:]
        else
            return AgentP.Q_MFi[s_ind,:]
        end
    end
end
export Func_agent_Q_state

function Func_agent_ΔQei_state(AgentP::Str_Agent_Policy, AgentS::Str_Agent_State)
    if AgentS.Epi == 1
        β_MBe = AgentP.Param.β_MBe_1; β_MBi = AgentP.Param.β_MBi_1
        β_MFe = AgentP.Param.β_MFe_1; β_MFi = AgentP.Param.β_MFi_1
    else
        β_MBe = AgentP.Param.β_MBe_2; β_MBi = AgentP.Param.β_MBi_2
        β_MFe = AgentP.Param.β_MFe_2; β_MFi = AgentP.Param.β_MFi_2
    end

    Qs_MBe = Func_agent_Q_state(AgentP, AgentS, extrinsic = true, MB = true)
    Qs_MFe = Func_agent_Q_state(AgentP, AgentS, extrinsic = true, MB = false)
    Qs_MBi = Func_agent_Q_state(AgentP, AgentS, extrinsic = false, MB = true)
    Qs_MFi = Func_agent_Q_state(AgentP, AgentS, extrinsic = false, MB = false)

    Qs_e = β_MBe .* Qs_MBe .+ β_MFe .* Qs_MFe
    Qs_i = β_MBi .* Qs_MBi .+ β_MFi .* Qs_MFi

    ΔQs_e = findmax(Qs_e)[1] - findmin(Qs_e)[1]
    ΔQs_i = findmax(Qs_i)[1] - findmin(Qs_i)[1]

    return [ΔQs_e, ΔQs_i]
end
export Func_agent_ΔQei_state


function Func_agent_ΔQMBMF_state(AgentP::Str_Agent_Policy, AgentS::Str_Agent_State)
    if AgentS.Epi == 1
        β_MBe = AgentP.Param.β_MBe_1; β_MBi = AgentP.Param.β_MBi_1
        β_MFe = AgentP.Param.β_MFe_1; β_MFi = AgentP.Param.β_MFi_1
    else
        β_MBe = AgentP.Param.β_MBe_2; β_MBi = AgentP.Param.β_MBi_2
        β_MFe = AgentP.Param.β_MFe_2; β_MFi = AgentP.Param.β_MFi_2
    end

    Qs_MBe = Func_agent_Q_state(AgentP, AgentS, extrinsic = true, MB = true)
    Qs_MFe = Func_agent_Q_state(AgentP, AgentS, extrinsic = true, MB = false)
    Qs_MBi = Func_agent_Q_state(AgentP, AgentS, extrinsic = false, MB = true)
    Qs_MFi = Func_agent_Q_state(AgentP, AgentS, extrinsic = false, MB = false)

    Qs_MB = β_MBe .* Qs_MBe .+ β_MBi .* Qs_MBi
    Qs_MF = β_MFe .* Qs_MFe .+ β_MFi .* Qs_MFi

    ΔQs_MB = findmax(Qs_MB)[1] - findmin(Qs_MB)[1]
    ΔQs_MF = findmax(Qs_MF)[1] - findmin(Qs_MF)[1]

    return [ΔQs_MB, ΔQs_MF]
end
export Func_agent_ΔQMBMF_state

function Func_agent_ΔQ_state(AgentP::Str_Agent_Policy, AgentS::Str_Agent_State)
    if AgentS.Epi == 1
        β_MBe = AgentP.Param.β_MBe_1; β_MBi = AgentP.Param.β_MBi_1
        β_MFe = AgentP.Param.β_MFe_1; β_MFi = AgentP.Param.β_MFi_1
    else
        β_MBe = AgentP.Param.β_MBe_2; β_MBi = AgentP.Param.β_MBi_2
        β_MFe = AgentP.Param.β_MFe_2; β_MFi = AgentP.Param.β_MFi_2
    end

    Qs_MBe = Func_agent_Q_state(AgentP, AgentS, extrinsic = true, MB = true)
    Qs_MFe = Func_agent_Q_state(AgentP, AgentS, extrinsic = true, MB = false)
    Qs_MBi = Func_agent_Q_state(AgentP, AgentS, extrinsic = false, MB = true)
    Qs_MFi = Func_agent_Q_state(AgentP, AgentS, extrinsic = false, MB = false)

    Qs_MB = β_MBe .* Qs_MBe .+ β_MBi .* Qs_MBi
    Qs_MF = β_MFe .* Qs_MFe .+ β_MFi .* Qs_MFi
    Qs = Qs_MB .+ Qs_MF

    ΔQ = findmax(Qs)[1] - findmin(Qs)[1]
    return ΔQ
end
export Func_agent_ΔQ_state



function Func_agent_π_A_all(AgentP::Str_Agent_Policy, AgentS::Str_Agent_State)
    if AgentS.Epi == 1
        β_MBe = AgentP.Param.β_MBe_1; β_MBi = AgentP.Param.β_MBi_1
        β_MFe = AgentP.Param.β_MFe_1; β_MFi = AgentP.Param.β_MFi_1
    else
        β_MBe = AgentP.Param.β_MBe_2; β_MBi = AgentP.Param.β_MBi_2
        β_MFe = AgentP.Param.β_MFe_2; β_MFi = AgentP.Param.β_MFi_2
    end


    Q_MB = β_MBe .* AgentP.Q_MBe .+ β_MBi .* AgentP.Q_MBi
    Q_MF = β_MFe .* AgentP.Q_MFe .+ β_MFi .* AgentP.Q_MFi
    Q = Q_MB .+ Q_MF

    π_A = zeros(size(Q))

    for s = 1:AgentS.state_num
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

function Func_agent_eR_t(AgentP::Str_Agent_Policy, AgentS::Str_Agent_State)
    AgentP.eR_t
end
export Func_agent_eR_t

function Func_agent_eRPE_t(AgentP::Str_Agent_Policy, AgentS::Str_Agent_State)
    AgentP.eRPE_t
end
export Func_agent_eRPE_t

function Func_agent_iR_t(AgentP::Str_Agent_Policy, AgentS::Str_Agent_State)
    AgentP.iR_t
end
export Func_agent_iR_t

function Func_agent_iRPE_t(AgentP::Str_Agent_Policy, AgentS::Str_Agent_State)
    AgentP.iRPE_t
end
export Func_agent_iRPE_t

function Func_agent_summarize_policy(AgentP::Str_Agent_Policy, AgentS::Str_Agent_State; X=-1, TM = Read_transM())
    states = Func_Image2State.(AgentS.State_Set)
    if X == -1
        X = Func_agent_π_A_all(AgentP)
    end
    Information_matrix = cat(states, X, zeros(length(states),3), dims=2)
    for i = eachindex(states)
        Information_matrix[i,5:7] = TM[states[i]+1,:]
    end
    Information_matrix = Information_matrix[sortperm(states),:]
end
export Func_agent_summarize_policy

function Func_agent_summarize_Q(AgentP::Str_Agent_Policy, AgentS::Str_Agent_State, Q::Array{Float64,2}; TM = Read_transM(), β=1.)
    states = Func_Image2State.(AgentS.State_Set)
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


function Func_agent_summarize_θ(AgentP::Str_Agent_Policy, AgentS::Str_Agent_State; TM = Read_transM())
    states = Func_Image2State.(AgentS.State_Set)
    Information_matrix = cat(states, zeros(length(states),3), zeros(length(states),3), dims=2)
    for i = eachindex(states)
        for j = 1:3
            inds = 1:length(states)
            inds = inds[states .== TM[states[i]+1,j]]
            Information_matrix[i,1+j] = findmax(AgentP.θ_sas[i,j,inds])[1]
        end
        Information_matrix[i,5:7] = TM[states[i]+1,:]
    end
    Information_matrix = Information_matrix[sortperm(states),:]
end
export Func_agent_summarize_θ
