# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Reading data and converting it to the Input format
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# reading data of subject Sub
function Read_data(Sub::Int64)
    path = string("data/BehaveData_S",string(Sub),".mat")
    vars = matread(path)
    vars = vars["Input"]
    for i = 1:length(vars["states"])
        vars["states"][i] = Int.(vars["states"][i][:])
        vars["actions"][i] = Int.(vars["actions"][i][:])
        vars["trial_time"][i] = vars["trial_time"][i][:]
        vars["resp_time"][i] = vars["resp_time"][i][:]
        vars["images"][i] = Int.(vars["images"][i])
    end
    Str_Input(; Sub=Sub,
                Gender=Int(vars["Gender"][1]),
                states=Vector{Vector{Int64}}(vars["states"][:]),
                actions=Vector{Vector{Int64}}(vars["actions"][:]),
                images=Vector{Matrix{Int64}}(vars["images"][:]),
                trial_time=Vector{Vector{Float64}}(vars["trial_time"][:]),
                resp_time=Vector{Vector{Float64}}(vars["resp_time"][:]),
                TM=(Int.(vars["TM"][1]).-1))
end
export Read_data

# reading demographic data of subject Sub
function Read_data_demo(Sub::Int64)
    path = string("data/BehaveData_S",string(Sub),".mat")
    vars = matread(path)
    vars = vars["Input"]
    return Int(vars["Gender"][1]), Int(vars["Age"][1])
end
export Read_data_demo

function Read_data_all(;Sub_set = 1:63)
    [Read_data(Sub) for Sub = Sub_set]
end
export Read_data_all

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Reading transition matrix
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Read_transM()
    path = string("data/Environment.mat")
    Env = matread(path)
    return Int.(Env["transM"] .- 1)
end
export Read_transM

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Accessing a subject's goal type
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_GoalType(Data::Str_Input; ifone = false)
    if !ifone
        if Data.images[1][1,end] != 2
            #error("No goal!")
            return -1
        end
        return Data.images[1][2,end]
    else
        for img = Data.images
            if img[1,end] == 2
                return img[2,end]
            end
        end
        return -1
    end
end
export Func_GoalType

function Func_Goal_type_conv(x)
    if x == 0
        return Symbol("2CHF")
    elseif x==1
        return Symbol("3CHF")
    elseif x==2
        return Symbol("4CHF")
    else
        error("Invalid Goal type")
    end
end
export Func_Goal_type_conv

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Extracting information based on groups
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# As input, it takes a Matrix of measurements X, where X[i,j] corresponds to 
# data of subject i in episode j. Goal_type_Set specifice the which goal type
# different subjects belong to.
function Func_group_based_information(X::Array{Float64,2},
                                      Goal_type_Set::Array{Int64,1};
                                      two_parts = false,
                                      Bootstrap_size = 1000,
                                      median_parts = false,
                                      Goal_types=1:3)
    if two_parts    # whether we treat episode 2-5 separately or all together
        X_temp = zeros(size(X)[1],2)
        X_temp[:,1] = X[:,1]
        for i = 1:size(X_temp)[1]
            if median_parts
                X_temp[i,2] = median(filter(!isnan,X[i,2:end]))
            else
                X_temp[i,2] = mean(filter(!isnan,X[i,2:end]))
            end
        end
        X = deepcopy(X_temp)
    end
    Epi_Num = size(X)[2]
    Mean_Epi_X = zeros(3,Epi_Num)
    SE_Epi_X = zeros(3,Epi_Num)
    Med_Epi_X = zeros(3,Epi_Num)
    SE_Med_Epi_X = zeros(3,Epi_Num)
    Q25_Epi_X = zeros(3,Epi_Num)
    Q75_Epi_X = zeros(3,Epi_Num)

    for i=Goal_types
        for Epi = 1:Epi_Num
            temp = filter(!isnan, X[Goal_type_Set .== (i-1),Epi])
            Mean_Epi_X[i,Epi] = mean(temp)
            SE_Epi_X[i,Epi] = std(temp) / sqrt(sum(Goal_type_Set .== (i-1)))

            Med_Epi_X[i,Epi] = median(temp)
            SE_Med_Epi_X[i,Epi] = stderror(bootstrap(median,temp,
                                           BasicSampling(Bootstrap_size)))[1]

            Q25_Epi_X[i,Epi] = quantile(temp, 0.25)
            Q75_Epi_X[i,Epi] = quantile(temp, 0.75)
        end
    end
    return Mean_Epi_X, SE_Epi_X, Med_Epi_X, SE_Med_Epi_X, Q25_Epi_X, Q75_Epi_X, X
end
export Func_group_based_information


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Accessing a subject's episode lengths
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_EpiLenghts(Data)
    Lens = length.(Data.states)
    Done = zeros(length(Lens))
    for i=1:length(Lens)
        if length(Data.states[i])>0
            Done[i] = (Data.states[i][end]==0)
        else
            Done[i] = 0
        end
    end
    return Lens,Done
end
export Func_EpiLenghts

function Func_EpiLenghts_all(Data)
    Sub_Num = length(Data)
    Epi_Num = length(Data[1].states)

    Epi_Len = zeros(Sub_Num, Epi_Num)
    Epi_Don = zeros(Sub_Num, Epi_Num)
    for i = 1:Sub_Num
        Epi_Len[i,:],Epi_Don[i,:] = Func_EpiLenghts(Data[i])
    end
    return Epi_Len, Epi_Don
end
export Func_EpiLenghts_all

function Func_EpiLenghts_statistics(Data)
    Epi_Len, Epi_Don = Func_EpiLenghts_all(Data)
    Goal_type_Set = Func_GoalType.(Data)

    return Func_group_based_information(Epi_Len,Goal_type_Set,two_parts = false)
end
function Func_EpiLenghts_statistics(Data,
                                    Goal_type_Set::Array{Int64,1})
    Epi_Len, Epi_Don = Func_EpiLenghts_all(Data)
    return Func_group_based_information(Epi_Len,Goal_type_Set,two_parts = false)
end
export Func_EpiLenghts_statistics

function Func_EpiLenghts_statistics_2parts(Data;
                                        median_parts = false)
    Epi_Len, Epi_Don = Func_EpiLenghts_all(Data)
    Goal_type_Set = Func_GoalType.(Data)
    return Func_group_based_information(Epi_Len,Goal_type_Set,two_parts = true,
                                        median_parts = median_parts)
end
function Func_EpiLenghts_statistics_2parts(Data,
                                        Goal_type_Set::Array{Int64,1};
                                        median_parts = false)
    Epi_Len, Epi_Don = Func_EpiLenghts_all(Data)
    return Func_group_based_information(Epi_Len,Goal_type_Set,two_parts = true,
                                        median_parts = median_parts)
end
export Func_EpiLenghts_statistics_2parts


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Removing outliers
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_DetectOutliers(Data; Epi_start = 2,
                             N_thresh = 3,
                             Goal_type_Set = -1,
                             Goal_types = 1:3,
                             N_max = Inf)
    Sub_Num = length(Data)
    Epi_Num = length(Data[1].states)

    Epi_Len, Epi_Don = Func_EpiLenghts_all(Data)
    if Goal_type_Set == -1
        Goal_type_Set = Func_GoalType.(Data)
    end

    # Finding the ones who didn't finish
    Quit_Subject = (sum(Epi_Don,dims=2) .< Epi_Num)[:]
    Quit_Subject = Quit_Subject .| ((sum(Epi_Len,dims=2) .> N_max)[:])

    # The one with N_thresh times number of actions in  their group
    Long_Subject = zeros(Sub_Num)
    y = sum(Epi_Len[:,Epi_start:end],dims=2)
    y_1 = y[Quit_Subject .== 0]
    Goal_type_Set_1 = Goal_type_Set[Quit_Subject .== 0]
    for i=Goal_types
        my = mean(y_1[Goal_type_Set_1 .== (i-1)])
        inds = 1:Sub_Num
        for j = inds[Goal_type_Set .== (i-1)]
            Long_Subject[j] = (y[j] > (my*N_thresh))
        end
    end
    Long_Subject = Bool.(Long_Subject)

    Outliers = Long_Subject .| Quit_Subject

    return Outliers, Long_Subject, Quit_Subject
end
export Func_DetectOutliers


function Read_processed_data(; Plotting = false, Sub_set = 1:63,
                             Epi_start = 2, Data = -1, Goal_type_Set = -1,
                             N_thresh = 3, N_max = Inf,
                             σ = 0.2,
                             Colors = ["#004D66","#0350B5","#00CCF5"],
                             Goal_types = 1:3)
    # Loading data
    if Data == -1
        Data = Read_data_all(Sub_set = Sub_set)
    end
    Sub_Num = length(Data)
    Epi_Num = length(Data[1].states)

    # Extracting subjects' info
    Epi_Len, Epi_Don = Func_EpiLenghts_all(Data)
    if Goal_type_Set == -1
        Goal_type_Set = Func_GoalType.(Data)
    end
    Sub_Num = length(Data)

    # Total episode length and removing outliers
    if Plotting
        y = sum(Epi_Len[:,2:end],dims=2)
        x_0 = (Goal_types)
        fig = figure(figsize=(12,7)); ax = gca()
        for i=Goal_types
            x = x_0[i]
            my = mean(y[Goal_type_Set .== (i-1)])
            ax.bar(x,my,color=Colors[i])
        end
        for i=Goal_types
            x = x_0[i]
            my = mean(y[Goal_type_Set .== (i-1)])
            dy = std(y[Goal_type_Set .== (i-1)]) ./ sqrt(sum(Goal_type_Set .== (i-1)))
            ax.plot([x_0[1]-1, x_0[end]+1], [1,1] .* my * N_thresh, linestyle="--", color=Colors[i])
            ax.errorbar(x,my,yerr=dy,color="k",
                        linewidth=1,drawstyle="steps",linestyle="",capsize=3)
            inds = 1:length(Goal_type_Set)
            for j = inds[Goal_type_Set .== (i-1)]
                ax.plot(x .+ 2*σ*(rand() - 0.5),y[j],".k",alpha = 0.5)
            end
        end
        ax.set_xticks(x_0)
        ax.set_xlim(x_0[1]-1, x_0[end]+1)
        ax.set_xticklabels(["CHF2","CHF3","CHF4"])
        ax.set_ylabel("Average number of actions")
    end

    # Removing bad data
    Outliers, Long_Subject, Quit_Subject = Func_DetectOutliers(Data,
                                               Epi_start = Epi_start,
                                               N_thresh = N_thresh,
                                               N_max = N_max,
                                               Goal_type_Set = Goal_type_Set)
    Data = Data[Outliers .== 0]
    Goal_type_Set = Goal_type_Set[Outliers .== 0]

    # Extracting subjects' info
    Sub_Num = length(Data)

    return Outliers, Long_Subject, Quit_Subject, Data,
           Goal_type_Set, Sub_Num
end
export Read_processed_data

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Accessing a subject's Gender
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_Gender(Data::Str_Input)
    Data.Gender
end
export Func_Gender

function Func_Gender_conv(x)
    if x == 0
        return Symbol("Female")
    elseif x==1
        return Symbol("Male")
    else
        error("Invalid Gender")
    end
end
export Func_Gender_conv

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Number of states
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_StateCount(Data::Str_Input,S_Numb=9+1)
    Epi_Num = length(Data.states)
    StateCount = zeros(Epi_Num,S_Numb)
    for state = 1:S_Numb
        for Epi = 1:Epi_Num
            StateCount[Epi,state] = sum(Data.states[Epi] .== (state-1))
        end
    end
    return StateCount
end
export Func_StateCount

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Number of images
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_ImageCount(Data::Str_Input;I_Numb=49+1)
    Epi_Num = length(Data.states)
    ImageCount = zeros(Epi_Num,I_Numb)
    for Epi = 1:Epi_Num
        temp = Data.images[Epi][2, Data.images[Epi][1,:].==1]
        for image = 1:I_Numb
            ImageCount[Epi,image] = sum(temp .== (image-1))
        end
    end
    Image_Seen = sum(cumsum(ImageCount,dims=1) .> 0,dims=2)
    return (ImageCount,Image_Seen)
end
export Func_ImageCount

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Image array to image vector
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_Image_Arr2Vec(images::Array{Int64,2})
    T = size(images)[2]
    images_vec = Array{Array{Int64,1},1}()
    for t=1:T
        push!(images_vec,images[:,t])
    end
    return images_vec
end
export Func_Image_Arr2Vec

function Func_Image_Vec2Arr(images::Array{Array{Int64,1},1})
    T = length(images)
    images_arr = zeros(2,T)
    for t=1:T
        images_arr[:,t] = images[t][:]
    end
    return Int64.(images_arr)
end
export Func_Image_Vec2Arr


function image2paperstate(image)
    if image[1] == 0
        if image[2] < 7
            return string(image[2] + 1)
        else
            return string(image[2])
        end
    elseif image[1] == 1
        return "S-" * string(image[2] + 1)
    elseif image[1] == 2
        if image[2] == 0
            return "2CHF"
        elseif image[2] == 1
            return "3CHF"
        elseif image[2] == 2
            return "4CHF"
        end
    end
end
export image2paperstate
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Action and response-time series
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_ActTime(Data::Str_Input;State=1,Epi=1,
                      TM=Read_transM(),Traps = [8,9], Inf_Loop = [7])
    Act_series = Func_ActType.(Data.actions[Epi][Data.states[Epi] .== State],
                               State; TM=TM,Traps = Traps, Inf_Loop = Inf_Loop)
    RespTime_series = Data.resp_time[Epi][Data.states[Epi] .== State]
    return (Act_series,RespTime_series)
end
export Func_ActTime

function Func_ActTime_Image(Data::Str_Input;Image=1,Epi=1,TM=Read_transM())
    State = 7
    inds = (Data.states[Epi] .== State) .& (Data.images[Epi][2,:] .== Image)
    Act_series = Func_ActType.(Data.actions[Epi][inds],
                               7; TM=TM,Traps = [8,9], Inf_Loop = [7])
    RespTime_series = Data.resp_time[Epi][inds]
    return (Act_series,RespTime_series)
end
export Func_ActTime_Image

function Func_ActType(act, state; TM=Read_transM(),
                      Traps = [8,9], Inf_Loop = [7], scores = [-1,0,0,1])
    next_state = TM[state+1,act+1]
    if next_state ∈ Traps
        return scores[1]                # Trap
    elseif next_state ∈ Inf_Loop
        return scores[2]                # Inf
    elseif next_state == state
        return scores[3]                # Stay
    else
        return scores[4]                # Progress
    end
end
export Func_ActType

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Computing average progress
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_action_progress_vec(Data, State::Array{Int64,1};
                                  inds = -1, Epi = 1, K_movemean = 0, scores = [-1,0,0,1],
                                  TM=Read_transM(), Traps = [8,9], Inf_Loop = [7])
    if inds != -1
        Data = Data[inds]
    end
    Sub_Num = length(Data)

    Actions = Array{Array{Float64,1},1}(undef,Sub_Num)

    for Sub = 1:Sub_Num
      Obs = Data[Sub].states[Epi]
      Act = Data[Sub].actions[Epi]

      isdesiredObs = zeros(size(Obs))
      for i=1:length(Obs)
          isdesiredObs[i] = (Obs[i] ∈ State)
      end
      Obs = Obs[isdesiredObs .== 1]
      Act = Act[isdesiredObs .== 1]

      N = length(Act)
      if N>0
          Actions[Sub] = zeros(N)
          for i = 1:N
              Actions[Sub][i] = Func_ActType(Act[i], Obs[i]; TM=TM, Traps=Traps,
                                              Inf_Loop=Inf_Loop, scores=scores)
          end
          Actions[Sub] = Func_movmean(Actions[Sub],K_movemean)
      else
          Actions[Sub] = deepcopy(Act)
      end
    end

    x_len = sort(length.(Actions),rev=true)[2]
    y = zeros(x_len)
    dy = zeros(x_len)
    y_med = zeros(x_len)
    N = zeros(x_len)
    p_values = zeros(x_len)
    for i = 1:x_len
        temp = []
        for Sub = 1:Sub_Num
            if length(Actions[Sub]) >= i
                push!(temp, Actions[Sub][i])
            end
        end
        y[i] = mean(temp)
        y_med[i] = median(temp)
        N[i] = length(temp)
        dy[i] = std(temp) / sqrt(length(temp))
        p_values[i] = pvalue(OneSampleTTest(Float64.(temp)))
    end

    return y, dy, N, Actions, y_med, p_values
end
export Func_action_progress_vec

function Func_action_progress_vec(Data, State::Int64;
                                  inds = -1, Epi = 1, K_movemean = 0,
                                  scores = [-1,0,0,1],
                                  TM=Read_transM(), Traps = [8,9], Inf_Loop = [7])
    return Func_action_progress_vec(Data,[State]; inds = inds, Epi = Epi,
                                      K_movemean = K_movemean,
                                      scores = scores,
                                      TM=TM, Traps=Traps, Inf_Loop=Inf_Loop)
end
export Func_action_progress_vec


function Func_reaction_time_vec(Data, State::Array{Int64,1};
                                  inds = -1, Epi = 1, K_movemean = 0)
    if inds != -1
        Data = Data[inds]
    end
    Sub_Num = length(Data)

    ReactionTimes = Array{Array{Float64,1},1}(undef,Sub_Num)

    for Sub = 1:Sub_Num
      Obs = Data[Sub].states[Epi]
      RTime = Data[Sub].resp_time[Epi]

      isdesiredObs = zeros(size(Obs))
      for i=eachindex(Obs)
          isdesiredObs[i] = (Obs[i] ∈ State)
      end
      ReactionTimes[Sub] = RTime[isdesiredObs .== 1]
    end

    x_len = sort(length.(ReactionTimes),rev=true)[2]
    y = zeros(x_len)
    dy = zeros(x_len)
    y_med = zeros(x_len)
    N = zeros(x_len)
    for i = 1:x_len
        temp = []
        for Sub = 1:Sub_Num
            if length(ReactionTimes[Sub]) >= i
                push!(temp, ReactionTimes[Sub][i])
            end
        end
        y[i] = mean(temp)
        y_med[i] = median(temp)
        N[i] = length(temp)
        dy[i] = std(temp) / sqrt(length(temp))
    end

    return y, dy, N, ReactionTimes, y_med
end
export Func_reaction_time_vec

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Computing the lenght of visit of a set of desired states
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_desired_states_visit(Data; inds = -1, Epi = 1,
                                   K_movemean = 0,
                                   TM=Read_transM(), DesiredStates = [8,9],
                                   first_half = false,second_half = false)
    if inds != -1
        Data = Data[inds]
    end
    Sub_Num = length(Data)

    Lenghts = Array{Array{Float64,1},1}(undef,Sub_Num)

    for Sub = 1:Sub_Num
        Obs = Data[Sub].states[Epi]
        if first_half
            Obs = Obs[1:Int64(ceil(length(Obs)/2))]
        elseif second_half
            Obs = Obs[Int64(floor(length(Obs)/2)):length(Obs)]
        end
        isdesiredObs = zeros(size(Obs))
        for i=eachindex(Obs)
            isdesiredObs[i] = (Obs[i] ∈ DesiredStates)
        end
        n = length(Obs)
        Inds = Array(2:n)
        y = (diff(isdesiredObs) .+ 1) ./ 2
        if sum(y .== 0) > sum(y .== 1)
            Lenghts[Sub] = Inds[y .== 0] .- vcat([1],Inds[y .== 1])
        elseif sum(y .== 0) < sum(y .== 1)
            Lenghts[Sub] = vcat(Inds[y .== 0],[Inds[end]+1]) .- Inds[y .== 1]
        elseif (isdesiredObs[1] == 1) & (isdesiredObs[end] == 1)
            Lenghts[Sub] = vcat(Inds[y .== 0],[Inds[end]+1]) .-
                           vcat([1],Inds[y .== 1])
        else
            Lenghts[Sub] = Inds[y .== 0] .- Inds[y .== 1]
        end
        Lenghts[Sub] = Func_movmean(Lenghts[Sub],K_movemean)
    end

    x_len = sort(length.(Lenghts),rev=true)[2]
    y = zeros(x_len)
    dy = zeros(x_len)
    y_med = zeros(x_len)
    y_Q25 = zeros(x_len)
    y_Q75 = zeros(x_len)
    N = zeros(x_len)
    for i = 1:x_len
        temp = []
        for Sub = 1:Sub_Num
            if length(Lenghts[Sub]) >= i
                push!(temp, Lenghts[Sub][i])
            end
        end
        y[i] = mean(temp)
        y_med[i] = median(temp)
        y_Q25[i] = quantile(temp, 0.25)
        y_Q75[i] = quantile(temp, 0.75)
        N[i] = length(temp)
        dy[i] = std(temp) / sqrt(length(temp))
    end

    return y, dy, N, Lenghts, y_med, y_Q25, y_Q75
end
export Func_desired_states_visit


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Function for converting graph drawing dataframe rows to matrix
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function dfg2matg(dfg,n;
    Names = ["Ts", "P1", "P2", "P3", "P4", "P5", "P6", "X", "N44"],
    NamesOrg = ["P8", "P9", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "N44"])
    
    Ns = length(NamesOrg)
    X = zeros(Ns,Ns) .* NaN
    for i = 1:Ns
            for j = (i+1):Ns
                    NN1 = NamesOrg[i] * "-" * NamesOrg[j]
                    NN2 = NamesOrg[j] * "-" * NamesOrg[i]
                    if NN1 ∈ names(dfg)
                            X[i,j] = dfg[n,NN1]
                    elseif NN2 ∈ names(dfg)
                            X[i,j] = dfg[n,NN2]
                    else
                            error("The combination is not in the set!")
                    end
            end
    end
    Ns = length(Names)
    X2 = X[2:end,2:end]; X2[1,1] = 0
    X2[1,:] .+= X[1,2:end]
    X2[1,:] .= (X2[1,:] .> 0) .* 1
    return X2
end
# dfg2matg returns the ground truth if no dataframe is passed to it
function dfg2matg(;
    Names = ["Ts", "P1", "P2", "P3", "P4", "P5", "P6", "X", "N44"])
    
    Ns = length(Names)
    X = zeros(Ns,Ns) .* NaN
    for i = 1:Ns
        if i < (Ns-2)
            X[i,i+1] = 1.
            for j = (i+2):Ns
                X[i,j] = 0.
            end
        else
            for j = (i+1):Ns
                X[i,j] = 0.
            end
        end
    end
    X[1,1:(Ns-2)] .= 1.
    X[5,Ns] = 1.
    return X
end
export dfg2matg
# ldfg2matg2goal returns only the path to the goal
function ldfg2matg2goal(;
    Names = ["Ts", "P1", "P2", "P3", "P4", "P5", "P6", "X", "N44"])
    
    Ns = length(Names)
    X = zeros(Ns,Ns) .* NaN
    for i = 1:Ns
        if i < (Ns-2)
            X[i,i+1] = 1.
            for j = (i+2):Ns
                X[i,j] = 0.
            end
        else
            for j = (i+1):Ns
                X[i,j] = 0.
            end
        end
    end
    X[1,2] = 1.
    X[1,1] = 0.
    X[1,3:(Ns-2)] .= 0.
    X[5,Ns] = 0.
    return X
end
export ldfg2matg2goal
# dfg2matg2HubMask 2-hub connections
function dfg2matg2HubMask(;
    Names = ["Ts", "P1", "P2", "P3", "P4", "P5", "P6", "X", "N44"])
    
    Ns = length(Names)
    X = zeros(Ns,Ns) .* NaN
    X[1,Ns] = 1.
    for i = 2:Ns
        if i < (Ns-3)
            X[i,i+2] = 1.
        end
    end
    X[1,Ns] = 1.
    X[4,Ns] = 1.
    X[6,Ns] = 1.
    return X
end
export dfg2matg2HubMask

function matg_accscore(X_truth::Matrix, X::Matrix)
    x1 = X_truth[:];    x1 = x1[isnan.(x1) .== 0]
    x2 = X[:];          x2 = x2[isnan.(x2) .== 0]
    return matg_accscore(x1,x2)
end
function matg_accscore(X_truth::Vector, X::Vector)
    x1 = X_truth; x2 = X
    s1 = sum(x1 .* x2) / sum(x1)
    s0 = sum((1 .- x1) .* x2) / sum(1 .- x1)
    return (s1 - s0)
end
export matg_accscore

function matg_2goalscore(X_truth::Matrix, X::Matrix)
    x1 = X_truth[:];    x1 = x1[isnan.(x1) .== 0]
    x2 = X[:];          x2 = x2[isnan.(x2) .== 0]
    return sum(x1 .* x2) ./ sum(x1)
end
export matg_2goalscore