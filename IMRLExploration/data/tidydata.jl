# code for converting the data format in BahveData_Si.mat to a simple CSV file
using IMRLExploration
using DataFrames
using CSV


Data = Read_data_all();
Datadfs = []

for d = Data
    T = sum(length.(d.states))
    df = DataFrame(subject = Int.(ones(T) .* d.Sub))
    df.episode = Int.(vcat([ones(length(d.states[i])) .* i for i = eachindex(d.states)]...))
    df.trial = Int.(vcat([1:length(d.states[i]) for i = eachindex(d.states)]...))
    df.state = image2paperstate.(Func_Image_Arr2Vec(hcat(d.images...)))
    df.action = Int.(vcat(d.actions...) .+ 1)
    df.rt = vcat(d.resp_time...)
    push!(Datadfs,df)
end

Datadf = vcat(Datadfs...)
CSV.write("data/tidydata.CSV", Datadf)
