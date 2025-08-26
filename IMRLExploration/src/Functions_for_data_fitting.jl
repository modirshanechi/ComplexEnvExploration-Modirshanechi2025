# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Goal-based-cross-validation
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_GoalCV(nfold; nfolds = 3, nsub = 57)
  Outliers, Long_Subject, Quit_Subject, Data, Goal_type_Set, Sub_Num =
                                      Read_processed_data(Plotting = false)
  df = DataFrame(subind = 1:nsub, goal = Goal_type_Set)
  stratified_cv_indices(df, :goal, nfolds, nfold)
end
function Func_GoalCV(nfold, Goal_type_Set; nfolds = 3)
  nsub = length(Goal_type_Set)
  df = DataFrame(subind = 1:nsub, goal = Goal_type_Set)
  stratified_cv_indices(df, :goal, nfolds, nfold)
end
export Func_GoalCV

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Fitting subject by subject
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_fit_sub_by_sub(;subset = 1:57, model_set = 1:8, init_set = [1],
                    total_leak=true, back_leak=false, Rs= [1.,1.,1.], ws= [1.,1.,1.],
                    ite_trig = 50, Grad_min_grad_norm = 0.01, lambda_l2 = 1e-3,
                    NoGrad_maxeval = 10, Grad_maxeval = 10,
                    Path_Save = "src/01_ModelFitting/Results/SubBySub/")
  # loading data
  Outliers, Long_Subject, Quit_Subject, Data, Goal_type_Set, Sub_Num =
        Read_processed_data(Plotting = false)
  Data = Str_Input2Agents.(Data);
  # initializing the params
  Param = Str_Param(; total_leak=total_leak, back_leak=back_leak)
  @sync @distributed for i_init = init_set
    for n_sub = subset
      for i_model = model_set
        A = Str_Agent_Policy(Param, Rs, NIS_object(ws))
        println("----------------------------------------------------")
        println("----------------------------------------------------")
        @show n_sub
        @show Data[n_sub][1].G_type
        @show keys(model_settings)[i_model]
        @show i_init
        @show η0_initialization[:, i_init]
        println("----------------------------------------------------")
        println("----------------------------------------------------")
        flush(stdout)
        Func_fit_one_sub(A, Data[n_sub], n_sub, i_model, i_init, Path_Save,
                  ite_trig = ite_trig, Grad_min_grad_norm = Grad_min_grad_norm, 
                  NoGrad_maxeval = NoGrad_maxeval, Grad_maxeval = Grad_maxeval,
                  lambda_l2 = lambda_l2)
      end
    end
  end
end
function Func_fit_one_sub(A, data, n_sub, i_model, i_init, Path_Save;
                    ite_trig = 50, Grad_min_grad_norm = 0.01, lambda_l2 = 1e-3,
                    NoGrad_maxeval = 10, Grad_maxeval = 10)
  Path_Save_specific = string(Path_Save, "Sub", string(n_sub), 
                              "_init", string(i_init), 
                              "_model", string(i_model))
  mid_writer = Callback((IterationTrigger(ite_trig), EventTrigger((:start, :end))), 
                  CheckPointSaver(Path_Save_specific * ".jld2", overwrite=true))                            
  result = maximize_logp(data, A,
                        parameters(A,η0_initialization[:, i_init]),
                        fixed = model_settings[i_model],
                        optimizer = DoubleOptimizer(
                                  Grad_min_grad_norm = Grad_min_grad_norm,
                                  NoGrad_maxeval = NoGrad_maxeval, 
                                  Grad_maxeval = Grad_maxeval),
                        hessian_ad  = Val(:ForwardDiff), 
                        gradient_ad = Val(:ForwardDiff),
                        return_g! = true,
                        lambda_l2 = lambda_l2,
                        callbacks = [mid_writer])
  save(Path_Save_specific * "_finall.jld2",
              "logp", result.logp,
              "ηt", NamedTuple(result.parameters),
              "η0", η0_initialization[:, i_init],
              "model", keys(model_settings)[i_model])
end
export Func_fit_sub_by_sub


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Fitting shared population
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_fit_shared_pop(;subset = 1:57, model_set = 1:8, init_set = [1],
                    total_leak=true, back_leak=false,Rs= [1.,1.,1.], ws= [1.,1.,1.],
                    ite_trig = 50, Grad_min_grad_norm = 0.01, lambda_l2 = 1e-3,
                    NoGrad_maxeval = 10, Grad_maxeval = 10,
                    Path_Save = "src/01_ModelFitting/Results/Population/")
  # loading data
  Outliers, Long_Subject, Quit_Subject, Data, Goal_type_Set, Sub_Num =
        Read_processed_data(Plotting = false)
  Data = Str_Input2Agents.(Data)[subset];
  # initializing the params
  Param = Str_Param(; total_leak=total_leak, back_leak=back_leak)
  @sync @distributed for i_model = model_set
    for i_init = init_set
      A = Str_Agent_Policy(Param, Rs, NIS_object(ws))
      pop_A = PopulationModel(A, shared = keys(parameters(A)))
      println("----------------------------------------------------")
      println("----------------------------------------------------")
      @show keys(model_settings)[i_model]
      @show η0_initialization[:, i_init]
      println("----------------------------------------------------")
      println("----------------------------------------------------")
      flush(stdout)
      η0 = η0_initialization[:, i_init]
      Path_Save_specific = Path_Save * "Shared" * "Pop_init" * string(i_init) *
                                       "_model" * string(i_model)
      Func_fit_one_shared_pop(pop_A, Data, i_model, η0, Path_Save_specific,
                    ite_trig = ite_trig, Grad_min_grad_norm = Grad_min_grad_norm, 
                    NoGrad_maxeval = NoGrad_maxeval, Grad_maxeval = Grad_maxeval,
                    lambda_l2 = lambda_l2)
    end
  end
end
function Func_fit_shared_pop(LoadPath::String;
                    subset = 1:57, model_set = 1:8,
                    total_leak=true, back_leak=false,Rs= [1.,1.,1.], ws= [1.,1.,1.],
                    ite_trig = 50, Grad_min_grad_norm = 0.01, lambda_l2 = 1e-3,
                    NoGrad_maxeval = 10, Grad_maxeval = 10,
                    Path_Save = "src/01_ModelFitting/Results/PopulationFine/")
  # loading data
  Outliers, Long_Subject, Quit_Subject, Data, Goal_type_Set, Sub_Num =
        Read_processed_data(Plotting = false)
  Data = Str_Input2Agents.(Data)[subset];
  # initializing the params
  Param = Str_Param(; total_leak=total_leak, back_leak=back_leak)
  # reading initial values
  η0df = CSV.read(LoadPath, DataFrame)
  @sync @distributed for i_model = model_set
    A = Str_Agent_Policy(Param, Rs, NIS_object(ws))
    pop_A = PopulationModel(A, shared = keys(parameters(A)))
    η0 = η0df[:,string(keys(model_settings)[i_model])]
    println("----------------------------------------------------")
    println("----------------------------------------------------")
    @show keys(model_settings)[i_model]
    @show η0
    println("----------------------------------------------------")
    println("----------------------------------------------------")  
    flush(stdout)
    Path_Save_specific = Path_Save * "Shared" * "Pop_model" * string(i_model)
    Func_fit_one_shared_pop(pop_A, Data, i_model, η0, Path_Save_specific,
                  ite_trig = ite_trig, Grad_min_grad_norm = Grad_min_grad_norm,
                  NoGrad_maxeval = NoGrad_maxeval, Grad_maxeval = Grad_maxeval, 
                  lambda_l2 = lambda_l2)
  end
end
function Func_fit_one_shared_pop(A, data, i_model, η0, Path_Save_specific;
                    ite_trig = 50, Grad_min_grad_norm = 0.01, lambda_l2 = 1e-3,
                    NoGrad_maxeval = 10, Grad_maxeval = 10)
  mid_writer = Callback((IterationTrigger(ite_trig), EventTrigger((:start, :end))), 
                  CheckPointSaver(Path_Save_specific * ".jld2", overwrite=true))
  result = maximize_logp(data, A,
                        parameters(A,η0),
                        fixed = model_settings[i_model],
                        optimizer = DoubleOptimizer(
                                  Grad_min_grad_norm = Grad_min_grad_norm,
                                  NoGrad_maxeval = NoGrad_maxeval, 
                                  Grad_maxeval = Grad_maxeval),
                        hessian_ad  = Val(:ForwardDiff), 
                        gradient_ad = Val(:ForwardDiff),
                        return_g! = true,
                        lambda_l2 = lambda_l2,
                        callbacks = [mid_writer])
  save(Path_Save_specific * "_finall.jld2",
              "logp", result.logp,
              "ηt", [NamedTuple(result.parameters[i]) for i = eachindex(result.parameters)],
              "popηt", NamedTuple(result.population_parameters),
              "η0", η0,
              "model", keys(model_settings)[i_model])
end
export Func_fit_shared_pop


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Fitting CV shared population
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_fit_shared_pop_CV(;nfold_set = 1:3, nfolds= 3,
                    model_set = 1:8, init_set = [1],
                    total_leak=true, back_leak=false,Rs= [1.,1.,1.], ws= [1.,1.,1.],
                    ite_trig = 50, Grad_min_grad_norm = 0.01, lambda_l2 = 1e-3,
                    NoGrad_maxeval = 10, Grad_maxeval = 10,
                    Path_Save = "src/01_ModelFitting/Results/CrossVal/")
  # loading data
  Outliers, Long_Subject, Quit_Subject, Data, Goal_type_Set, Sub_Num =
        Read_processed_data(Plotting = false)
  Data = Str_Input2Agents.(Data);
  # initializing the params
  Param = Str_Param(; total_leak=total_leak, back_leak=back_leak)
  @sync @distributed for (i_model, nCV) = collect(Iterators.product(model_set,nfold_set))
    sub_train_inds = Func_GoalCV(nCV; nfolds = nfolds)[1]
    for i_init = init_set
      A = Str_Agent_Policy(Param, Rs, NIS_object(ws))
      pop_A = PopulationModel(A, shared = keys(parameters(A)))
      println("----------------------------------------------------")
      println("----------------------------------------------------")
      @show keys(model_settings)[i_model]
      @show nCV
      @show η0_initialization[:, i_init]
      println("----------------------------------------------------")
      println("----------------------------------------------------")
      flush(stdout)
      η0 = η0_initialization[:, i_init]
      Path_Save_specific = Path_Save * "Shared" * "PopCVfold" * string(nCV) *
                          "_init" * string(i_init) * "_model" * string(i_model)
      Func_fit_one_shared_pop(pop_A, Data[sub_train_inds], 
                  i_model, η0, Path_Save_specific,
                  ite_trig = ite_trig, Grad_min_grad_norm = Grad_min_grad_norm, 
                  NoGrad_maxeval = NoGrad_maxeval, Grad_maxeval = Grad_maxeval,
                  lambda_l2 = lambda_l2)
    end
  end
end
function Func_fit_shared_pop_CV(LoadPath::String;
                    nfold_set = 1:3, nfolds= 3, model_set = 1:8,
                    total_leak=true, back_leak=false,Rs= [1.,1.,1.], ws= [1.,1.,1.],
                    ite_trig = 50, Grad_min_grad_norm = 0.01, lambda_l2 = 1e-3,
                    NoGrad_maxeval = 10, Grad_maxeval = 10,
                    Path_Save = "src/01_ModelFitting/Results/CrossValFine/")
  # loading data
  Outliers, Long_Subject, Quit_Subject, Data, Goal_type_Set, Sub_Num =
        Read_processed_data(Plotting = false)
  Data = Str_Input2Agents.(Data);
  # initializing the params
  Param = Str_Param(; total_leak=total_leak, back_leak=back_leak)
  # reading initial values
  η0df = CSV.read(LoadPath, DataFrame)
  @sync @distributed for (i_model, nCV) = collect(Iterators.product(model_set,nfold_set))
    sub_train_inds = Func_GoalCV(nCV; nfolds = nfolds)[1]
    A = Str_Agent_Policy(Param, Rs, NIS_object(ws))
    pop_A = PopulationModel(A, shared = keys(parameters(A)))
    η0 = η0df[:,string(keys(model_settings)[i_model]) * "_f" * string(nCV)]
    println("----------------------------------------------------")
    println("----------------------------------------------------")
    @show keys(model_settings)[i_model]
    @show nCV
    @show η0
    println("----------------------------------------------------")
    println("----------------------------------------------------")
    flush(stdout)
    Path_Save_specific = Path_Save * "Shared" * "PopCVfold" * string(nCV) *
                        "_model" * string(i_model)
    Func_fit_one_shared_pop(pop_A, Data[sub_train_inds], 
                i_model, η0, Path_Save_specific,
                ite_trig = ite_trig, Grad_min_grad_norm = Grad_min_grad_norm, 
                NoGrad_maxeval = NoGrad_maxeval, Grad_maxeval = Grad_maxeval,
                lambda_l2 = lambda_l2)
  end
end
export Func_fit_shared_pop_CV

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Fitting CV shared population: recovery
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_fit_shared_pop_CV_recovery(
                    ParamLoadPath::String, DataLoadPath::String;
                    nfold_set = 1:3, nfolds= 3, model_set = 5:8, rep_set = 1:5,
                    total_leak=true, back_leak=false,Rs= [1.,1.,1.], ws= [1.,1.,1.],
                    ite_trig = 50, Grad_min_grad_norm = 0.01, lambda_l2 = 1e-3,
                    NoGrad_maxeval = 10, Grad_maxeval = 10,
                    Path_Save = "src/03_ModelRecovery/Results/")
  # initializing the params
  Param = Str_Param(; total_leak=total_leak, back_leak=back_leak)
  # reading initial values
  η0df = CSV.read(ParamLoadPath, DataFrame)
  model_set_true = deepcopy(model_set)
  @sync @distributed for (i_model_true, i_model, nCV, i_rep) = 
        collect(Iterators.product(model_set_true, model_set, nfold_set, rep_set))
    #
    temp = load(DataLoadPath * string(keys(model_settings)[i_model_true]) * 
                      "/SData_rep" * string(i_rep) * ".jld2")
    SGoal_type_Set = temp["SGoal_type_Set"]
    SData = Str_Input2Agents.(temp["SData"])
    sub_train_inds = Func_GoalCV(nCV, SGoal_type_Set; nfolds = nfolds)[1]
    A = Str_Agent_Policy(Param, Rs, NIS_object(ws))
    pop_A = PopulationModel(A, shared = keys(parameters(A)))
    η0 = η0df[:,string(keys(model_settings)[i_model])]
    println("----------------------------------------------------")
    println("----------------------------------------------------")
    @show keys(model_settings)[i_model_true]
    @show keys(model_settings)[i_model]
    @show nCV
    @show i_rep
    @show η0
    println("----------------------------------------------------")
    println("----------------------------------------------------")
    flush(stdout)
    Path_Save_specific = Path_Save * "SharedPopCV" *
                        "_TrueModel"  * string(i_model_true) *
                        "_rep"        * string(i_rep) *
                        "_fold"       * string(nCV) *
                        "_model"      * string(i_model) * "_rec"
    Func_fit_one_shared_pop(pop_A, SData[sub_train_inds], 
                i_model, η0, Path_Save_specific,
                ite_trig = ite_trig, Grad_min_grad_norm = Grad_min_grad_norm, 
                NoGrad_maxeval = NoGrad_maxeval, Grad_maxeval = Grad_maxeval,
                lambda_l2 = lambda_l2)
  end
end
export Func_fit_shared_pop_CV_recovery


