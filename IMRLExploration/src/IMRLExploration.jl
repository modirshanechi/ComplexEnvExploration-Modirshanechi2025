module IMRLExploration


using Dates
using PyPlot
using Distributed
using CSV, DataFrames, JLD2, MAT

using Random, Distributions, Statistics, StatsBase, HypothesisTests, Bootstrap
using LinearAlgebra, LogExpFunctions
using ConcreteStructs, ComponentArrays

import NLopt, Optim

using FitPopulations
import FitPopulations: parameters, logp, simulate, sample, initialize!
import FitPopulations: gradient_logp, hessian_logp
import FitPopulations: maximize
import FitPopulations: maximize_logp
import FitPopulations: PopulationModel

import Enzyme; Enzyme.API.runtimeActivity!(true)

# defining all structurs, including the data structure, agent structure, etc.
include("Structs_form.jl")  
# general constants defining the model classes and initial parameters for optimization
include("GlobalConstants.jl")

# functions for handling data
include("Functions_for_data.jl")

# functions for defining intrinsic rewards
include("Functions_for_rewards.jl")
# functions for defining the environment
include("Functions_for_environments.jl")

# functions for initializing an agent for modeling
include("Functions_for_model_initialization.jl")
# functions for updating the agent after a new observation
include("Functions_for_model_update.jl")
# functions for agent-environment interactions
include("Functions_for_model_envinteract.jl")
# functions for likelihood and simulations
include("Functions_for_model_likesim.jl")           # for Str_Agent
include("Functions_for_model_likesim_2state.jl")    # for Str_Agent_State + Str_Agent_Policy
# functions for accessing model variables
include("Functions_for_model_access.jl")            # for Str_Agent
include("Functions_for_model_access_2state.jl")     # for Str_Agent_State + Str_Agent_Policy

# functions for MCMC sampling for the Bayesian model comparison
include("Functions_MCMC_random_effects.jl")

# functions for fitting models to data
include("Functions_for_data_fitting.jl")
# basic functions for extending FitPopulations.jl package
include("Functions_for_data_fitting_FitPop.jl")

# some general purpose functions
include("Functions_general.jl")

# functions for plotting
include("Functions_plotting.jl")
include("Functions_plotting_specific.jl")


end # module IMRLExploration