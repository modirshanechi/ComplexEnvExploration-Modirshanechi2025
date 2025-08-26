# ------------------------------------------------------------------------------
# default_optimizer
# ------------------------------------------------------------------------------
import FitPopulations: default_optimizer
function default_optimizer(model::Str_Agent, parameters, fixed)
      DoubleOptimizer()
end

# ------------------------------------------------------------------------------
# optimizer function
# ------------------------------------------------------------------------------
@concrete struct DoubleOptimizer
      NoGrad_maxeval
      NoGrad_ftol
      NoGrad_xtol_rel
      Grad_maxeval
      Grad_min_grad_norm
end
function DoubleOptimizer(; 
                  NoGrad_ftol = 1e-3, NoGrad_xtol_rel = 1e-3, 
                  NoGrad_maxeval = 10, Grad_maxeval = 10,                        # for debugging and test
                  #NoGrad_maxeval = 100000, Grad_maxeval = 1,                    # for coarse-grained
                  #NoGrad_maxeval = 1, Grad_maxeval = 10000,                     # for fine tuning
                  #NoGrad_maxeval = 500, Grad_maxeval = 10000,                   # for recovery
                  #NoGrad_maxeval = 10000, Grad_maxeval = 0,                     # for back-leak coarse grained
                  #NoGrad_maxeval = 50, Grad_maxeval = 3000,                     # for back-leak fine tuning
                  Grad_min_grad_norm = 1e-3)
      DoubleOptimizer(NoGrad_maxeval, NoGrad_ftol, NoGrad_xtol_rel,
                      Grad_maxeval, Grad_min_grad_norm)
end
export DoubleOptimizer


@concrete struct MyLN_SBPLX 
      maxeval
      ftol
      xtol_rel
end

# ------------------------------------------------------------------------------
# maximize function
# ------------------------------------------------------------------------------
function maximize(opt::DoubleOptimizer, g!, params)
      opt1 = MyLN_SBPLX(opt.NoGrad_maxeval, opt.NoGrad_ftol,opt.NoGrad_xtol_rel)
      res1 = maximize(opt1, g!, params)

      dp = zero(g!.xmax); g!(true, dp, nothing, g!.xmax)
      dp_max = maximum(abs, dp)
      if dp_max < opt.Grad_min_grad_norm
            return res1  
      else
            println("Gradient inf-norm: " * string(dp_max) * 
                        " > " * string(opt.Grad_min_grad_norm) *
                        "; switching to gradient-based optimization")
            println(" eval   | current    | best")
            println("_"^33)
            opt2 = FitPopulations.OptimisersOptimizer(maxeval = opt.Grad_maxeval)
            res2 = maximize(opt2, g!, g!.xmax)
            return merge(res2, (; result_opt1 = res1))
      end
end

function maximize(opt::MyLN_SBPLX, g!, params)
      o = FitPopulations.Opt(:LN_SBPLX, length(params))
      o.max_objective = (params, dparams) -> g!(true, dparams, nothing, params)
      o.maxeval   = opt.maxeval
      o.ftol_abs      = opt.ftol
      o.xtol_rel  = opt.xtol_rel
      _, _, extra = FitPopulations.NLopt.optimize(o, params)
      (; extra)
end
