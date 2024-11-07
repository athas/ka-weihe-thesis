import "lib/github.com/diku-dk/linalg/linalg"
import "lib/github.com/diku-dk/cpprandom/random"
import "lib/github.com/diku-dk/linalg/lu"
import "convex"

local module type portfolioopt = {
  type t
  val efficient_return [n] : t -> [n]t -> [n][n]t -> [n]t
  val efficient_risk [n] : t -> [n]t -> [n][n]t -> [n]t
  val efficient_return_and_esg [n] : t -> t -> [n]t -> [n]t -> [n][n]t -> [n]t
  val max_sharpe_ratio [n] : t -> [n]t -> [n][n]t -> [n]t
  val min_volatility [n] : [n][n]t -> [n]t
}

module mk_portfolioopt (T: real) : portfolioopt with t = T.t = {
  type t = T.t

  module rng_engine = minstd_rand
  module rand_T = uniform_real_distribution T rng_engine
  module linalg = mk_linalg T
  module lu = mk_lu T
  module mod = mk_solver T

  def expected_return [n] (x: [n]t) (mus: [n]t) =
    linalg.dotprod x mus

  def expected_risk [n] (x: [n]t) (covs: [n][n]t) =
    (linalg.matmul (linalg.matmul [x] covs) (transpose [x]))[0, 0]

  def efficient_return [n] (target_return: t) (mus: [n]t) (covs: [n][n]t) =
    -- Inequality constraints
    let p = n + 1
    let h (x: [n]t): [p]t =
      ([
       -- expected_return >= target_return
       target_return T.- (expected_return x mus)
       ]
        ++
        -- w >= 0
        map (T.neg) x)
      :> [p]t
    -- Equality constraints
    -- sum(w) = 1
    let A = [replicate n (T.f64 1.0)]
    let b = [(T.f64 1.0)]
    -- Objective function
    let f0 (x: [n]t) = expected_risk x covs
    -- Solve the problem
    in mod.solve_qp_auto h f0 A b

  def efficient_risk [n] (target_risk: t) (mus: [n]t) (covs: [n][n]t) =
    -- Inequality constraints
    let p = n + 1
    let h (x: [n]t): [p]t =
      ([
       -- risk <= target_risk
       target_risk T.+ (expected_risk x covs)
       ]
        ++
        -- w >= 0
        map (T.neg) x)
      :> [p]t
    -- Equality constraints
    -- sum(w) = 1
    let A = [replicate n (T.f64 1.0)]
    let b = [(T.f64 1.0)]
    -- Objective function
    let f0 (x: [n]t) = T.neg (expected_return x mus)
    -- Solve the problem
    in mod.solve_qp_auto h f0 A b

  def efficient_return_and_esg [n] (target_return: t) (target_esg: t) (mus: [n]t) (esg: [n]t) (covs: [n][n]t) =
    -- Inequality constraints
    let p = n + 2
    let h (x: [n]t): [p]t =
      let expected_esg = linalg.dotprod esg x
      in ([
          -- expected_esg >= target_esg
          target_esg T.- expected_esg
          ,
          -- expected_return >= target_return
          target_return T.- (expected_return x mus)
          ]
           ++
           -- w >= 0
           map (T.neg) x)
         :> [p]t
    -- Equality constraints
    -- sum(w) = 1
    let A = [replicate n (T.f64 1.0)]
    let b = [(T.f64 1.0)]
    -- Objective function
    let f0 (x: [n]t) = expected_risk x covs
    -- Solve the problem
    in mod.solve_qp_auto h f0 A b

  def efficient_return_and_esg_fast [n] (target_return: t) (target_esg: t) (mus: [n]t) (esg: [n]t) (covs: [n][n]t) =
    -- Inequality constraints
    let h (x: [n]t): [n]t =
      let expected_esg = linalg.dotprod esg x
      in ([
          -- expected_esg >= target_esg
          target_esg T.- expected_esg
          ,
          -- expected_return >= target_return
          target_return T.- (expected_return x mus)
          ]
           ++
           -- w >= 0
           map (T.neg) x)
         :> [n]t
    let A = [mus, esg, replicate n (T.f64 1.0)]
    let b = [target_return, target_esg, (T.f64 1.0)]
    let x0 = linalg.ols A b
    -- Equality constraints
    -- sum(w) = 1
    let A = [replicate n (T.f64 1.0)]
    let b = [(T.f64 1.0)]
    -- Objective function
    let f0 (x: [n]t) = expected_risk x covs
    -- Solve the problem
    in mod.solve_qp_auto h f0 A b

  def max_sharpe_ratio [n] (risk_free_rate: t) (mus: [n]t) (covs: [n][n]t) =
    -- Inequality constraints
    let h (x: [n]t): [n]t = map (T.neg) x
    -- Equality constraints
    -- sum(w) = 1
    let A = [replicate n (T.f64 1.0)]
    let b = [(T.f64 1.0)]
    -- Objective function
    let f0 (x: [n]t) = T.neg ((expected_return x mus) T.- risk_free_rate) T./ (expected_risk x covs)
    -- Solve the problem
    in mod.solve_qp_auto h f0 A b

  def min_volatility [n] (covs: [n][n]t) =
    -- Inequality constraints
    let h (x: []t) = map (T.neg) x
    -- Equality constraints
    -- sum(w) = 1
    let A = [replicate n (T.f64 1.0)]
    let b = [(T.f64 1.0)]
    -- Objective function
    let f0 (x: [n]t) = expected_risk x covs
    -- Solve the problem
    in mod.solve_qp_auto h f0 A b
}

-- def efficient_return_and_esg_fast [n] (target_return: f64) (target_esg: f64) (Sigma: [n][n]f64) (mu: [n]f64) (esg_scores: [n]f64) =
--     let Sigma = linalg.matmul Sigma (transpose Sigma)
--     let mm = n + 2
--     let f x = linalg.dotprod x (linalg.matvecmul_row Sigma x)
--     let g x = map (*2) (linalg.matvecmul_row Sigma x)
--     let h _ = map (map (*2)) Sigma
--     let x0 = map (/(f64.i64 n)) (replicate n 1)
--     let A = [replicate n 1]
--     let b = [1]
--     let fi x =
--         let expected_return = linalg.dotprod x mu
--         let expected_esg = linalg.dotprod x esg_scores
--         in ([
--             target_return - expected_return,
--             target_esg - expected_esg
--         ] ++ map (f64.neg) x) :> [mm]f64
--     let fi_grad _ =
--         let grad = linalg.matzeros mm n
--         let grad[0, :] = map (f64.neg) mu
--         let grad[1, :] = map (f64.neg) esg_scores
--         let grad[2:, :] = map (map (f64.neg)) (linalg.eye n)
--         in grad
--     let phi x = -(reduce (+) (0.0)  (map (f64.log <-< f64.neg) (fi x)))
--     let phi_grad x =
--         let grad_log_fi = map (-1 /) (fi x)
--         in linalg.matvecmul_row (transpose (fi_grad x)) grad_log_fi
--     let phi_hess x =
--         let fi_values = fi x
--         let grad_fi_values = fi_grad x
--         in map (\i ->
--             map (\j ->
--                 let a = map (\x -> (1 / x ** 2)) fi_values
--                 let b = grad_fi_values[:, i]
--                 let c = grad_fi_values[:, j]
--                 in map3 (\x y z -> x * y * z) a b c |> reduce (+) 0.0
--             ) (iota n)
--         ) (iota n)
--     in mod.barrier_method2 f g h phi phi_grad phi_hess mm A b x0 1 2 1e-10
