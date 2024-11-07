import "lib/github.com/diku-dk/linalg/linalg"
import "lib/github.com/diku-dk/cpprandom/random"
import "lib/github.com/diku-dk/linalg/lu"
import "convex"

local module type portfolio = {
  type t
  val efficient_return [n] : t -> [n]t -> [n][n]t -> [n]t
  val efficient_risk [n] : t -> [n]t -> [n][n]t -> [n]t
  val efficient_return_and_esg [n] : t -> t -> [n]t -> [n]t -> [n][n]t -> [n]t
  val min_volatility [n] : [n][n]t -> [n]t
  val efficient_return_and_esg_fast [n] : t -> t -> [n]t -> [n]t -> [n][n]t -> [n]t
}

module mk_portfolio (T: real) : portfolio with t = T.t = {
  type t = T.t

  module rng_engine = minstd_rand
  module rand_T = uniform_real_distribution T rng_engine
  module linalg = mk_linalg T
  module lu = mk_lu T
  module convex = mk_solver T

  def expected_return [n] (x: [n]t) (mus: [n]t) =
    linalg.dotprod x mus

  def expected_risk [n] (x: [n]t) (covs: [n][n]t) =
    (linalg.matmul (linalg.matmul [x] covs) (transpose [x]))[0, 0]

  def efficient_return [n] (target_return: t) (mus: [n]t) (covs: [n][n]t) =
    -- Inequality constraints
    let p = n + 1
    let fi (x: [n]t): [p]t =
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
    in convex.solve_qp_auto fi f0 A b

  def efficient_risk [n] (target_risk: t) (mus: [n]t) (covs: [n][n]t) =
    -- Inequality constraints
    let p = n + 1
    let fi (x: [n]t): [p]t =
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
    in convex.solve_qp_auto fi f0 A b

  def efficient_return_and_esg [n] (target_return: t) (target_esg: t) (mus: [n]t) (esg: [n]t) (covs: [n][n]t) =
    -- Inequality constraints
    let p = n + 2
    let fi (x: [n]t): [p]t =
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
    in convex.solve_qp_auto fi f0 A b

  def efficient_return_and_esg_fast [n] (target_return: t) (target_esg: t) (mus: [n]t) (esg: [n]t) (covs: [n][n]t) =
    let f x = linalg.dotprod x (linalg.matvecmul_row covs x)
    let g x = map (T.* (T.f64 2.0)) (linalg.matvecmul_row covs x)
    let h _ = map (map (T.* (T.f64 2.0))) covs
    let mm = n + 2
    let fi x =
      let expected_return = linalg.dotprod x mus
      let expected_esg = linalg.dotprod x esg
      in ([ target_return T.- expected_return
          , target_esg T.- expected_esg
          ]
           ++ map (T.neg) x)
         :> [mm]t
    let fi_grad _ =
      let grad = linalg.matzeros mm n
      let grad[0, :] = map (T.neg) mus
      let grad[1, :] = map (T.neg) esg
      let grad[2:, :] = map (map (T.neg)) (linalg.eye n)
      in grad
    let phi x = T.neg (reduce (T.+) (T.f64 0.0) (map (T.log <-< T.neg) (fi x)))
    let phi_grad x =
      let grad_log_fi = map ((T.i64 (-1)) T./) (fi x)
      in linalg.matvecmul_row (transpose (fi_grad x)) grad_log_fi
    let phi_hess x =
      let fi_values = fi x
      let grad_fi_values = fi_grad x
      in map (\i ->
                 map (\j ->
                         let a = map (\x -> ((T.f64 1) T./ x T.** (T.f64 2))) fi_values
                         let b = grad_fi_values[:, i]
                         let c = grad_fi_values[:, j]
                         in map3 (\x y z -> x T.* y T.* z) a b c |> reduce (T.+) (T.f64 0.0))
                     (iota n))
             (iota n)
    -- Equality constraints
    -- sum(w) = 1
    let A = [replicate n (T.f64 1.0)]
    let b = [(T.f64 1.0)]
    -- Solve the problem
    in convex.solve_qp f g h fi phi phi_grad phi_hess A b

  def min_volatility [n] (covs: [n][n]t) =
    -- Inequality constraints
    let fi (x: []t) = map (T.neg) x
    -- Equality constraints
    -- sum(w) = 1
    let A = [replicate n (T.f64 1.0)]
    let b = [(T.f64 1.0)]
    -- Objective function
    let f0 (x: [n]t) = expected_risk x covs
    -- Solve the problem
    in convex.solve_qp_auto fi f0 A b
}
