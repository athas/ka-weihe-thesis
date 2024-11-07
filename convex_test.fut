import "convex"
import "lib/github.com/diku-dk/linalg/linalg"
import "linear"

module convex = mk_solver f64
module linalg = mk_linalg f64
module linear = mk_linear f64

def all_close [n] (xs: [n]f64) (ys: [n]f64) (eps: f64) : bool =
  let diffs = map2 (-) xs ys
  in reduce (&&) true (map (\x -> f64.abs x < eps) diffs)

entry test_newtons =
  let mu = [0.417022, 0.72032449, 0.00011437, 0.30233257, 0.14675589]
  let Sigma =
    [ [0.6103794, 0.60015296, 0.49513994, 1.10949956, 0.65248394]
    , [0.60015296, 1.45889084, 0.80992882, 1.69516247, 0.61445426]
    , [0.49513994, 0.80992882, 0.99475700, 1.38684737, 0.85491688]
    , [1.10949956, 1.69516247, 1.38684737, 2.92432561, 1.69810919]
    , [0.65248394, 0.61445426, 0.85491688, 1.69810919, 1.60905550]
    ]
  let lambda = 0.5
  -- λμ^T * x - x^T * Σ * x
  let f x = lambda * linalg.dotprod mu x - linalg.dotprod x (linalg.matvecmul_row Sigma x)
  -- 2 * Σ * x + λ * μ
  let g x = map (* 2.0) (linalg.matvecmul_row Sigma x) |> map2 (*) x |> map2 (+) (map (* lambda) mu)
  -- 2 * Σ
  let h _ = map (map (* 2.0)) Sigma
  let x0 = [0.0, 0.0, 0.0, 0.0, 0.0]
  let expected = [0.50115136, 0.60573405, -0.12259852, -0.66124337, 0.35124636]
  let actual = convex.newtons_method g h x0 1e-6 100 |> map (f64.neg)
  in all_close actual expected 1e-6

-- ==
-- entry: test_newtons
-- input {}
-- output {true}

entry test_newton_equality =
  let mu = [0.5488135, 0.71518937, 0.60276338, 0.54488318, 0.4236548]
  let Sigma =
    [ [2.47959417, 2.16857618, 1.89111097, 2.19083846, 1.98036542]
    , [2.16857618, 2.09100825, 1.33470254, 2.19046532, 1.63155136]
    , [1.89111097, 1.33470254, 2.06370517, 1.19593515, 1.61204505]
    , [2.19083846, 2.19046532, 1.19593515, 2.43252611, 1.63310741]
    , [1.98036542, 1.63155136, 1.61204505, 1.63310741, 1.76671864]
    ]
  let f x = linalg.dotprod x (linalg.matvecmul_row Sigma x)
  let g x = map (* 2) (linalg.matvecmul_row Sigma x)
  let h _ = map (map (* 2)) Sigma
  let x0 = [0.2, 0.2, 0.2, 0.2, 0.2]
  let A = [[1, 1, 1, 1, 1]]
  let b = [1]
  let expected = [-4.50382344, 3.03231336, 1.4483085, -0.00804062, 1.0312422]
  let actual = convex.newton_equality g h A b x0 1e-6 1
  in all_close actual expected 1e-6

-- ==
-- entry: test_newton_equality
-- input {}
-- output {true}

entry test_newton_equality_auto_jvp =
  let mu = [0.5488135, 0.71518937, 0.60276338, 0.54488318, 0.4236548]
  let Sigma =
    [ [2.47959417, 2.16857618, 1.89111097, 2.19083846, 1.98036542]
    , [2.16857618, 2.09100825, 1.33470254, 2.19046532, 1.63155136]
    , [1.89111097, 1.33470254, 2.06370517, 1.19593515, 1.61204505]
    , [2.19083846, 2.19046532, 1.19593515, 2.43252611, 1.63310741]
    , [1.98036542, 1.63155136, 1.61204505, 1.63310741, 1.76671864]
    ]
  let f x = linalg.dotprod x (linalg.matvecmul_row Sigma x)
  let g = convex.grad_jvp f
  let h = convex.hess_jvp f
  let x0 = [0.2, 0.2, 0.2, 0.2, 0.2]
  let A = [[1, 1, 1, 1, 1]]
  let b = [1]
  let expected = [-4.50382344, 3.03231336, 1.4483085, -0.00804062, 1.0312422]
  let actual = convex.newton_equality g h A b x0 1e-6 1
  in all_close actual expected 1e-6

-- ==
-- entry: test_newton_equality_auto_jvp
-- input {}
-- output {true}

entry test_newton_equality_auto_vjp =
  let mu = [0.5488135, 0.71518937, 0.60276338, 0.54488318, 0.4236548]
  let Sigma =
    [ [2.47959417, 2.16857618, 1.89111097, 2.19083846, 1.98036542]
    , [2.16857618, 2.09100825, 1.33470254, 2.19046532, 1.63155136]
    , [1.89111097, 1.33470254, 2.06370517, 1.19593515, 1.61204505]
    , [2.19083846, 2.19046532, 1.19593515, 2.43252611, 1.63310741]
    , [1.98036542, 1.63155136, 1.61204505, 1.63310741, 1.76671864]
    ]
  let f x = linalg.dotprod x (linalg.matvecmul_row Sigma x)
  let g = convex.grad_vjp f
  let h = convex.hess_vjp f
  let x0 = [0.2, 0.2, 0.2, 0.2, 0.2]
  let A = [[1, 1, 1, 1, 1]]
  let b = [1]
  let expected = [-4.50382344, 3.03231336, 1.4483085, -0.00804062, 1.0312422]
  let actual = convex.newton_equality g h A b x0 1e-6 1
  in all_close actual expected 1e-6

-- ==
-- entry: test_newton_equality_auto_vjp
-- input {}
-- output {true}

entry barrier_test =
  let n = 5
  let mm = n + 2
  let mu = [0.5488135, 0.71518937, 0.60276338, 0.54488318, 0.4236548]
  let Sigma =
    [ [2.47959417, 2.16857618, 1.89111097, 2.19083846, 1.98036542]
    , [2.16857618, 2.09100825, 1.33470254, 2.19046532, 1.63155136]
    , [1.89111097, 1.33470254, 2.06370517, 1.19593515, 1.61204505]
    , [2.19083846, 2.19046532, 1.19593515, 2.43252611, 1.63310741]
    , [1.98036542, 1.63155136, 1.61204505, 1.63310741, 1.76671864]
    ]
  let esg_scores = [0.7, 0.8, 0.9, 0.6, 0.5]
  let f x = linalg.dotprod x (linalg.matvecmul_row Sigma x)
  let g x = map (* 2) (linalg.matvecmul_row Sigma x)
  let h _ = map (map (* 2)) Sigma
  let x0 = [0.2, 0.2, 0.2, 0.2, 0.2]
  let A = [[1, 1, 1, 1, 1]]
  let b = [1]
  let fi x =
    let expected_return = linalg.dotprod x mu
    let expected_esg = linalg.dotprod x esg_scores
    in ([ 0.5 - expected_return
        , 0.65 - expected_esg
        ]
         ++ map (f64.neg) x)
       :> [7]f64
  let fi_grad _ =
    let grad = linalg.matzeros 7 5
    let grad[0, :] = map (f64.neg) mu
    let grad[1, :] = map (f64.neg) esg_scores
    let grad[2:, :] = map (map (f64.neg)) (linalg.eye 5)
    in grad
  let phi x = -(reduce (+) (0.0) (map (f64.log <-< f64.neg) (fi x)))
  let phi_grad x =
    let grad_log_fi = map (-1 /) (fi x)
    in linalg.matvecmul_row (transpose (fi_grad x)) grad_log_fi
  let phi_hess x =
    let fi_values = fi x
    let grad_fi_values = fi_grad x
    in map (\i ->
               map (\j ->
                       let a = map (\x -> (1 / x ** 2)) fi_values
                       let b = grad_fi_values[:, i]
                       let c = grad_fi_values[:, j]
                       in map3 (\x y z -> x * y * z) a b c |> reduce (+) 0.0)
                   (iota 5))
           (iota 5)
  let expected = [-1.63338970e-23, 1.48968254e-01, 3.55360555e-01, 1.40226847e-01, 3.55444343e-01]
  let actual = convex.barrier_method f g h phi phi_grad phi_hess mm A b x0 1 2 1e-10
  in all_close actual expected 1e-6

-- ==
-- entry: barrier_test
-- input {}
-- output {true}

entry admm_test =
  let mu = [0.5488135, 0.71518937, 0.60276338, 0.54488318, 0.4236548]
  let Sigma =
    [ [2.47959417, 2.16857618, 1.89111097, 2.19083846, 1.98036542]
    , [2.16857618, 2.09100825, 1.33470254, 2.19046532, 1.63155136]
    , [1.89111097, 1.33470254, 2.06370517, 1.19593515, 1.61204505]
    , [2.19083846, 2.19046532, 1.19593515, 2.43252611, 1.63310741]
    , [1.98036542, 1.63155136, 1.61204505, 1.63310741, 1.76671864]
    ]
  let esg_scores = [0.7, 0.8, 0.9, 0.6, 0.5]
  let ones = replicate 5 1.0
  let neg_ones = map (f64.neg) ones
  let A = [mu, esg_scores, ones, neg_ones] ++ linalg.eye 5 :> [9][5]f64
  let b = [0.5, 0.65, 1, -1] ++ replicate 5 0.0 :> [9]f64
  let x_init = replicate 5 0.0
  let x_init[0] = 1.0
  let rho = 0.15
  let tol = 1e-10
  let max_iter = 300
  let g x = map (* 2) (linalg.matvecmul_row Sigma x)
  let h _ = map (map (* 2)) Sigma
  let actual = convex.admm g h A b x_init rho tol max_iter
  let expected = [-1.63338970e-23, 1.48968254e-01, 3.55360555e-01, 1.40226847e-01, 3.55444343e-01]
  in all_close actual expected 1e-4

-- ==
-- entry: admm_test
-- input {}
-- output {true}
