import "portfolio"
import "lib/github.com/diku-dk/linalg/linalg"
import "convex"

module linalg = mk_linalg f64
module popt = mk_portfolio f64
module convex = mk_solver f64

let all_close [n] (xs: [n]f64) (ys: [n]f64) (eps: f64): bool =
  let diffs = map2 (-) xs ys
  in reduce (&&) true (map (\x -> f64.abs x < eps) diffs)

entry test_esg_and_return = 
    let mu = [0.5488135, 0.71518937, 0.60276338, 0.54488318, 0.4236548]
    let Sigma = [
        [2.47959417, 2.16857618, 1.89111097, 2.19083846, 1.98036542],
        [2.16857618, 2.09100825, 1.33470254, 2.19046532, 1.63155136],
        [1.89111097, 1.33470254, 2.06370517, 1.19593515, 1.61204505],
        [2.19083846, 2.19046532, 1.19593515, 2.43252611, 1.63310741],
        [1.98036542, 1.63155136, 1.61204505, 1.63310741, 1.76671864]]
    let esg_scores = [0.7, 0.8, 0.9, 0.6, 0.5]
    let expected = [-1.63338970e-23, 1.48968254e-01,  3.55360555e-01, 1.40226847e-01, 3.55444343e-01]
    let actual = popt.efficient_return_and_esg 0.5 0.65 mu esg_scores Sigma
    in all_close actual expected 1e-6

-- ==
-- entry: test_esg_and_return
-- input {}
-- output {true}

let range a b n = 
    let step = (b-a)/f64.i64 n
    in map (\i -> a + step * f64.i64 i) (iota n)

let has_nan [n] (xs: [n]f64): bool =
  reduce (||) false (map f64.isnan xs)

entry batched_solve esg_scores mu Sigma =
    let n = 20
    let mu = mu[:n]
    let Sigma = Sigma[:n, :n]
    let esg_scores = esg_scores[:n]

    -- Normalize
    let mu_avg = (reduce (+) 0.0 mu) / (f64.i64 n)
    let esg_avg = (reduce (+) 0.0 esg_scores) / (f64.i64 n)
    let mu = map (/mu_avg) mu
    let esg_scores = map (/esg_avg) esg_scores

    let esg_min = reduce f64.min f64.inf esg_scores
    let esg_max = reduce f64.max 0 esg_scores 
    let mu_min = reduce f64.min f64.inf mu
    let mu_max = reduce f64.max 0 mu
    let range_esg = range esg_min esg_max 200
    let range_mu = range mu_min mu_max 200

    -- let res =  popt.efficient_return_and_esg_fast (mu_max-0.1) (esg_min) mu esg_scores Sigma
    -- in res
    -- Remember to use seq lu
    let res = map (\r -> map (\e -> popt.efficient_return_and_esg r e mu esg_scores Sigma) range_esg) range_mu 
    in flatten res |> filter (\x -> not (has_nan x)) 
