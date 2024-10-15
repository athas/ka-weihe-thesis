import "portfolioopt"
import "lib/github.com/diku-dk/linalg/linalg"

module popt = mk_portfolioopt f64
module linalg = mk_linalg f64

let range a b n = 
    let step = (b-a)/f64.i64 n
    in map (\i -> a + step * f64.i64 i) (iota n)

def expected_risk [n] (x: [n]f64) (covs: [n][n]f64) =
        (linalg.matmul (linalg.matmul [x] covs) (transpose [x]))[0,0]

let main = 
    let mu = [0.073, -0.019, -0.02, -0.041, 0.039, -0.082, 0.068, -0.028, 0.013, -0.01]
    let S = [[0.009, -0.001, -0.0, 0.0, 0.001, 0.001, 0.001, -0.001, -0.0, -0.0], [-0.001, 0.009, 0.001, -0.001, 0.0, -0.0, -0.001, -0.0, -0.0, -0.001], [-0.0, 0.001, 0.011, 0.0, 0.0, 0.0, 0.001, 0.001, -0.0, -0.001], [0.0, -0.001, 0.0, 0.011, -0.002, -0.001, -0.001, -0.0, -0.0, 0.001], [0.001, 0.0, 0.0, -0.002, 0.011, 0.0, 0.001, 0.001, -0.001, -0.0], [0.001, -0.0, 0.0, -0.001, 0.0, 0.009, 0.0, -0.001, -0.0, 0.001], [0.001, -0.001, 0.001, -0.001, 0.001, 0.0, 0.011, -0.0, 0.001, -0.0], [-0.001, -0.0, 0.001, -0.0, 0.001, -0.001, -0.0, 0.01, 0.001, 0.001], [-0.0, -0.0, -0.0, -0.0, -0.001, -0.0, 0.001, 0.001, 0.01, -0.001], [-0.0, -0.001, -0.001, 0.001, -0.0, 0.001, -0.0, 0.001, -0.001, 0.01]]
    let esg_scores = [0.9, 0.548, 0.906, 0.552, 0.758, 0.578, 0.639, 0.941, 0.816, 0.982]
    let n = 100
    let range1 = (range 0.0 0.072 n)
    let range2 = (range 0.8 0.90 n)
    let res = map (\e -> map (\r -> popt.efficient_return_and_esg r e mu esg_scores S) range1) range2
    let return = map (\_ -> range1) (iota n) |> flatten
    let risk = map (map (\x -> expected_risk x S)) res |> flatten
    let esg = map (\i -> replicate n range2[i]) (iota n) |> flatten
    in (return, risk, esg) 

    -- in ((range 0.0 0.072 20), (range 0.0 0.90 20))