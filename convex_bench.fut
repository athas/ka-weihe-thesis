import "convex"
import "lib/github.com/diku-dk/linalg/linalg"
import "linear"

module convex = mk_solver f64
module linalg = mk_linalg f64
module linear = mk_linear f64

entry bench_newtons [n] (Sigma: [n][n]f64) (mu: [n]f64) =
  let Sigma = linalg.matmul Sigma (transpose Sigma)
  let lambda = 0.5
  let g x = map (* 2.0) (linalg.matvecmul_row Sigma x) |> map2 (*) x |> map2 (+) (map (* lambda) mu)
  let h _ = map (map (* 2.0)) Sigma
  let x0 = replicate n 0.0
  in convex.newtons_method g h x0 1e-8 1000

-- == 
-- entry: bench_newtons
-- input @ data/2d_100x100_1d_100f64
-- input @ data/2d_500x500_1d_500f64
-- input @ data/2d_1000x1000_1d_1000f64
-- input @ data/2d_5000x5000_1d_5000f64
-- input @ data/2d_10000x10000_1d_10000f64

entry bench_newtons_ls [n] (Sigma: [n][n]f64) (mu: [n]f64) =
  let Sigma = linalg.matmul Sigma (transpose Sigma)
  let lambda = 0.5
  let g x = map (* 2.0) (linalg.matvecmul_row Sigma x) |> map2 (*) x |> map2 (+) (map (* lambda) mu)
  let h _ = map (map (* 2.0)) Sigma
  let x0 = replicate n 0.0
  let f x = lambda * linalg.dotprod mu x - linalg.dotprod x (linalg.matvecmul_row Sigma x)
  in convex.newtons_method_ls f g h x0 1e-8 0.3 0.5 1000 10

-- == 
-- entry: bench_newtons_ls
-- input @ data/2d_100x100_1d_100f64
-- input @ data/2d_500x500_1d_500f64
-- input @ data/2d_1000x1000_1d_1000f64
-- input @ data/2d_5000x5000_1d_5000f64
-- input @ data/2d_10000x10000_1d_10000f64

-- entry bench_newtons3 [n] (Sigma: [n][n]f64) (mu: [n]f64) =
--     let Sigma = linalg.matmul Sigma (transpose Sigma)
--     let lambda = 0.5
--     let g x = map (* 2.0) (linalg.matvecmul_row Sigma x) |> map2 (*) x |> map2 (+) (map (* lambda) mu) 
--     let h _ = map (map (* 2.0)) Sigma
--     let x0 =  replicate n 0.0
--     let f x = lambda * linalg.dotprod mu x - linalg.dotprod x (linalg.matvecmul_row Sigma x)
--     in convex.newtons_method3 f g h x0 1e-8 0.3 0.5 1000 10

-- == 
-- entry: bench_newtons3
-- input @ data/2d_100x100_1d_100f64
-- input @ data/2d_500x500_1d_500f64
-- input @ data/2d_1000x1000_1d_1000f64
-- input @ data/2d_5000x5000_1d_5000f64
-- input @ data/2d_10000x10000_1d_10000f64

-- ######################
-- WITH LU
-- ######################
-- convex_bench.fut:bench_newtons (no tuning file):
-- data/2d_100x100_1d_100f64:             4238μs (95% CI: [    4228.2,     4251.6])
-- data/2d_500x500_1d_500f64:            12118μs (95% CI: [   12113.8,    12125.9])
-- data/2d_1000x1000_1d_1000f64:         26513μs (95% CI: [   26493.5,    26533.7])
-- data/2d_5000x5000_1d_5000f64:        398021μs (95% CI: [  397838.9,   398252.6])
-- data/2d_10000x10000_1d_10000f64:    3339618μs (95% CI: [ 3338158.4,  3340866.4])

-- convex_bench.fut:bench_newtons2 (no tuning file):
-- data/2d_100x100_1d_100f64:             4382μs (95% CI: [    4379.6,     4384.2])
-- data/2d_500x500_1d_500f64:            11984μs (95% CI: [   11972.1,    11995.3])
-- data/2d_1000x1000_1d_1000f64:         26384μs (95% CI: [   26330.9,    26446.6])
-- data/2d_5000x5000_1d_5000f64:        397840μs (95% CI: [  397685.8,   397984.8])
-- data/2d_10000x10000_1d_10000f64:    3337620μs (95% CI: [ 3336169.3,  3338892.1])

-- convex_bench.fut:bench_newtons3 (no tuning file):
-- data/2d_100x100_1d_100f64:             4331μs (95% CI: [    4316.4,     4364.9])
-- data/2d_500x500_1d_500f64:            12324μs (95% CI: [   12312.5,    12337.0])
-- data/2d_1000x1000_1d_1000f64:         27367μs (95% CI: [   27334.9,    27399.1])
-- data/2d_5000x5000_1d_5000f64:        401075μs (95% CI: [  400858.8,   401248.3])
-- data/2d_10000x10000_1d_10000f64:    3347706μs (95% CI: [ 3346865.3,  3348729.9])

-- ######################
-- WITH CHOLESKY
-- ######################
-- convex_bench.fut:bench_newtons (no tuning file):
-- data/2d_100x100_1d_100f64:             4395μs (95% CI: [    4387.0,     4408.1])
-- data/2d_500x500_1d_500f64:            22026μs (95% CI: [   21797.8,    22540.0])
-- data/2d_1000x1000_1d_1000f64:         43337μs (95% CI: [   43308.2,    43378.8])
-- data/2d_5000x5000_1d_5000f64:        444724μs (95% CI: [  444489.0,   444900.4])
-- data/2d_10000x10000_1d_10000f64:    2280493μs (95% CI: [ 2279410.2,  2281407.7])

-- ######################
-- WITH Gaussian elimination no pivoting
-- ######################
-- convex_bench.fut:bench_newtons (no tuning file):
-- data/2d_100x100_1d_100f64:             1238μs (95% CI: [    1237.4,     1239.3])
-- data/2d_500x500_1d_500f64:             5335μs (95% CI: [    5333.7,     5336.6])
-- data/2d_1000x1000_1d_1000f64:         15091μs (95% CI: [   15086.1,    15102.8])
-- data/2d_5000x5000_1d_5000f64:       1707163μs (95% CI: [ 1615275.8,  1858928.5])
-- data/2d_10000x10000_1d_10000f64:   12538721μs (95% CI: [  12537921,   12539521])

entry bench_newtons_batched [n] (Sigmas: [][n][n]f64) (mus: [][n]f64) =
  let Sigmas = map (\Sigma -> linalg.matmul Sigma (transpose Sigma)) Sigmas
  let lambda = 0.5
  in map2 (\Sigma mu ->
              let g x = map (* 2.0) (linalg.matvecmul_row Sigma x) |> map2 (*) x |> map2 (+) (map (* lambda) mu)
              let h _ = map (map (* 2.0)) Sigma
              let x0 = replicate n 0.0
              in convex.newtons_method g h x0 1e-8 1000)
          Sigmas
          mus

-- == 
-- entry: bench_newtons_batched
-- input @ data/2d_1000000x5x5_1d_1000000x5f64
-- input @ data/2d_1000000x10x10_1d_1000000x10f64
-- input @ data/2d_100000x20x20_1d_100000x20f64
-- input @ data/2d_10000x50x50_1d_10000x50f64
-- input @ data/2d_1000x100x100_1d_1000x100f64
-- input @ data/2d_100x500x500f64_1d_100x500f64

entry bench_newtons2_batched [n] (Sigmas: [][n][n]f64) (mus: [][n]f64) =
  let Sigmas = map (\Sigma -> linalg.matmul Sigma (transpose Sigma)) Sigmas
  let lambda = 0.5
  in map2 (\Sigma mu ->
              let f x = lambda * linalg.dotprod mu x - linalg.dotprod x (linalg.matvecmul_row Sigma x)
              let g x = map (* 2.0) (linalg.matvecmul_row Sigma x) |> map2 (*) x |> map2 (+) (map (* lambda) mu)
              let h _ = map (map (* 2.0)) Sigma
              let x0 = replicate n 0.0
              in convex.newtons_method_ls f g h x0 1e-8 0.3 0.5 1000 10)
          Sigmas
          mus

-- == 
-- entry: bench_newtons2_batched
-- input @ data/2d_1000000x5x5_1d_1000000x5f64
-- input @ data/2d_1000000x10x10_1d_1000000x10f64
-- input @ data/2d_100000x20x20_1d_100000x20f64
-- input @ data/2d_10000x50x50_1d_10000x50f64
-- input @ data/2d_1000x100x100_1d_1000x100f64
-- input @ data/2d_100x500x500f64_1d_100x500f64

-- entry bench_newtons3_batched [n] (Sigmas: [][n][n]f64) (mus: [][n]f64) =
--     let Sigmas = map (\Sigma -> linalg.matmul Sigma (transpose Sigma)) Sigmas
--     let lambda = 0.5
--     in map2 (\Sigma mu -> 
--         let f x = lambda * linalg.dotprod mu x - linalg.dotprod x (linalg.matvecmul_row Sigma x)
--         let g x = map (* 2.0) (linalg.matvecmul_row Sigma x) |> map2 (*) x |> map2 (+) (map (* lambda) mu) 
--         let h _ = map (map (* 2.0)) Sigma
--         let x0 =  replicate n 0.0
--         in convex.newtons_method3 f g h x0 1e-8 0.3 0.5 1000 10
--     ) Sigmas mus

-- == 
-- entry: bench_newtons3_batched
-- input @ data/2d_1000000x5x5_1d_1000000x5f64
-- input @ data/2d_1000000x10x10_1d_1000000x10f64
-- input @ data/2d_100000x20x20_1d_100000x20f64
-- input @ data/2d_10000x50x50_1d_10000x50f64
-- input @ data/2d_1000x100x100_1d_1000x100f64
-- input @ data/2d_100x500x500f64_1d_100x500f64

-- Seq Cho
-- convex_bench.fut:bench_newtons_batched (no tuning file):
-- data/2d_1000000x5x5_1d_1000000x5f64:          1840μs (95% CI: [    1837.2,     1850.2])
-- data/2d_1000000x10x10_1d_1000000x10f64:      26570μs (95% CI: [   23643.2,    30101.5])
-- data/2d_100000x20x20_1d_100000x20f64:        21636μs (95% CI: [   17715.3,    24839.2])
-- data/2d_10000x50x50_1d_10000x50f64:          17048μs (95% CI: [   16676.4,    18503.5])
-- data/2d_1000x100x100_1d_1000x100f64:         38656μs (95% CI: [   38634.8,    38680.3])
-- data/2d_100x500x500f64_1d_100x500f64:      6820933

-- Seq Cho LS
-- convex_bench.fut:bench_newtons2_batched (no tuning file):
-- data/2d_1000000x5x5_1d_1000000x5f64:          2277μs (95% CI: [    2274.0,     2279.9])
-- data/2d_1000000x10x10_1d_1000000x10f64:      25879μs (95% CI: [   23287.7,    28554.9])
-- data/2d_100000x20x20_1d_100000x20f64:        15691μs (95% CI: [   13651.5,    18386.9])
-- data/2d_10000x50x50_1d_10000x50f64:          16761μs (95% CI: [   16616.2,    17047.3])
-- data/2d_1000x100x100_1d_1000x100f64:         38950μs (95% CI: [   38918.5,    38984.0])
-- data/2d_100x500x500f64_1d_100x500f64:      5603183μs (95% CI: [ 5399281.9,  5801422.9])

-- Seq Cho Par-LS
-- convex_bench.fut:bench_newtons3_batched (no tuning file):
-- data/2d_1000000x5x5_1d_1000000x5f64:          4575μs (95% CI: [    4572.0,     4577.8])
-- data/2d_1000000x10x10_1d_1000000x10f64:      41468μs (95% CI: [   37011.5,    46529.4])
-- data/2d_100000x20x20_1d_100000x20f64:        18845μs (95% CI: [   18543.2,    19184.1])
-- data/2d_10000x50x50_1d_10000x50f64:          24962μs (95% CI: [   24878.8,    25249.6])
-- data/2d_1000x100x100_1d_1000x100f64:        126396μs (95% CI: [  121694.4,   129341.9])
-- data/2d_100x500x500f64_1d_100x500f64:      7935806μs (95% CI: [ 7755282.9,  8116330.9])

-- Cho
-- convex_bench.fut:bench_newtons_batched (no tuning file):
-- data/2d_1000000x5x5_1d_1000000x5f64:          4010μs (95% CI: [    3839.0,     4201.9])
-- data/2d_1000000x10x10_1d_1000000x10f64:      11145μs (95% CI: [   11130.6,    11159.9])
-- data/2d_100000x20x20_1d_100000x20f64:        18422μs (95% CI: [   16542.2,    20195.0])
-- data/2d_10000x50x50_1d_10000x50f64:          39673μs (95% CI: [   36181.6,    43970.1])
-- data/2d_1000x100x100_1d_1000x100f64:         64009μs (95% CI: [   56616.5,    72414.4])
-- data/2d_100x500x500f64_1d_100x500f64:     13008815μs

-- Gauss NP
-- convex_bench.fut:bench_newtons_batched (no tuning file):
-- data/2d_1000000x5x5_1d_1000000x5f64:          3390μs (95% CI: [    3388.8,     3391.9])
-- data/2d_1000000x10x10_1d_1000000x10f64:      60084μs (95% CI: [   54285.3,    67054.2])
-- data/2d_100000x20x20_1d_100000x20f64:        40249μs (95% CI: [   39398.1,    41172.6])
-- data/2d_10000x50x50_1d_10000x50f64:          86402μs (95% CI: [   86166.8,    86794.7])
-- data/2d_1000x100x100_1d_1000x100f64:        701396μs (95% CI: [  612683.5,   828564.8])
-- data/2d_100x500x500f64_1d_100x500f64:     87000395μs

-- CG
-- convex_bench.fut:bench_newtons_batched (no tuning file):
-- data/2d_1000000x5x5_1d_1000000x5f64:          3318μs (95% CI: [    3315.9,     3319.3])
-- data/2d_1000000x10x10_1d_1000000x10f64:      22163μs (95% CI: [   22152.9,    22170.3])
-- data/2d_100000x20x20_1d_100000x20f64:        16825μs (95% CI: [   16815.0,    16831.3])
-- data/2d_10000x50x50_1d_10000x50f64:          50623μs (95% CI: [   50597.7,    50651.5])
-- data/2d_1000x100x100_1d_1000x100f64:        285931μs (95% CI: [  285873.9,   285980.1])
-- data/2d_100x500x500f64_1d_100x500f64:     30334389μs (95% CI: [30299389.0, 30369389.0])

-- Cvxpy
-- n = 5, elapsed = 1654617786.41 us
-- n = 10, elapsed = 1724611043.93 us
-- n = 20, elapsed = 200374889.37 us
-- n = 50, elapsed = 55150985.72 us
-- n = 100, elapsed = 313039350.51 us
-- n = 500, elapsed = 122469816.21 us

entry bench_newtoneq [n] (Sigma: [n][n]f64) =
  let f x = linalg.dotprod x (linalg.matvecmul_row Sigma x)
  let g x = map (* 2) (linalg.matvecmul_row Sigma x)
  let h _ = map (map (* 2)) Sigma
  let x0 = map (/ (f64.i64 n)) (replicate n 1)
  let A = [replicate n 1]
  let b = [1]
  in convex.newton_equality g h A b x0 1e-8 1000

-- == 
-- entry: bench_newtoneq
-- input @ data/2d_100x100f64
-- input @ data/2d_500x500f64
-- input @ data/2d_1000x1000f64
-- input @ data/2d_5000x5000f64
-- input @ data/2d_10000x10000f64

-- Blocked LU
-- convex_bench.fut:bench_newtoneq (no tuning file):
-- convex_bench.fut:bench_newtoneq (no tuning file):
-- data/2d_100x100_1d_100f64:                   10099μs (95% CI: [   10083.4,    10130.0])
-- data/2d_500x500_1d_500f64:                   23710μs (95% CI: [   23691.0,    23731.6])
-- data/2d_1000x1000_1d_1000f64:                50771μs (95% CI: [   50721.5,    50832.0])
-- data/2d_5000x5000_1d_5000f64:               596213μs (95% CI: [  596096.8,   596365.9])
-- data/2d_10000x10000_1d_10000f64:           7645244μs (95% CI: [ 7644454.0,  7646034.0])

-- Guass NP
-- convex_bench.fut:bench_newtoneq (no tuning file):
-- data/2d_100x100_1d_100f64:                    2502μs (95% CI: [    2498.9,     2505.4])
-- data/2d_500x500_1d_500f64:                   11850μs (95% CI: [   11828.3,    11899.2])
-- data/2d_1000x1000_1d_1000f64:                28136μs (95% CI: [   28093.1,    28288.4])
-- data/2d_5000x5000_1d_5000f64:              3022461μs (95% CI: [ 3021481.2,  3023338.2])
-- data/2d_10000x10000_1d_10000f64:          35249391μs (95% CI: [35240852.4, 35256504.6])

-- Simple LU
-- data/2d_100x100_1d_100f64:                    6916μs (95% CI: [    6905.4,     6925.9])
-- data/2d_500x500_1d_500f64:                   35543μs (95% CI: [   35363.9,    36195.3])
-- data/2d_1000x1000_1d_1000f64:                74481μs (95% CI: [   74453.8,    74507.7])
-- data/2d_5000x5000_1d_5000f64:              3364351μs (95% CI: [ 3363194.8,  3365178.5])
-- data/2d_10000x10000_1d_10000f64:          37224828μs (95% CI: [37219874.0, 37229781.0])

-- CVXPY
-- n = 100, elapsed = 290593.15 us
-- n = 500, elapsed = 3208202.36 us
-- n = 1000, elapsed = 12529201.03 us

entry bench_newtoneq_batched [n] (Sigmas: [][n][n]f64) =
  let Sigmas = map (\Sigma -> linalg.matmul Sigma (transpose Sigma)) Sigmas
  in map (\Sigma ->
             let f x = linalg.dotprod x (linalg.matvecmul_row Sigma x)
             let g x = map (* 2) (linalg.matvecmul_row Sigma x)
             let h _ = map (map (* 2)) Sigma
             let x0 = map (/ (f64.i64 n)) (replicate n 1)
             let A = [replicate n 1]
             let b = [1]
             -- in #[sequential]
             in convex.newton_equality g h A b x0 1e-8 1000)
         Sigmas

-- == 
-- entry: bench_newtoneq_batched
-- input @ data/2d_1000000x5x5f64
-- input @ data/2d_1000000x10x10f64
-- input @ data/2d_100000x20x20f64
-- input @ data/2d_10000x50x50f64
-- input @ data/2d_1000x100x100f64
-- input @ data/2d_100x500x500f64

-- Seq LU
-- convex_bench.fut:bench_newtoneq_batched (no tuning file):
-- data/2d_1000000x5x5f64:                       5874μs (95% CI: [    5866.3,     5891.8])
-- data/2d_1000000x10x10f64:                    29754μs (95% CI: [   29510.9,    30321.0])
-- data/2d_100000x20x20f64:                     17829μs (95% CI: [   17816.4,    17857.8])
-- data/2d_10000x50x50f64:                      46418μs (95% CI: [   46394.1,    46437.7])
-- data/2d_1000x100x100f64:                    211924μs (95% CI: [  211868.4,   212010.0])
-- data/2d_100x500x500f64:                   21089124μs (95% CI: [21087680.0, 21090567.0])

-- Guass NP
-- convex_bench.fut:bench_newtoneq_batched (no tuning file):
-- data/2d_1000000x5x5f64:                      10141μs (95% CI: [   10135.6,    10146.0])
-- data/2d_1000000x10x10f64:                    63800μs (95% CI: [   63752.0,    63887.1])
-- data/2d_100000x20x20f64:                     46294μs (95% CI: [   46282.5,    46303.7])
-- data/2d_10000x50x50f64:                     180192μs (95% CI: [  180125.6,   180264.4])
-- data/2d_1000x100x100f64:                   1113271μs (95% CI: [ 1113083.3,  1113598.6])
-- data/2d_100x500x500f64:                  124667678μs (95% CI: [124659902.0, 124675454.0])

-- CVXPY
-- n = 5, elapsed = 1506617546.08 us
-- n = 10, elapsed = 1538765192.03 us
-- n = 20, elapsed = 172541308.40 us
-- n = 50, elapsed = 52582621.57 us
-- n = 100, elapsed = 277966308.59 us
-- n = 500, elapsed = 112454667.09 us

entry barrier_bench [n] (Sigma: [n][n]f64) (mu: [n]f64) (esg_scores: [n]f64) =
  let Sigma = linalg.matmul Sigma (transpose Sigma)
  let mm = n + 2
  let f x = linalg.dotprod x (linalg.matvecmul_row Sigma x)
  let g x = map (* 2) (linalg.matvecmul_row Sigma x)
  let h _ = map (map (* 2)) Sigma
  let x0 = map (/ (f64.i64 n)) (replicate n 1)
  let A = [replicate n 1]
  let b = [1]
  let fi x =
    let expected_return = linalg.dotprod x mu
    let expected_esg = linalg.dotprod x esg_scores
    in ([ 0.07 - expected_return
        , 0.08 - expected_esg
        ]
         ++ map (f64.neg) x)
       :> [mm]f64
  let fi_grad _ =
    let grad = linalg.matzeros mm n
    let grad[0, :] = map (f64.neg) mu
    let grad[1, :] = map (f64.neg) esg_scores
    let grad[2:, :] = map (map (f64.neg)) (linalg.eye n)
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
                   (iota n))
           (iota n)
  in convex.barrier_method f g h phi phi_grad phi_hess mm A b x0 1 2 1e-10

-- ==
-- entry: barrier_bench
-- input @ data/2x_100x100_1d_100_1d_100f64
-- input @ data/2x_500x500_1d_500_1d_500f64
-- input @ data/2x_1000x1000_1d_1000_1d_1000f64
-- input @ data/2x_5000x5000_1d_5000_1d_5000f64
-- input @ data/2x_10000x10000_1d_10000_1d_10000f64

entry barrier_bench_jvp [n] (Sigma: [n][n]f64) (mu: [n]f64) (esg_scores: [n]f64) =
  let Sigma = linalg.matmul Sigma (transpose Sigma)
  let mm = n + 2
  let f x = linalg.dotprod x (linalg.matvecmul_row Sigma x)
  let g x = convex.grad_jvp f x
  let h x = convex.hess_jvp f x
  let x0 = map (/ (f64.i64 n)) (replicate n 1)
  let A = [replicate n 1]
  let b = [1]
  let fi x =
    let expected_return = linalg.dotprod x mu
    let expected_esg = linalg.dotprod x esg_scores
    in ([ 0.07 - expected_return
        , 0.08 - expected_esg
        ]
         ++ map (f64.neg) x)
       :> [mm]f64
  let phi x = -(reduce (+) (0.0) (map (f64.log <-< f64.neg) (fi x)))
  let phi_grad x = convex.grad_jvp phi x
  let phi_hess x = convex.hess_jvp phi x
  in convex.barrier_method f g h phi phi_grad phi_hess mm A b x0 1 2 1e-10

-- ==
-- entry: barrier_bench_jvp
-- input @ data/2x_100x100_1d_100_1d_100f64
-- input @ data/2x_500x500_1d_500_1d_500f64
-- input @ data/2x_1000x1000_1d_1000_1d_1000f64
-- input @ data/2x_5000x5000_1d_5000_1d_5000f64
-- input @ data/2x_10000x10000_1d_10000_1d_10000f64

entry barrier_bench_vjp [n] (Sigma: [n][n]f64) (mu: [n]f64) (esg_scores: [n]f64) =
  let Sigma = linalg.matmul Sigma (transpose Sigma)
  let mm = n + 2
  let f x = linalg.dotprod x (linalg.matvecmul_row Sigma x)
  let g x = convex.grad_vjp f x
  let h x = convex.hess_vjp f x
  let x0 = map (/ (f64.i64 n)) (replicate n 1)
  let A = [replicate n 1]
  let b = [1]
  let fi x =
    let expected_return = linalg.dotprod x mu
    let expected_esg = linalg.dotprod x esg_scores
    in ([ 0.07 - expected_return
        , 0.08 - expected_esg
        ]
         ++ map (f64.neg) x)
       :> [mm]f64
  let phi x = -(reduce (+) (0.0) (map (f64.log <-< f64.neg) (fi x)))
  let phi_grad x = convex.grad_vjp phi x
  let phi_hess x = convex.hess_vjp phi x
  in convex.barrier_method f g h phi phi_grad phi_hess mm A b x0 1 2 1e-10

-- ==
-- entry: barrier_bench_vjp
-- input @ data/2x_100x100_1d_100_1d_100f64
-- input @ data/2x_500x500_1d_500_1d_500f64
-- input @ data/2x_1000x1000_1d_1000_1d_1000f64
-- input @ data/2x_5000x5000_1d_5000_1d_5000f64
-- input @ data/2x_10000x10000_1d_10000_1d_10000f64

entry admm_bench [n] (Sigma: [n][n]f64) (mu: [n]f64) (esg_scores: [n]f64) =
  let ones = replicate n 1.0
  let neg_ones = map (f64.neg) ones
  let m = n + 4
  let A = [mu, esg_scores, ones, neg_ones] ++ linalg.eye n :> [m][n]f64
  let b = [0.5, 0.65, 1, -1] ++ replicate n 0.0 :> [m]f64
  let x_init = replicate n 0.0
  let x_init[0] = 1.0
  let rho = 0.15
  let tol = 1e-10
  let max_iter = 300
  let g x = map (* 2) (linalg.matvecmul_row Sigma x)
  let h _ = map (map (* 2)) Sigma
  in convex.admm g h A b x_init rho tol max_iter

-- ==
-- entry: admm_bench
-- input @ data/2x_100x100_1d_100_1d_100f64
-- input @ data/2x_500x500_1d_500_1d_500f64
-- input @ data/2x_1000x1000_1d_1000_1d_1000f64
-- input @ data/2x_5000x5000_1d_5000_1d_5000f64
-- input @ data/2x_10000x10000_1d_10000_1d_10000f64

-- Barrier
-- convex_bench.fut:barrier_bench (no tuning file):
-- data/2x_100x100_1d_100_1d_100f64:             118998μs (95% CI: [  115305.0,   129994.0])
-- data/2x_500x500_1d_500_1d_500f64:             281977μs (95% CI: [  281601.7,   282345.3])
-- data/2x_1000x1000_1d_1000_1d_1000f64:         657501μs (95% CI: [  656795.6,   658402.3])
-- data/2x_5000x5000_1d_5000_1d_5000f64:       11423628μs (95% CI: [11409746.8, 11434527.3])
-- data/2x_10000x10000_1d_10000_1d_10000f64:   98800274μs (95% CI: [98759680.0, 98840868.0])

-- Barrier JVP
-- convex_bench.fut:barrier_bench_jvp (no tuning file):
-- data/2x_100x100_1d_100_1d_100f64:             116995μs (95% CI: [  115774.5,   118254.9])
-- data/2x_500x500_1d_500_1d_500f64:           17840563μs (95% CI: [17840238.1, 17840902.3])
-- data/2x_1000x1000_1d_1000_1d_1000f64:     
-- Failed to allocate memory in space 'device'.
-- Attempted allocation:   8001536000 bytes
-- Currently allocated:   40128633672 bytes

-- Barrier VJP
-- convex_bench.fut:barrier_bench_vjp (no tuning file):
-- convex_bench.fut:barrier_bench_vjp (no tuning file):
-- data/2x_100x100_1d_100_1d_100f64:             154389μs (95% CI: [  152342.4,   160494.5])
-- data/2x_500x500_1d_500_1d_500f64:           28815188μs (95% CI: [28812442.1, 28818031.6])

-- ADMM LU
-- convex_bench.fut:admm_bench (no tuning file):
-- data/2x_100x100_1d_100_1d_100f64:            1197242μs (95% CI: [ 1191163.3,  1214905.8])
-- data/2x_500x500_1d_500_1d_500f64:            3704565μs (95% CI: [ 3703122.1,  3705856.1])
-- data/2x_1000x1000_1d_1000_1d_1000f64:        8030992μs (95% CI: [ 8022620.1,  8041895.0])
-- data/2x_5000x5000_1d_5000_1d_5000f64:      124910869μs (95% CI: [124866000.0, 124955738.0])

-- ADMM CG
-- convex_bench.fut:admm_bench (no tuning file):
-- data/2x_100x100_1d_100_1d_100f64:            1767997μs (95% CI: [ 1766280.1,  1772275.8])
-- data/2x_500x500_1d_500_1d_500f64:           10400217μs (95% CI: [10290228.5, 10469881.1])
-- data/2x_1000x1000_1d_1000_1d_1000f64:       19616014μs (95% CI: [19417575.3, 19752321.0])

-- CVXPY
-- n = 100, elapsed = 333689.21 us
-- n = 500, elapsed = 1039505.96 us
-- n = 1000, elapsed = 12988537.07 us

entry barrier_bench_batched [n] (Sigmas: [][n][n]f64) (mus: [][n]f64) (esg_scoress: [][n]f64) =
  let Sigmas = map (\Sigma -> linalg.matmul Sigma (transpose Sigma)) Sigmas
  in map3 (\Sigma mu esg_scores ->
              let Sigma = linalg.matmul Sigma (transpose Sigma)
              let mm = n + 2
              let f x = linalg.dotprod x (linalg.matvecmul_row Sigma x)
              let g x = map (* 2) (linalg.matvecmul_row Sigma x)
              let h _ = map (map (* 2)) Sigma
              let x0 = map (/ (f64.i64 n)) (replicate n 1)
              let A = [replicate n 1]
              let b = [1]
              let fi x =
                let expected_return = linalg.dotprod x mu
                let expected_esg = linalg.dotprod x esg_scores
                in ([ 0.07 - expected_return
                    , 0.08 - expected_esg
                    ]
                     ++ map (f64.neg) x)
                   :> [mm]f64
              let fi_grad _ =
                let grad = linalg.matzeros mm n
                let grad[0, :] = map (f64.neg) mu
                let grad[1, :] = map (f64.neg) esg_scores
                let grad[2:, :] = map (map (f64.neg)) (linalg.eye n)
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
                               (iota n))
                       (iota n)
              in convex.barrier_method f g h phi phi_grad phi_hess mm A b x0 1 4 1e-8)
          Sigmas
          mus
          esg_scoress

-- == 
-- entry: barrier_bench_batched
-- input @ data/2d_1000000x5x5_1d_1000000x5_1d_1000000x5f64
-- input @ data/2d_1000000x10x10_1d_1000000x10_1d_1000000x10f64
-- input @ data/2d_100000x20x20_1d_100000x20_1d_100000x20f64
-- input @ data/2d_10000x50x50_1d_10000x50_1d_10000x50f64
-- input @ data/2d_1000x100x100_1d_1000x100_1d_1000x100f64
-- input @ data/2d_100x500x500f64_1d_100x500_1d_100x500f64

entry barrier_bench_jvp_batched [n] (Sigmas: [][n][n]f64) (mus: [][n]f64) (esg_scoress: [][n]f64) =
  let Sigmas = map (\Sigma -> linalg.matmul Sigma (transpose Sigma)) Sigmas
  in map3 (\Sigma mu esg_scores ->
              let mm = n + 2
              let f x = linalg.dotprod x (linalg.matvecmul_row Sigma x)
              let g x = convex.grad_jvp f x
              let h x = convex.hess_jvp f x
              let x0 = map (/ (f64.i64 n)) (replicate n 1)
              let A = [replicate n 1]
              let b = [1]
              let fi x =
                let expected_return = linalg.dotprod x mu
                let expected_esg = linalg.dotprod x esg_scores
                in ([ 0.07 - expected_return
                    , 0.08 - expected_esg
                    ]
                     ++ map (f64.neg) x)
                   :> [mm]f64
              let phi x = -(reduce (+) (0.0) (map (f64.log <-< f64.neg) (fi x)))
              let phi_grad x = convex.grad_vjp phi x
              let phi_hess x = convex.hess_vjp phi x
              in convex.barrier_method f g h phi phi_grad phi_hess mm A b x0 1 4 1e-8)
          Sigmas
          mus
          esg_scoress

-- == 
-- entry: barrier_bench_jvp_batched
-- input @ data/2d_1000000x5x5_1d_1000000x5_1d_1000000x5f64
-- input @ data/2d_1000000x10x10_1d_1000000x10_1d_1000000x10f64
-- input @ data/2d_100000x20x20_1d_100000x20_1d_100000x20f64
-- input @ data/2d_10000x50x50_1d_10000x50_1d_10000x50f64
-- input @ data/2d_1000x100x100_1d_1000x100_1d_1000x100f64
-- input @ data/2d_100x500x500f64_1d_100x500_1d_100x500f64

entry barrier_bench_vjp_batched [n] (Sigmas: [][n][n]f64) (mus: [][n]f64) (esg_scoress: [][n]f64) =
  let Sigmas = map (\Sigma -> linalg.matmul Sigma (transpose Sigma)) Sigmas
  in map3 (\Sigma mu esg_scores ->
              let mm = n + 2
              let f x = linalg.dotprod x (linalg.matvecmul_row Sigma x)
              let g x = convex.grad_vjp f x
              let h x = convex.hess_vjp f x
              let x0 = map (/ (f64.i64 n)) (replicate n 1)
              let A = [replicate n 1]
              let b = [1]
              let fi x =
                let expected_return = linalg.dotprod x mu
                let expected_esg = linalg.dotprod x esg_scores
                in ([ 0.07 - expected_return
                    , 0.08 - expected_esg
                    ]
                     ++ map (f64.neg) x)
                   :> [mm]f64
              let phi x = -(reduce (+) (0.0) (map (f64.log <-< f64.neg) (fi x)))
              let phi_grad x = convex.grad_vjp phi x
              let phi_hess x = convex.hess_vjp phi x
              in convex.barrier_method f g h phi phi_grad phi_hess mm A b x0 1 4 1e-8)
          Sigmas
          mus
          esg_scoress

-- == 
-- entry: barrier_bench_vjp_batched
-- input @ data/2d_1000000x5x5_1d_1000000x5_1d_1000000x5f64
-- input @ data/2d_1000000x10x10_1d_1000000x10_1d_1000000x10f64
-- input @ data/2d_100000x20x20_1d_100000x20_1d_100000x20f64
-- input @ data/2d_10000x50x50_1d_10000x50_1d_10000x50f64
-- input @ data/2d_1000x100x100_1d_1000x100_1d_1000x100f64
-- input @ data/2d_100x500x500f64_1d_100x500_1d_100x500f64

entry admm_bench_batched [n] (Sigmas: [][n][n]f64) (mus: [][n]f64) (esg_scoress: [][n]f64) =
  let Sigmas = map (\Sigma -> linalg.matmul Sigma (transpose Sigma)) Sigmas
  in map3 (\Sigma mu esg_scores ->
              let ones = replicate n 1.0
              let neg_ones = map (f64.neg) ones
              let m = n + 4
              let A = [mu, esg_scores, ones, neg_ones] ++ linalg.eye n :> [m][n]f64
              let b = [0.5, 0.65, 1, -1] ++ replicate n 0.0 :> [m]f64
              let x_init = replicate n 0.0
              let x_init[0] = 1.0
              let rho = 0.15
              let tol = 1e-10
              let max_iter = 300
              let g x = map (* 2) (linalg.matvecmul_row Sigma x)
              let h _ = map (map (* 2)) Sigma
              in convex.admm g h A b x_init rho tol max_iter)
          Sigmas
          mus
          esg_scoress

-- == 
-- entry: admm_bench_batched
-- input @ data/2d_1000000x5x5_1d_1000000x5_1d_1000000x5f64
-- input @ data/2d_1000000x10x10_1d_1000000x10_1d_1000000x10f64
-- input @ data/2d_100000x20x20_1d_100000x20_1d_100000x20f64
-- input @ data/2d_10000x50x50_1d_10000x50_1d_10000x50f64
-- input @ data/2d_1000x100x100_1d_1000x100_1d_1000x100f64
-- input @ data/2d_100x500x500f64_1d_100x500_1d_100x500f64

-- Barrier
-- convex_bench.fut:barrier_bench_batched (no tuning file):
-- data/2d_1000000x5x5_1d_1000000x5_1d_1...:     381133μs (95% CI: [  380694.6,   381561.7])
-- data/2d_1000000x10x10_1d_1000000x10_1...:    1823316μs (95% CI: [ 1822856.2,  1823738.1])
-- data/2d_100000x20x20_1d_100000x20_1d_...:     552519μs (95% CI: [  552417.3,   552687.6])
-- data/2d_10000x50x50_1d_10000x50_1d_10...:     879378μs (95% CI: [  879251.4,   879506.0])
-- data/2d_1000x100x100_1d_1000x100_1d_1...:    3281759μs (95% CI: [ 3281554.6,  3281916.7])
-- data/2d_100x500x500f64_1d_100x500_1d_...:  282061036μs (95% CI: [282055972.0,282066100.0])

-- Barrier JVP
-- convex_bench.fut:barrier_bench_jvp_batched (no tuning file):
-- data/2d_1000000x5x5_1d_1000000x5_1d_1...:     920946μs (95% CI: [  920257.5,   921469.5])
-- data/2d_1000000x10x10_1d_1000000x10_1...:    4887080μs (95% CI: [ 4886358.2,  4887756.7])
-- data/2d_100000x20x20_1d_100000x20_1d_...:    5974507μs (95% CI: [ 5971914.7,  5976048.7])
-- data/2d_10000x50x50_1d_10000x50_1d_10...:   15708619μs (95% CI: [15708196.5, 15709149.4])
-- data/2d_1000x100x100_1d_1000x100_1d_1...:   27219190μs (95% CI: [27218764.0, 27219616.0])

-- Barrier VJP
-- convex_bench.fut:barrier_bench_vjp_batched (no tuning file):
-- data/2d_1000000x5x5_1d_1000000x5_1d_1...:     913159μs (95% CI: [  912050.6,   914204.3])
-- data/2d_1000000x10x10_1d_1000000x10_1...:    4669273μs (95% CI: [ 4661014.1,  4675026.1])
-- data/2d_100000x20x20_1d_100000x20_1d_...:    5140952μs (95% CI: [ 5137715.2,  5144218.4])
-- data/2d_10000x50x50_1d_10000x50_1d_10...:   21879821μs (95% CI: [21876678.1, 21882071.3])
-- data/2d_1000x100x100_1d_1000x100_1d_1...:   38868396μs (95% CI: [38865206.5, 38870747.6])

-- ADMM LU
-- convex_bench.fut:admm_bench_batched (no tuning file):
-- data/2d_1000000x5x5_1d_1000000x5_1d_1...:    2507170μs (95% CI: [ 2504539.0,  2510548.9])
-- data/2d_1000000x10x10_1d_1000000x10_1...:   10113011μs (95% CI: [10015835.5, 10145440.8])
-- data/2d_100000x20x20_1d_100000x20_1d_...:    4766267μs (95% CI: [ 4763828.4,  4769037.5])
-- data/2d_10000x50x50_1d_10000x50_1d_10...:   37977384μs (95% CI: [37447457.5, 38238057.4])
-- data/2d_1000x100x100_1d_1000x100_1d_1...:  139446494μs 

-- ADMM CG
-- convex_bench.fut:admm_bench_batched (no tuning file):
-- data/2d_1000000x5x5_1d_1000000x5_1d_1...:    3664286μs (95% CI: [ 3651655.4,  3671302.2])
-- data/2d_1000000x10x10_1d_1000000x10_1...:   17110162μs (95% CI: [17101379.2, 17118574.2])
-- data/2d_100000x20x20_1d_100000x20_1d_...:    9227441μs (95% CI: [ 9221793.4,  9234917.3])
-- data/2d_10000x50x50_1d_10000x50_1d_10...:   51621440μs (95% CI: [50955127.9, 51966451.1])
-- data/2d_1000x100x100_1d_1000x100_1d_1...:  344393814μs (95% CI: [344384000.0,344403648.0])

-- CVXPY
-- n = 5, elapsed = 2034657001.50 us
-- n = 10, elapsed = 2039673566.82 us
-- n = 20, elapsed = 220676279.07 us
-- n = 50, elapsed = 57815551.76 us
-- n = 100, elapsed = 304475522.04 us
-- n = 500, elapsed = 167484545.71 us
