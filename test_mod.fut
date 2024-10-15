import "mod"
import "lib/github.com/diku-dk/linalg/linalg"

module mod = mk_solver f64
module linalg = mk_linalg f64

let all_close [n] (xs: [n]f64) (ys: [n]f64) (eps: f64): bool =
  let diffs = map2 (-) xs ys
  in reduce (&&) true (map (\x -> f64.abs x < eps) diffs)

let all_close2d [n][m] (xss: [n][m]f64) (yss: [n][m]f64) (eps: f64): bool =
  let diffs = map2 (map2 (-)) xss yss
  in reduce (&&) true (map (\xs -> reduce (&&) true (map (\x -> f64.abs x < eps) xs)) diffs)

entry test_power_iteration [n][m] (A: [n][m]f64) (test_tol: f64): bool = 
  let (u, s, v) = mod.power_iteration A 10000 1e-8
  let Av = linalg.matvecmul_row A v
  let su = map (*s) u
  in all_close Av su test_tol

-- ==
-- entry: test_power_iteration
-- compiled input {
-- [[1.0, 2.0, 3.0],
-- [4.0, 5.0, 6.0]] 1e-8f64}
-- output { true }

let diag [n] (xs: [n]f64): [n][n]f64 =
  tabulate_2d n n (\i j -> if i == j then xs[i] else 0.0)

entry test_svd [n][m] (A: [n][m]f64) (test_tol: f64): bool =
  let (U, S, V) = mod.svd A 10000 1e-8
  let k = i64.min n m
  let U = U :> [n][k]f64
  let V = V :> [k][m]f64
  let S = S :> [k]f64
  let S_diag = diag S
  let A_reconstructed = linalg.matmul (linalg.matmul U S_diag) V
  in all_close2d A A_reconstructed test_tol

-- ==
-- entry: test_svd
-- compiled input {
-- [[1.0, 2.0, 3.0],
-- [4.0, 5.0, 6.0]] 1e-8f64}
-- output { true }

entry test_pseudoinverse [n][m] (A: [n][m]f64): bool =
  let A_pinv = mod.pseudo_inverse A
  let A_reconstructed = linalg.matmul (linalg.matmul A A_pinv) A
  in all_close2d A_reconstructed A 1e-8

-- ==
-- entry: test_pseudoinverse
-- compiled input {
-- [[1.0, 2.0, 3.0],
-- [4.0, 5.0, 6.0]]}
-- output { true }

entry test_least_squares [n][m] (A: [n][m]f64) (b: [n]f64): bool =
  let x = mod.least_squares A b
  let b_reconstructed = linalg.matvecmul_row A x
  in all_close b_reconstructed b 1e-8

-- ==
-- entry: test_least_squares
-- compiled input {
-- [[1.0, 2.0, 3.0],
-- [4.0, 5.0, 6.0]] [1.0, 2.0]}
-- output { true }

    -- def line_search [n] (f        : [n]t -> t) 
    --                     (f_grad   : [n]t     ) 
    --                     (x        : [n]t     ) 
    --                     (d        : [n]t     ) 
    --                     (alpha    : t        ) 
    --                     (beta     : t        )
    --                     (max_iter : i64      ): t =


let is_close a b eps =
  let diff = f64.abs (a - b)
  in diff < eps

entry test_line_search =
  let f x = x[0] ** 2.0
  let grad_f x = [2.0 * x[0]]
  let x = [2.0]
  let delta_x = map f64.neg (grad_f x)
  let grad = grad_f x
  let f_val = f x
  let alpha = 0.3
  let beta = 0.8
  let t = mod.line_search f x delta_x grad alpha beta 1000
  let expected_t = 0.64
  in is_close t expected_t 1e-6

-- ==
-- entry: test_line_search
-- input { }
-- output { true }

-- entry test_build_kkt_matrix = mod.build_kkt_matrix
-- -- ==
-- -- entry: test_build_kkt_matrix
-- -- input { [[1.0, 2.0], [3.0, 4.0]] [[1.0, 2.0], [3.0, 4.0]]} 
-- -- output { [[1.0, 2.0, 1.0, 3.0], [3.0, 4.0, 2.0, 4.0], [1.0, 2.0, 0.0, 0.0], [3.0, 4.0, 0.0, 0.0]] }

-- entry test_compute_newton_step = 
--   let hessian = [[1.0, 2.0], [3.0, 4.0]]
--   let grad = [1.0, 2.0]
--   let A = [[1.0, 2.0], [3.0, 4.0]]
--   let b = [1.0, 2.0]
--   let x = [1.0, 2.0]
--   let (delta_x, lambda_x) = mod.compute_newton_step hessian grad A b x
--   let output = ([-1.0, -1.5], [4.5, -0.5])
--   in all_close delta_x output.0 1e-6 && all_close lambda_x output.1 1e-6

-- -- ==
-- -- entry: test_compute_newton_step
-- -- input { }
-- -- output { true }

entry test_newton_equality =
  let Q = [[4.0, 1.0], [1.0, 3.0]]
  let b = [1.0, 1.0]
  let f x = (linalg.matmul [x] (linalg.matmul Q (transpose [x])))[0, 0] * 0.5 - linalg.dotprod b x
  let f_grad x = map2 (-) (linalg.matmul [x] Q)[0] b
  let f_hess _ = Q
  let A = [[1.0, 1.0]]
  let b_eq = [0.0]
  let x0 = [-1.0, 1.0]
  let x_min = mod.newton_equality f A b_eq x0 1e-8 0.3 0.8 10000 100000
  let expected_result = [0.0, 0.0]
  in all_close x_min expected_result 1e-3

-- ==
-- entry: test_newton_equality
-- input { }
-- output { true }


let range a b n = 
    let step = (b-a)/f64.i64 n
    in map (\i -> a + step * f64.i64 i) (iota n)

let range1 = (range 0.0 0.072 1000)

let main = 
    let mu = [0.073, -0.019, -0.02, -0.041, 0.039, -0.082, 0.068, -0.028, 0.013, -0.01]
    let esg_scores = [0.9, 0.548, 0.906, 0.552, 0.758, 0.578, 0.639, 0.941, 0.816, 0.982]
    let S = [[0.009, -0.001, -0.0, 0.0, 0.001, 0.001, 0.001, -0.001, -0.0, -0.0], [-0.001, 0.009, 0.001, -0.001, 0.0, -0.0, -0.001, -0.0, -0.0, -0.001], [-0.0, 0.001, 0.011, 0.0, 0.0, 0.0, 0.001, 0.001, -0.0, -0.001], [0.0, -0.001, 0.0, 0.011, -0.002, -0.001, -0.001, -0.0, -0.0, 0.001], [0.001, 0.0, 0.0, -0.002, 0.011, 0.0, 0.001, 0.001, -0.001, -0.0], [0.001, -0.0, 0.0, -0.001, 0.0, 0.009, 0.0, -0.001, -0.0, 0.001], [0.001, -0.001, 0.001, -0.001, 0.001, 0.0, 0.011, -0.0, 0.001, -0.0], [-0.001, -0.0, 0.001, -0.0, 0.001, -0.001, -0.0, 0.01, 0.001, 0.001], [-0.0, -0.0, -0.0, -0.0, -0.001, -0.0, 0.001, 0.001, 0.01, -0.001], [-0.0, -0.001, -0.001, 0.001, -0.0, 0.001, -0.0, 0.001, -0.001, 0.01]]
    
  let h (exp: f64) (x: [10]f64): [12]f64 =
    let expected_return = linalg.dotprod x mu
    let expected_esg = linalg.dotprod x esg_scores
    -- constraints
    in ([
        -- -- expected_return >= 0.07           
        exp - expected_return,
        -- -- expected esg <= 0.8   
        0.8 - expected_esg             
        -- -- w >= -0.0001
    ] ++ map (f64.neg <-< (+1e-6)) x) :> [12]f64
  
  let f0 (x: [10]f64) = (linalg.matmul (linalg.matmul [x] S) (transpose [x]))[0,0]
  let A = [replicate 10 1.0]
  let b = [1.0]
  let res = map (\r -> mod.solve_qp (h r) f0 A b) range1
  in res
  -- let x0 = replicate 10 0.0
  -- let x0[0] = 1.0
  -- -- let res = map (\r -> mod.barrier_method (h r) f0 A b x0 0.3 0.8 1e-8 1e-8 0.3 r 100 100 100) range1
  -- let res = map (\r -> mod.newton_equality f0 A b x0 1e-8 0.3 0.8 100 100) range1
  -- in res
--   let output = [ 6.00438987e-01, -6.75602330e-23, -7.97331739e-23, -9.00311533e-23, 3.47138099e-02, -1.27938597e-22,  3.64847203e-01, -8.63565107e-23, -4.70026822e-23, -7.13413811e-23]
--   in all_close res output 1e-3

-- Known compiler limitation encountered.  Sorry.
-- Revise your program or try a different Futhark compiler.
-- Cannot handle un-sliceable allocation size: (((thread; virtualise; 
-- groups=num_groups_78798; groupsize=segmap_group_size_78797), [phys_tid_78802]), bytes_94637, )
-- Likely cause: irregular nested operations inside parallel constructs.