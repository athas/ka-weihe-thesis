import "lib/github.com/diku-dk/linalg/linalg"
import "lib/github.com/diku-dk/cpprandom/random"
import "lib/github.com/diku-dk/linalg/lu"
import "linear"


local module type convex = {
    type t
    val power_iteration    [n][m] : [n][m]t -> i64 -> t -> ([n]t, t, [m]t)
    val svd                [n][m] : [n][m]t -> i64 -> t -> ([n][]t, []t, [][m]t)
    val pseudo_inverse     [n][m] : [n][m]t -> [m][n]t
    val least_squares      [n][m] : [n][m]t -> [n]t -> [m]t
    val line_search        [n]    : ([n]t -> t) -> [n]t -> [n]t -> [n]t -> t -> t -> i64 -> t
    val grad_jvp           [n]    : ([n]t -> t) -> [n]t -> [n]t
    val hess_jvp           [n]    : ([n]t -> t) -> [n]t -> [n][n]t
    val newton_equality    [n][m] : ([n]t -> [n]t) -> ([n]t -> [n][n]t) -> [m][n]t -> [m]t -> [n]t -> t -> i64 -> [n]t
    val newtons_method     [n] : ([n]t -> [n]t) -> ([n]t -> [n][n]t) -> [n]t -> t -> i64 -> [n]t
    val newtons_method_ls  [n] : ([n]t -> t) -> ([n]t -> [n]t) -> ([n]t -> [n][n]t) -> [n]t -> t -> t -> t -> i64 -> i64 -> [n]t
    val grad_vjp           [n]    : ([n]t -> t) -> [n]t -> [n]t
    val hess_vjp           [n]    : ([n]t -> t) -> [n]t -> [n][n]t
    val admm               [n][m] : ([n]t -> [n]t) -> ([n]t -> [n][n]t) -> [m][n]t -> [m]t -> [n]t -> t -> t -> i64 -> [n]t    
    val barrier_method     [n][m] : ([n]t -> t) -> ([n]t -> [n]t) -> ([n]t -> [n][n]t) -> ([n]t -> t) -> ([n]t -> [n]t) -> ([n]t -> [n][n]t) -> i64 -> [m][n]t -> [m]t -> [n]t -> t -> t -> t -> [n]t
    val newton_equality_ls [n][m] : ([n]t -> t) -> [m][n]t -> [m]t -> [n]t -> t -> t -> t -> i64 -> i64 -> [n]t
    val solve_qp_auto      [n][m][p] : ([n]t -> [p]t) -> ([n]t -> t) -> [m][n]t -> [m]t -> [n]t
    val solve_qp           [n][m][p] : ([n]t -> t) -> ([n]t -> [n]t) -> ([n]t -> [n][n]t) -> ([n]t -> [p]t) -> ([n]t -> t) -> ([n]t -> [n]t) -> ([n]t -> [n][n]t) -> [m][n]t -> [m]t -> [n]t
    val newton_equality_auto [n][m] : ([n]t -> t) -> [m][n]t -> [m]t -> [n]t -> t -> i64 -> [n]t
    val barrier_method_auto [n][m][p] : ([n]t -> t) -> ([n]t -> [n]t) -> [m][n]t -> [m]t -> [n]t -> [n]t
    val newtons_method_auto [n] : ([n]t -> t) -> [n]t -> t -> i64 -> [n]t
}

module mk_solver (T: real) : convex with t = T.t = {
    type t = T.t

    -- Modules used in the solver
    module rng_engine = minstd_rand 
    module rand_T     = uniform_real_distribution T rng_engine
    module linalg     = mk_linalg T
    module linear     = mk_linear T
    module lu         = mk_lu T

    -- Random 1D array of length n with elements in [low, high]
    def stream (n: i64) (low: t) (high: t) (seed: i32): [n]t =
        let rng_state = rng_engine.rng_from_seed [seed]
        let rng_states = rng_engine.split_rng n rng_state
        let (_,rng) = unzip (map (rand_T.rand (low, high)) rng_states)
        in rng

    -- Power iteration algorithm for finding the largest eigenvalue and corresponding eigenvector
    def power_iteration [n][m] (A: [n][m]t) (max_iter: i64) (tol: t): ([n]t, t, [m]t) =
        let v = stream m (T.f64 0.0) (T.f64 1.0) 42
        let ATA = linalg.matmul (transpose A) A
        let (v, _, _) = loop (v, i, r) = (v, 0, true) while r && i < max_iter do
            let old_v = v
            let v = linalg.matvecmul_row ATA v
            let v = map (T./(linalg.vecnorm v)) v 
            in if linalg.vecnorm (map2 (T.-) v old_v) T.< tol then (v, i, false) else (v, i + 1, true)
        let u = linalg.matvecmul_row A v
        let sigma = linalg.vecnorm u
        let u = map (T./sigma) u 
        in (u, sigma, v)

    -- Singular value decomposition
    def svd [n][m] (A: [n][m]t) (max_iter: i64) (tol: t) : ([n][]t, []t, [][m]t) =
        let k = i64.min n m
        let U = linalg.matzeros n k
        let S = linalg.veczeros k
        let V = linalg.matzeros m k
        let (_, U, S, V) = loop (A, U, S, V) for i in 0..<k do
            let (u, s, v) = power_iteration A max_iter tol
            let U[:,i] = u
            let S[i] = s
            let V[:,i] = v
            let A = linalg.matop (T.-) A (linalg.matunary (T.*s) (linalg.outer u v))
            in (A, U, S, V)
        in (U, S, transpose V)

    -- Moore-Penrose pseudo inverse
    def pseudo_inverse [n][m] (A: [n][m]t): [m][n]t =
        let (U, S, V) = svd A 1000 (T.f64 1e-8)
        let S_inv = map (\x: t -> if x T.> (T.f64 1e-15) then (T.f64 1.0) T./ x else T.f64 0.0) S
        let S_inv_diag = linalg.todiag S_inv
        in linalg.matmul (linalg.matmul (transpose V) S_inv_diag) (transpose U)

    -- Least squares solution
    def least_squares [n][m] (A: [n][m]t) (b: [n]t) =
        let A_pinv = pseudo_inverse A
        in linalg.matvecmul_row A_pinv b

    def armijo f x delta_x grad alpha t = 
        let left t = f (map2 (T.+) x (map (T.*t) delta_x))
        let right t = f x T.+ alpha T.* t T.* linalg.dotprod grad delta_x
        in left t T.<= right t

    def line_search [n] 
        (f: [n]t -> t)  -- Objective function
        (x: [n]t)       -- Current point
        (delta_x: [n]t) -- Search direction
        (grad: [n]t)    -- Gradient of f at x
        (alpha: t)      -- Parameter for Armijo condition
        (beta: t)       -- Parameter for Armijo condition
        (max_iter: i64) -- Maximum number of iterations
        : t =
        let (t, _) = loop (t, i) = (T.f64 1.0, 0) while i <= max_iter && !(armijo f x delta_x grad alpha t) do
            (t T.* beta, i + 1)
        in t

    def parallel_line_search f x delta_x grad alpha beta n_ts =
        let ts = map (\i -> (T.f64 1.0) T./ (beta T.** (T.i64 i))) (iota n_ts)
        let conds = map (\t -> (t, armijo f x delta_x grad alpha t)) ts
        in (reduce (\(t1, b1) (t2, b2) -> if b1 then (t1, b1) else (t2, b2)) ((T.f64 0.0), false) conds).0

    def line_search_default [n] (f: [n]t -> t) (x: [n]t) (delta_x: [n]t) (grad: [n]t) : t =
        line_search f x delta_x grad (T.f64 0.3) (T.f64 0.8) 100

    -- Compute gradient of f at x
    def grad_jvp [n] (f: [n]t -> t) (x: [n]t) =
        map (\i -> 
            let X = linalg.veczeros n
            let X[i] = T.f64 1.0
            in jvp f x X
        ) (iota n)
    
    -- Compute hessian of f at x
    def hess_jvp [n] (f: [n]t -> t) (x: [n]t) =
        map (\i ->
            map (\j ->
                let X = linalg.veczeros n
                let X[i] = T.f64 1.0
                let Y = linalg.veczeros n
                let Y[j] = T.f64 1.0
                in jvp (\w -> jvp f w X) x Y
            ) (iota n)
        ) (iota n)
    
    def grad_vjp [n] (f: [n]t -> t) (x: [n]t): [n]t =
        vjp f x (T.f64 1f64)

    def hess_vjp [n] (f: [n]t -> t) (x: [n]t): [n][n]t =
        map (\i ->
            let X = linalg.veczeros n
            let X[i] = T.f64 1.0
            in vjp (\w -> vjp f w (T.f64 2.0)) x X
        ) (iota n)

    -- Gradient descent for solving quadratic program without constraints
    def gradient_descent [n] 
        (f: [n]t -> t) 
        (f_grad: [n]t -> [n]t) 
        (x0: [n]t) 
        (epsilon: t) 
        (alpha: t) 
        (beta: t) 
        (max_iter: i64)
        (max_iter_line_search: i64): [n]t =
        let (x, _, _) = loop (x, i, r) = (x0, 0, true) while i < max_iter && r do
            let grad_val = f_grad x
            let delta_x = map (T.neg) grad_val
            let t = line_search f x delta_x grad_val alpha beta max_iter_line_search
            let x = map2 (T.+) x (map (T.*t) delta_x)
            let r = linalg.vecnorm delta_x T.> epsilon
            in (x, i + 1, r)
        in x

    -- Newtons method for solving unconstrained problems
    def newtons_method [n] 
        (f_grad: [n]t -> [n]t) 
        (f_hess: [n]t -> [n][n]t) 
        (x0: [n]t) 
        (epsilon: t)                
        (max_iter: i64)
        : [n]t =
        let res = loop (x, i, r) = (x0, 0, true) while r && i < max_iter do
            let grad = f_grad x
            let hess = f_hess x
            let step = linear.solve_Ab_cholesky_seq hess (map (T.neg) grad)
            let x = map2 (T.+) x step
            let r = linalg.vecnorm step T.< epsilon
            in (x, i + 1, r)
        in res.0

    def newtons_method_auto [n] (f: [n]t -> t) (x0: [n]t) (epsilon: t) (max_iter: i64): [n]t =
        let f_grad = grad_jvp f
        let f_hess = hess_jvp f
        in newtons_method f_grad f_hess x0 epsilon max_iter
        

    -- Newtons method for solving unconstrained problems with line search
    def newtons_method_ls [n] 
        (f: [n]t -> t) 
        (f_grad: [n]t -> [n]t) 
        (f_hess: [n]t -> [n][n]t) 
        (x0: [n]t) 
        (epsilon: t) 
        (alpha: t) 
        (beta: t) 
        (max_iter: i64)
        (max_iter_line_search: i64)
        : [n]t =
        let res = loop (x, i, r) = (x0, 0, true) while r && i < max_iter do
            let grad = f_grad x
            let hess = f_hess x
            let step = linear.solve_Ab_cholesky_seq hess (map (T.neg) grad)
            let t = line_search f x step grad alpha beta max_iter_line_search
            -- let t = parallel_line_search f x step grad alpha beta n_ts
            let x = map2 (\a b -> a T.+ b T.* t) x step
            let r = linalg.vecnorm step T.< epsilon
            in (x, i + 1, r)
        in res.0

    def newton_equality [n][m] 
        (f_grad: [n]t -> [n]t)
        (f_hess: [n]t -> [n][n]t)
        (A: [m][n]t) 
        (b: [m]t) 
        (x0: [n]t) 
        (epsilon: t) 
        (max_iter: i64)
        : [n]t =
        let res = loop (x, i, r) = (x0, 0, true) while r && i < max_iter do
            let grad_val = f_grad x
            let hessian_val = f_hess x
            let KKT = linalg.matzeros (n + m) (n + m)
            let rhs = linalg.veczeros (n + m)
            let KKT[:n, :n] = hessian_val
            let KKT[:n, n:] = transpose A
            let KKT[n:, :n] = A
            let rhs[:n] = map T.neg grad_val
            let rhs[n:] = map2 (T.-) b (linalg.matvecmul_row A x)
            let delta_x_lam = linear.solve_Ab_lu_seq KKT rhs
            let delta_x = delta_x_lam[:n]
            let x_new = map2 (T.+) x delta_x
            in if linalg.vecnorm delta_x T.< epsilon then
                (x, i+1, false)
            else
                (x_new, i+1, true)
        in res.0

    def newton_equality_auto [n][m] (f: [n]t -> t) (A: [m][n]t) (b: [m]t) (x0: [n]t) (epsilon: t) (max_iter: i64): [n]t =
        let f_grad = grad_jvp f
        let f_hess = hess_jvp f
        in newton_equality f_grad f_hess A b x0 epsilon max_iter

    def newton_equality_ls [n][m] 
        (f: [n]t -> t) 
        (A: [m][n]t) 
        (b: [m]t) 
        (x0: [n]t) 
        (epsilon: t) 
        (alpha: t) 
        (beta: t) 
        (max_iter: i64)
        (max_iter_line_search: i64)
        : [n]t =
        let f_grad = grad_jvp f
        let f_hess = hess_jvp f
        let (x, _, _) = loop (x, r, i) = (x0, true, 0) while r && i < max_iter do
            let grad_val = f_grad x
            let hessian_val = f_hess x
            let KKT = linalg.matzeros (n + m) (n + m)
            let rhs = linalg.veczeros (n + m)
            let KKT[:n, :n] = hessian_val
            let KKT[:n, n:] = transpose A
            let KKT[n:, :n] = A
            let rhs[:n] = map T.neg grad_val
            let rhs[n:] = map2 (T.-) b (linalg.matvecmul_row A x)
            let delta_x_lam = linear.solve_Ab_lu_seq KKT rhs
            let delta_x = delta_x_lam[:n]
            let t = line_search f x delta_x grad_val alpha beta max_iter_line_search
            let x_new = map2 (T.+) x (map (T.*t) delta_x)
            in if linalg.vecnorm delta_x T.< epsilon then
                (x, false, i+1)
            else
                (x_new, true, i+1)
        in x


    def barrier_method [n][m]
        (f0: [n]t -> t) 
        (f0_grad: [n]t -> [n]t)
        (f0_hess: [n]t -> [n][n]t)
        (phi: [n]t -> t)
        (phi_grad: [n]t -> [n]t)
        (phi_hess: [n]t -> [n][n]t)
        (mm: i64)
        (A: [m][n]t) 
        (b: [m]t) 
        (x0: [n]t)     
        (t0: t)         
        (mu: t)           
        (eps1: t)              
        : [n]t =
        let iters = T.to_i64 (T.log (T.i64 mm T./ (eps1 T.* t0)) T./ T.log mu T.+ T.f64 1.0)
        let res = loop (x, t) = (x0, t0) for i < iters do
            let f x = t T.* (f0 x) T.+ phi x
            let f_grad x = map (T.* t) (f0_grad x) |> map2 (T.+) (phi_grad x)
            let f_hess x = map (map (T.* t)) (f0_hess x) |> map2 (map2 (T.+)) (phi_hess x)
            let grad_val = f_grad x
            let hessian_val = f_hess x
            let KKT = linalg.matzeros (n + m) (n + m)
            let rhs = linalg.veczeros (n + m)
            let KKT[:n, :n] = hessian_val
            let KKT[:n, n:] = transpose A
            let KKT[n:, :n] = A
            let rhs[:n] = map T.neg grad_val
            let rhs[n:] = map2 (T.-) b (linalg.matvecmul_row A x)
            let delta_x_lam = linear.solve_Ab_lu_seq KKT rhs
            let delta_x = delta_x_lam[:n]
            let s = 
                loop s = (T.f64 1.0) 
                while T.isnan (f (map2 (\a b -> a T.+ b T.* s) x delta_x)) && s T.> T.f64 1e-15
                do s T.* (T.f64 0.8)
            let x_new = map2 (\a b -> a T.+ b T.* s) x delta_x
            in (x_new, t T.* mu)
        in res.0

    def barrier_method_auto [n][m][p] (f0: [n]t -> t) (fi: [n]t -> [p]t)  (A: [m][n]t) (b: [m]t) (x: [n]t) =
        let f0_grad = grad_jvp f0
        let f0_hess = hess_jvp f0
        let phi x = T.neg (fi x |> map (T.log <-< T.neg) |> reduce (T.+) (T.f64 0.0))
        let phi_grad = grad_jvp phi
        let phi_hess = hess_jvp phi
        let mm = length (fi x)
        in barrier_method f0 f0_grad f0_hess phi phi_grad phi_hess mm A b x (T.f64 1.0) (T.f64 3) (T.f64 1e-10)

    def find_feasible_point [n][m][p] (fi: [n]t -> [p]t) (A: [m][n]t) (b: [m]t): [n]t =
        let x = least_squares A b -- Solving Ax = b
        -- Compute s, an upper bound for the maximum constraint violation
        let s = (fi x |> reduce T.max (T.f64 0.0)) T.+ (T.f64 0.001)
        let n' = n + 1
        let x = x ++ [s] :> [n']t
        let A = transpose ((transpose A) ++ [linalg.veczeros m]) :> [m][n']t
        let f [n'] (x: [n']t): t = x[n'-1] -- New objective function
        let fi_new [n'] (x: [n']t): [p]t = -- Define the modified constraint function
            let s = x[n'-1]
            let x = x[0: n'-1] :> [n]t
            in (map (\x' -> x' T.- s) (fi x)) :> [p]t
        let x = barrier_method_auto f fi_new A b x 
        in x[0: n] :> [n]t

    -- Solve quadratic program with inequality constraints using barrier method
    def solve_qp_auto [n][m][p] (fi: [n]t -> [p]t) (f: [n]t -> t) (A: [m][n]t) (b: [m]t): [n]t =
        let x = find_feasible_point fi A b
        let x = barrier_method_auto f fi A b x
        in x

    def solve_qp [n][m][p] 
        (f: [n]t -> t)
        (f_grad: [n]t -> [n]t)
        (f_hess: [n]t -> [n][n]t) 
        (fi: [n]t -> [p]t)
        (phi: [n]t -> t)
        (phi_grad: [n]t -> [n]t)
        (phi_hess: [n]t -> [n][n]t)
        (A: [m][n]t) 
        (b: [m]t): [n]t =
        let x = find_feasible_point fi A b
        let x = barrier_method f f_grad f_hess phi phi_grad phi_hess p A b x (T.f64 1.0) (T.f64 3) (T.f64 1e-10)
        in x    
    
    -- ADMM
    -- Ax >= b
    def admm [n][m] (f_grad: [n]t -> [n]t)    -- gradient of the objective function
                    (f_hess: [n]t -> [n][n]t) -- Hessian of the objective function
                    (A: [m][n]t)              -- constraint matrix
                    (b: [m]t)                 -- constraint vector
                    (x_init: [n]t)            -- initial guess for x
                    (rho: t)                  -- penalty parameter (default: 5)
                    (tol: t)                  -- tolerance for convergence (default: 1e-6)
                    (max_iter: i64)           -- maximum number of iterations (default: 1000)
                    : [n]t =
     
        let update_x (x: [n]t) (z: [m]t) (u: [m]t) =
            let Ax = linalg.matvecmul_row A x
            let g1 = f_grad x
            let Ax_z = map2 (T.-) Ax z
            let Ax_z_u = map2 (T.+) Ax_z u
            let At_t = linalg.matvecmul_row (transpose A) Ax_z_u
            let g2 = map (rho T.*) At_t
            let gradient = map2 (T.+) g1 g2
            let h1 = f_hess x
            let At_A = linalg.matmul (transpose A) A
            let h2 = map (map (rho T.*)) At_A
            let hessian = map2 (map2 (T.+)) h1 h2
            -- let delta_x = linear.solve_Ab_cg hessian gradient (linalg.veczeros n) (T.f64 1e-8) n
            let delta_x = linear.solve_Ab_cholesky_seq hessian gradient
            in map2 (T.-) x delta_x

        let update_z (x: [n]t) (u: [m]t) =
            let Ax = linalg.matvecmul_row A x
            let zu = map2 (T.+) Ax u
            in map2 T.max b zu        
        
        let update_u x z u =
            let Ax = linalg.matvecmul_row A x
            let Ax_z = map2 (T.-) Ax z
            in map2 (T.+) u Ax_z
        
        let convergence_criterion x x_prev =
            let delta_x = map2 (T.-) x x_prev
            let left = linalg.dotprod delta_x delta_x
            let right = linalg.dotprod x x
            let update = left T./ (right T.+ (T.f64 1e-16))
            in update T.< tol

        let x = x_init
        let z = linalg.veczeros m
        let u = linalg.veczeros m
        let i = 0
        let r = true
        let res = loop (x, z, u, r, i) while r && i < max_iter do
            let x_prev =     x
            let x = update_x x z u
            let z = update_z x u
            let u = update_u x z u

            in if convergence_criterion x x_prev then
                (x, z, u, false, i + 1)
            else
                (x, z, u, true, i + 1)

        in res.0
}
