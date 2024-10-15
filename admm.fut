import "lib/github.com/diku-dk/linalg/linalg"
module linalg = mk_linalg f64

let dotprod [n] (xs: [n]f64) (ys: [n]f64): f64 =
  reduce (+) 0.0 (map2 (*) xs ys)

let matmul [n][p][m] (xss: [n][p]f64) (yss: [p][m]f64): [n][m]f64 =
  map (\xs -> map (dotprod xs) (transpose yss)) xss

let tri_num (n: i64): i64 = (n * (n + 1)) / 2

-- let tril_indices (n: i64): *[](i16, i16) =
--   tabulate (n * n) (\x ->
--     let i = i16.i64 (x / n)
--     let j = i16.i64 (x % n)
--     in if j >= i then (j, i) else (-1i16, -1i16)
--   ) |> filter (\ (x, _) -> x != -1)

def tril k n =
    let kf = f64.i64 k
    let nf = f64.i64 n
    let kp = nf * (nf + 1) / 2 - kf - 1
    let p = f64.floor((f64.sqrt(1 + 8 * kp) - 1) / 2)
    let i = (nf - (kp - p * (p + 1) / 2)) - 1
    let j = nf - p - 1
    in (i16.f64 i, i16.f64 j)

let tril_indices n: *[](i16, i16) = 
    let s = tri_num n
    let is = iota s
    let is = map (\x -> tril x n) is
    in is

let low_outer (v: []f64) (ti: [](i16, i16)): *[]f64 =
  map (\ x -> 
    let (i, j) = x
    in #[unsafe] v[i] * v[j]
  ) ti

-- We compute the upper triangular matrix R because it is more efficient
let flat_cho [n] (A: [n][n]f64): []f64 =
  -- Convert A to 1D array of upper triangular elements
  let tril_is = tril_indices n
  let s = length tril_is 
  let A = map (\(i, j) -> A[i, j]) tril_is 
  let m = tri_num n
  let A = loop A for j in 0..<n-1 do  
    -- Take the square root of the diagonal element
    let di = (tri_num n) - (tri_num (n - j))
    let sdi = f64.sqrt A[di]
    let A[di] = sdi
    -- Division step
    let steps = n - j - 1 
    let start = di + 1
    let stop = start + steps
    let A[start:stop] = map (/sdi) A[start:stop]
    -- Elimination step
    let skip_cols = j + 1
    let rest = n - skip_cols 
    let k = tri_num rest
    let outer_is = tril_is[s-k:]
    let outer_is = map (\(a, b) -> 
      let (c, d) = outer_is[0]
      in (a - c, b - d)
    ) outer_is
    let col = A[start:stop] :> *[steps]f64
    let outer = (low_outer col outer_is) :> [k]f64
    let A[m-k:] = map2 (\x y -> x - y) (A[m-k:] :> [k]f64) outer 
    in A
  
  -- Take the square root of the last element
  let di = m - 1
  let A[di] = f64.sqrt A[di] 
  in A

-- def i_j_to_index_colwise_diag(i, j, n):
--     p = n - j - 1
--     kp = p * (p + 1) / 2 + n - i - 1
--     k = n * (n + 1) / 2 - kp - 1
--     return int(k)

let i_j_to_index_colwise_diag i j (n: i64): i64 =
    let p = n - j - 1
    let kp = p * (p + 1) / 2 + n - i - 1
    let k = n * (n + 1) / 2 - kp - 1
    in k

let cholesky [n] (A:[n][n]f64): [n][n]f64 =
  let L = flat_cho A
--   in map (\i ->
--     map (\j -> 
--         if j >= i then 0f64 else L[i_j_to_index_colwise_diag i j n]
--     ) (iota n)
--   ) (iota n)
  in tabulate n (\x ->
    let tn = tri_num n
    let start = tn - (tri_num (n - x))
    let stop = tn - (tri_num (n - x - 1))
    in replicate n 0 with [x:] = L[start:stop]
  ) |> transpose

entry tril_indices_bench = tril_indices
-- ==
-- entry: tril_indices_bench
-- input { 10000i64}

-- let compare n =
--     let is = tril_indices n
--     let is2 = tril_indices2 n
--     in map (\x -> is[x] == is2[x]) (iota (length is)) |> reduce (&&) true

entry flat_cho_bench_batch [n] (As: [][n][n]f64) = 
    let s = tri_num n
    in map (\A -> (cholesky A)) As
-- ==
-- entry: flat_cho_bench_batch
-- input @ 10000x10

entry flat_cho_bench [n] (A: [n][n]f64) = 
    flat_cho A
-- ==
-- entry: flat_cho_bench
-- input @ 1000


-- def admm(f, f_grad, f_hess, A, b, x_init, rho=200, tol=1e-6, max_iter=1000):
--     """
--     Solve an optimization problem using the Alternating Direction Method of Multipliers (ADMM).
--     Ax >= b
    
--     Parameters:
--     - f: objective function
--     - f_grad: gradient of the objective function
--     - f_hess: Hessian of the objective function
--     - A: constraint matrix
--     - b: constraint vector
--     - x_init: initial guess for x
--     - rho: penalty parameter (default: 5)
--     - tol: tolerance for convergence (default: 1e-6)
--     - max_iter: maximum number of iterations (default: 1000)
    
--     Returns:
--     - x: optimal solution
--     """
    -- def update_x(x, z, u):
    --     gradient = f_grad(x) + rho * A.T @ (A @ x - z + u)
    --     hessian = f_hess(x) + rho * A.T @ A
    --     delta_x = np.linalg.inv(hessian) @ gradient
    --     return x - delta_x
    
--     def update_z(x, u):
--         return np.maximum(b, A @ x + u)
    
--     def update_u(x, z, u):
--         return u + A @ x - z
    
--     def convergence_criterion(x, x_prev):
--         delta_x = x - x_prev
--         update = delta_x.T @ delta_x / (x.T @ x + 1e-12)
--         return update < tol

--     m, n = A.shape
--     x = x_init.copy()
--     z = np.zeros(m)
--     u = np.zeros(m)

--     for _ in range(max_iter):
--         x_prev = x.copy()
--         x = update_x(x, z, u)
--         z = update_z(x, u)    
--         u = update_u(x, z, u) 

--         if convergence_criterion(x, x_prev):
--             break

--     return x

let grad [n] (f: [n]f64 -> f64) (x: [n]f64) =
  map (\i -> 
    let X = replicate n 0.0
    let X[i] = 1.0
    in jvp f x X
  ) (iota n)

let hess [n] (f: [n]f64 -> f64) (x: [n]f64) =
    map (\i ->
        map (\j ->
            let X = replicate n 0.0
            let X[i] = 1.0
            let Y = replicate n 0.0
            let Y[j] = 1.0
            in jvp (\w -> jvp f w X) x Y
        ) (iota n)
    ) (iota n)

def admm [n][m] (f: [n]f64 -> f64)            -- objective function
                (f_grad: [n]f64 -> [n]f64)    -- gradient of the objective function
                (f_hess: [n]f64 -> [n][n]f64) -- Hessian of the objective function
                (A: [m][n]f64)                -- constraint matrix
                (b: [m]f64)                   -- constraint vector
                (x_init: [n]f64)              -- initial guess for x
                (rho: f64)                    -- penalty parameter (default: 5)
                (tol: f64)                    -- tolerance for convergence (default: 1e-6)
                (max_iter: i64)               -- maximum number of iterations (default: 1000)
                =
     
    let update_x (x: [n]f64) (z: [m]f64) (u: [m]f64) =
        let Ax = linalg.matvecmul_row A x
        let g1 = grad f x
        let Ax_z = map2 (-) Ax z
        let Ax_z_u = map2 (+) Ax_z u
        let At_t = linalg.matvecmul_row (transpose A) Ax_z_u
        let g2 = map (rho*) At_t
        let gradient = map2 (+) g1 g2
        let h1 = hess f x
        let At_A = linalg.matmul (transpose A) A
        let h2 = map (map (rho*)) At_A
        let hessian = map2 (map2 (+)) h1 h2
        let inv_hess = linalg.inv hessian
        let delta_x = linalg.matvecmul_row inv_hess gradient
        in map2 (-) x delta_x

    let update_z (x: [n]f64) (u: [m]f64) =
        let Ax = linalg.matvecmul_row A x
        let zu = map2 (+) Ax u
        in map2 f64.max b zu        
    
    let update_u x z u =
        let Ax = linalg.matvecmul_row A x
        let Ax_z = map2 (-) Ax z
        in map2 (+) u Ax_z
    
    let convergence_criterion x x_prev =
        let delta_x = map2 (-) x x_prev
        let left = linalg.dotprod delta_x delta_x
        let right = linalg.dotprod x x
        let update = left / (right+1e-12)
        in update < tol
    
    
    let x = x_init
    let z = replicate m 0.0f64
    let u = replicate m 0.0f64
    let i = 0
    let r = true

    let (x, z, u, r, i) = loop (x, z, u, r, i) while r && i < max_iter do
        let x_prev =     x
        let x = update_x x z u
        let z = update_z x u
        let u = update_u x z u

        in if convergence_criterion x x_prev then
            (x, z, u, false, i + 1)
        else
            (x, z, u, true, i + 1)

    in x

def eye (n : i64) : [n][n]f64 =
    tabulate_2d n n (\i j -> if i == j then f64.i64 1 else f64.i64 0)

-- def conjugate_gradient(A, b, x0=None, tol=1e-10, max_iter=100):
--     r = b - A @ x0
--     p = r.copy()
--     rs_old = np.dot(r, r)
--     x = x0.copy()

--     for _ in range(max_iter):
--         Ap = A @ p
--         alpha = rs_old / np.dot(p, Ap)
--         x += alpha * p
--         r -= alpha * Ap
--         rs_new = np.dot(r, r)
--         if np.sqrt(rs_new) < tol:
--             break

--         p = r + (rs_new / rs_old) * p
--         rs_old = rs_new

--     return x

def conjugate_gradient [n] (A: [n][n]f64) (b: [n]f64) (x0: [n]f64) (tol: f64) (max_iter: i64) =
  let r = map2 (-) b (linalg.matvecmul_row A x0)
  let rs_old = linalg.dotprod r r
  let p = r
  let x = x0
  let i = 0
  let running = true
  let res = loop (x, p, r, rs_old, i, running) while running && i < max_iter do
    let Ap = linalg.matvecmul_row A p
    let alpha = rs_old / linalg.dotprod p Ap
    let x = map2 (+) x (map (alpha*) p)
    let r = map2 (-) r (map (alpha*) Ap)
    let rs_new = linalg.dotprod r r
    let running = f64.sqrt rs_new > tol
    let p = map2 (+) r (map ((rs_new/rs_old)*) p)
    let rs_old = rs_new
    let i = i + 1
    in (x, p, r, rs_old, i, running)
  in res.0

entry cg = 
  (\As b x0 -> 
    (map (\A -> conjugate_gradient A b x0 1e-6 10) As))

-- ==
-- entry: cg
-- input @ conjugate
let range a b n = 
    let step = (b-a)/f64.i64 n
    in map (\i -> a + step * f64.i64 i) (iota n)


entry hej l = 
  let mu = [0.073, -0.019, -0.02, -0.041, 0.039, -0.082, 0.068, -0.028, 0.013, -0.01]
  let esg_scores = [0.9, 0.548, 0.906, 0.552, 0.758, 0.578, 0.639, 0.941, 0.816, 0.982]
  let S = [[0.009, -0.001, -0.0, 0.0, 0.001, 0.001, 0.001, -0.001, -0.0, -0.0], [-0.001, 0.009, 0.001, -0.001, 0.0, -0.0, -0.001, -0.0, -0.0, -0.001], [-0.0, 0.001, 0.011, 0.0, 0.0, 0.0, 0.001, 0.001, -0.0, -0.001], [0.0, -0.001, 0.0, 0.011, -0.002, -0.001, -0.001, -0.0, -0.0, 0.001], [0.001, 0.0, 0.0, -0.002, 0.011, 0.0, 0.001, 0.001, -0.001, -0.0], [0.001, -0.0, 0.0, -0.001, 0.0, 0.009, 0.0, -0.001, -0.0, 0.001], [0.001, -0.001, 0.001, -0.001, 0.001, 0.0, 0.011, -0.0, 0.001, -0.0], [-0.001, -0.0, 0.001, -0.0, 0.001, -0.001, -0.0, 0.01, 0.001, 0.001], [-0.0, -0.0, -0.0, -0.0, -0.001, -0.0, 0.001, 0.001, 0.01, -0.001], [-0.0, -0.001, -0.001, 0.001, -0.0, 0.001, -0.0, 0.001, -0.001, 0.01]]
  let ones = replicate 10 1.0
  let neg_ones = map (f64.neg) ones
  let A = [mu, esg_scores, ones, neg_ones] ++ eye 10 :> [14][10]f64
  let b r e =  [r, e, 1, -1] ++ replicate 10 0.0 :> [14]f64
  let x_init = replicate 10 0.0
  let x_init[0] = 1.0
  let rho = 0.1
  let tol = 1e-9
  let max_iter = 5000

  let f0 x = (matmul (matmul [x] S) (transpose [x]))[0,0]
  let f0_grad = grad f0
  let f0_hess = hess f0
  in (map (\e -> (map (\r -> admm f0 f0_grad f0_hess A (b r e) x_init rho tol max_iter) (range 0.0 l 100))) (range 0.8 0.9 100))
-- ==
-- entry: hej
-- input { 0.07 }


let main =
    let mu = [0.073, -0.019, -0.02, -0.041, 0.039, -0.082, 0.068, -0.028, 0.013, -0.01]
    let esg_scores = [0.9, 0.548, 0.906, 0.552, 0.758, 0.578, 0.639, 0.941, 0.816, 0.982]
    let S = [[0.009, -0.001, -0.0, 0.0, 0.001, 0.001, 0.001, -0.001, -0.0, -0.0], [-0.001, 0.009, 0.001, -0.001, 0.0, -0.0, -0.001, -0.0, -0.0, -0.001], [-0.0, 0.001, 0.011, 0.0, 0.0, 0.0, 0.001, 0.001, -0.0, -0.001], [0.0, -0.001, 0.0, 0.011, -0.002, -0.001, -0.001, -0.0, -0.0, 0.001], [0.001, 0.0, 0.0, -0.002, 0.011, 0.0, 0.001, 0.001, -0.001, -0.0], [0.001, -0.0, 0.0, -0.001, 0.0, 0.009, 0.0, -0.001, -0.0, 0.001], [0.001, -0.001, 0.001, -0.001, 0.001, 0.0, 0.011, -0.0, 0.001, -0.0], [-0.001, -0.0, 0.001, -0.0, 0.001, -0.001, -0.0, 0.01, 0.001, 0.001], [-0.0, -0.0, -0.0, -0.0, -0.001, -0.0, 0.001, 0.001, 0.01, -0.001], [-0.0, -0.001, -0.001, 0.001, -0.0, 0.001, -0.0, 0.001, -0.001, 0.01]]
  
    let ones = replicate 10 1.0
    let neg_ones = map (f64.neg) ones
    let A = [mu, esg_scores, ones, neg_ones] ++ eye 10 :> [14][10]f64
    let b r e =  [r, e, 1, -1] ++ replicate 10 0.0 :> [14]f64
    let x_init = replicate 10 0.0
    let x_init[0] = 1.0
    let rho = 0.1
    let tol = 1e-8
    let max_iter = 5000
    let f0 x = (matmul (matmul [x] S) (transpose [x]))[0,0]
    let f0_grad = grad f0
    let f0_hess = hess f0
    let xs = (map (\e -> (map (\r -> admm f0 f0_grad f0_hess A (b r e) x_init rho tol max_iter) (range 0.0 0.07 10))) (range 0.8 0.9 10))
    in ( replicate 10 (range 0.0 0.07 10), map (replicate 10) (range 0.8 0.9 10), map (map f0) xs)
  
  -- let f0 (x: [10]f64) = (matmul (matmul [x] S) (transpose [x]))[0,0]
  -- let grad_f0 = grad f0
  -- let hess_f0 = hess f0

