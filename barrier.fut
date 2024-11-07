def dotprod [n] (xs: [n]f64) (ys: [n]f64) : f64 =
  reduce (+) 0.0 (map2 (*) xs ys)

def matmul [n] [p] [m] (xss: [n][p]f64) (yss: [p][m]f64) : [n][m]f64 =
  map (\xs -> map (dotprod xs) (transpose yss)) xss

def outer [n] [m] (xs: [n]f64) (ys: [m]f64) : *[n][m]f64 =
  map (\x -> map (\y -> x * y) ys) xs

def matsub [n] [m] (xss: [n][m]f64) (yss: [n][m]f64) : *[n][m]f64 =
  map2 (map2 (-)) xss yss

def identity (n: i64) : [n][n]f64 =
  tabulate_2d n n (\i j -> if j == i then 1f64 else 0f64)

def vecdiv_scalar [n] (xs: [n]f64) (k: f64) : *[n]f64 =
  map (/ k) xs

def forward_substitution [n] (L: [n][n]f64) (b: [n]f64) : [n]f64 =
  let y = replicate n 0.0f64
  in loop y for i in 0..<n do
       let sumy = dotprod L[i, :i] y[:i]
       let y[i] = (b[i] - sumy) / L[i, i]
       in y

def back_substitution [n] (U: [n][n]f64) (y: [n]f64) : [n]f64 =
  let x = replicate n 0.0f64
  in loop (x) for j in 0..<n do
       let i = n - j - 1
       let sumx = dotprod U[i, i + 1:n] x[i + 1:n]
       let x[i] = (y[i] - sumx) / U[i, i]
       in x

def lu [n] (A: [n][n]f64) : ([n][n]f64, [n][n]f64) =
  let L = identity n
  let (L, A) =
    loop (L, A) = (L, copy A) for k in 0..<(n) do
      let upper = vecdiv_scalar A[k + 1:n, k] A[k, k]
      let L[k + 1:n, k] = upper
      let A[k + 1:n, :] = matsub A[k + 1:n, :] (outer upper A[k, :])
      in (L, A)
  in (L, A)

def solveLUb [n] (L: [n][n]f64) (U: [n][n]f64) (b: [n]f64) =
  forward_substitution L b |> back_substitution U

def lu_solveAB [n] [m] (A: [n][n]f64) (B: [m][n]f64) =
  let (L, U) = lu A
  in map (solveLUb L U) B |> transpose

def lu_inv [n] (A: [n][n]f64) : [n][n]f64 =
  lu_solveAB A (identity n)

def eye (n: i64) : [n][n]f64 =
  let arr =
    map (\x ->
            let i = x / n
            let j = x % n
            in f64.bool (i == j))
        (iota (n * n))
  in unflatten arr

def h [n] (w: [n]f64) (mu: [n]f64) (esg: [n]f64) =
  let slack = 1e-6 -- slack variable
  let w_sum = reduce (+) 0 w
  let expected_return = dotprod w mu
  let expected_esg = dotprod w esg
  -- constraints
  in [
     -- sum(w) <= 1.00005                               
     w_sum - 1.0 - slack
     ,
     -- sum(w) >= 0.99995           
     1.0 - w_sum - slack
     ,
     -- expected return >= 0.07           
     0.07 - expected_return
     ,
     -- expected esg <= 0.8   
     0.8 - expected_esg
     ]
     ++
     -- w >= -0.0001
     map (f64.neg <-< (+ slack)) w

-- let h_grad [n] (w: [n]f64) (mu: [n]f64) (esg: [n]f64): [][n]f64 =
--     let grad = replicate (4 + n) (replicate n 0.0)
--     let grad[0, :] = replicate n 1.0
--     let grad[1, :] = replicate n (-1.0)
--     let grad[2, :] = map (f64.neg) mu
--     let grad[3, :] = map (f64.neg) esg
--     let grad[4:, :] = map (map f64.neg) (eye n)
--     in grad
-- let h_hess [n] (w: [n]f64) (mu: [n]f64) (esg: [n]f64): [][n][n]f64 =
--     replicate (n + 3) (replicate n (replicate n 0.0))

def phi [n] (w: [n]f64) (mu: [n]f64) (esg: [n]f64) =
  -(map (f64.log <-< f64.neg) (h w mu esg) |> reduce (+) 0.0)

def grad [n] (f: [n]f64 -> f64) (x: [n]f64) =
  map (\i ->
          let X = replicate n 0.0
          let X[i] = 1.0
          in jvp f x X)
      (iota n)

def hess [n] (f: [n]f64 -> f64) (x: [n]f64) =
  map (\i ->
          map (\j ->
                  let X = replicate n 0.0
                  let X[i] = 1.0
                  let Y = replicate n 0.0
                  let Y[j] = 1.0
                  in jvp (\w -> jvp f w X) x Y)
              (iota n))
      (iota n)

-- let phi_grad [n] (w: [n]f64) (mu: [n]f64) (esg: [n]f64): [n]f64 =
--     let h_values = h w mu esg
--     let h_len = length h_values
--     let grad_h_values = (h_grad w mu esg)
--     let grad_log_h = map (-1 /) h_values
--     in (matmul [grad_log_h :> [h_len]f64] (grad_h_values :> [h_len][n]f64))[0]
-- let phi_hess [n] (w: [n]f64) (mu: [n]f64) (esg: [n]f64) =
--     let h_values = h w mu esg
--     let grad_h_values = h_grad w mu esg
--     in map (\i ->
--         map (\j ->
--             let a = map (\x -> (1 / x ** 2)) h_values
--             let len = length a
--             let a = a :> [len]f64
--             let b = grad_h_values[:, i] :> [len]f64
--             let c = grad_h_values[:, j] :> [len]f64
--             in map3 (\x y z -> x * y * z) a b c |> reduce (+) 0.0
--         ) (iota n)
--     ) (iota n)

def quad [n] (w: [n]f64) (S: [n][n]f64) =
  (matmul (matmul [w] S) (transpose [w]))[0, 0]

-- let quad_grad [n] (w: [n]f64) (S: [n][n]f64): [n]f64 =
--     (map (*2.0) (flatten (matmul S (transpose [w])))):> [n]f64
-- let quad_hess [n] (w: [n]f64) (S: [n][n]f64): [n][n]f64 =
--     map (map (*2)) S

def f [n] (w: [n]f64) (S: [n][n]f64) (mu: [n]f64) (esg: [n]f64) (t: f64) =
  t * quad w S + phi w mu esg

-- let f_grad [n] (w: [n]f64) (S: [n][n]f64) (mu: [n]f64) (esg: [n]f64) (t: f64) =
--     map2 (+) (map (*t) (quad_grad w S)) (phi_grad w mu esg)

def f_grad_auto [n] (w: [n]f64) (S: [n][n]f64) (mu: [n]f64) (esg: [n]f64) (t: f64) =
  grad (\w -> f w S mu esg t) w

-- let f_hess [n] (w: [n]f64) (S: [n][n]f64) (mu: [n]f64) (esg: [n]f64) (t: f64) =
--     map (\i ->
--         map (\j ->
--             t * (quad_hess w S)[i, j] + (phi_hess w mu esg)[i, j]
--         ) (iota n)
--     ) (iota n)

def f_hess_auto [n] (w: [n]f64) (S: [n][n]f64) (mu: [n]f64) (esg: [n]f64) (t: f64) =
  hess (\w -> f w S mu esg t) w

-- let f_hessian_inv [n] (w: [n]f64) (S: [n][n]f64) (mu: [n]f64) (esg: [n]f64) (t: f64) =
--     lu_inv (f_hess w S mu esg t)

def f_hessian_inv_auto [n] (w: [n]f64) (S: [n][n]f64) (mu: [n]f64) (esg: [n]f64) (t: f64) =
  lu_inv (f_hess_auto w S mu esg t)

def argmax_f (x: f64, i: i64) (y: f64, j: i64) : (f64, i64) =
  if x > y
  then (x, i)
  else if y > x
  then (y, j)
  else if i < j
  then (x, i)
  else (y, j)

def argmax [n] (xs: [n]f64) : i64 =
  let (x, i) = reduce_comm argmax_f (xs[0], 0) (zip xs (iota n))
  in i

-- TODO: solve for w0
def find_feasible [n] (mu: [n]f64) (esg: [n]f64) =
  let max_mu = argmax mu
  let w0 = replicate n 0.0
  let w0[max_mu] = 1.0
  in w0

def newton [n] (w: [n]f64) (f: [n]f64 -> f64 -> f64) (f_grad: [n]f64 -> f64 -> [n]f64) (f_hessian_inv: [n]f64 -> f64 -> [n][n]f64) (t: f64) (beta: f64) (alpha: f64) (max_iter: i64) (eps: f64) =
  let (w, i, lambd, r) =
    loop (w, i, lambd, r) = (w, 0, 0.0, true) while i < max_iter && r do
      let fw = f w t
      let gradient = f_grad w t
      let hessian_inv = f_hessian_inv w t
      let v = map2 (-) (replicate n 0.0) (flatten (matmul hessian_inv (transpose [gradient])) :> [n]f64)
      let lambd = -1 * (map2 (*) gradient v |> reduce (+) 0.0)
      in if lambd / 2.0 <= eps
         then (w, i + 1, lambd, false)
         else let (new_w, new_f, s, running) =
                loop (new_w, new_f, s, running) = (w, fw, 1.0, true) while running do
                  let new_w = map2 (+) w (map (* s) v)
                  let new_f = f new_w t
                  in if new_f <= fw - alpha * s * lambd
                     then (new_w, new_f, s, false)
                     else (new_w, new_f, s * beta, true)
              in if fw == new_f
                 then (w, i + 1, lambd, false)
                 else (new_w, i + 1, lambd, true)
  let gradient = f_grad w t
  let hessian_inv = f_hessian_inv w t
  let v = map2 (-) (replicate n 0.0) (flatten (matmul hessian_inv (transpose [gradient])) :> [n]f64)
  let lambd = -1 * (map2 (*) gradient v |> reduce (+) 0.0)
  in (w, i, lambd)

def barrier [n] (w: [n]f64) (f: [n]f64 -> f64 -> f64) (f_grad: [n]f64 -> f64 -> [n]f64) (f_hessian_inv: [n]f64 -> f64 -> [n][n]f64) (beta: f64) (alpha: f64) (max_iter: i64) (eps: f64) (m: f64) =
  let (w, gap, iters, obj, t, r) =
    loop (w, gap, iters, obj, t, r) = (w, [], [], [], 1f64, true) while r do
      if (f64.i64 n) / t <= eps
      then (w, gap, iters, obj, t * m, false)
      else let (w, i, lambd) = newton w f f_grad f_hessian_inv t beta alpha max_iter eps
           let gap = gap ++ [n / 1]
           let iters = iters ++ [i]
           let obj = obj -- ++ [quad w S]
           in (w, gap, iters, obj, t * m, true)
  in (w, gap, iters, obj)

def beta = 0.5
def alpha = 0.01
def max_iter = 20i64
def eps = 1e-8
def mus = [1.5, 3, 5, 10]
def num_assets = 10i64
def num_days = 252

def mu = [0.073, -0.019, -0.02, -0.041, 0.039, -0.082, 0.068, -0.028, 0.013, -0.01]
def esg_scores = [0.9, 0.548, 0.906, 0.552, 0.758, 0.578, 0.639, 0.941, 0.816, 0.982]
def S = [[0.009, -0.001, -0.0, 0.0, 0.001, 0.001, 0.001, -0.001, -0.0, -0.0], [-0.001, 0.009, 0.001, -0.001, 0.0, -0.0, -0.001, -0.0, -0.0, -0.001], [-0.0, 0.001, 0.011, 0.0, 0.0, 0.0, 0.001, 0.001, -0.0, -0.001], [0.0, -0.001, 0.0, 0.011, -0.002, -0.001, -0.001, -0.0, -0.0, 0.001], [0.001, 0.0, 0.0, -0.002, 0.011, 0.0, 0.001, 0.001, -0.001, -0.0], [0.001, -0.0, 0.0, -0.001, 0.0, 0.009, 0.0, -0.001, -0.0, 0.001], [0.001, -0.001, 0.001, -0.001, 0.001, 0.0, 0.011, -0.0, 0.001, -0.0], [-0.001, -0.0, 0.001, -0.0, 0.001, -0.001, -0.0, 0.01, 0.001, 0.001], [-0.0, -0.0, -0.0, -0.0, -0.001, -0.0, 0.001, 0.001, 0.01, -0.001], [-0.0, -0.001, -0.001, 0.001, -0.0, 0.001, -0.0, 0.001, -0.001, 0.01]]

def w0 = find_feasible mu esg_scores

def f' = (\w t -> f w S mu esg_scores t)
def f_grad_auto' = (\w t -> f_grad_auto w S mu esg_scores t)
def f_hess_auto' = (\w t -> f_hess_auto w S mu esg_scores t)
def f_hessian_inv_auto' = (\w t -> f_hessian_inv_auto w S mu esg_scores t)

def main () =
  let (w, gap, iters, obj) = barrier w0 f' f_grad_auto' f_hessian_inv_auto' beta alpha max_iter eps 1.5
  in copy w
