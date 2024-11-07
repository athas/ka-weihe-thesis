import "lib/github.com/diku-dk/linalg/linalg"
module linalg = mk_linalg f64

def cholesky [n] (A: *[n][n]f64) : [n][n]f64 =
  -- Set upper triangular part to zero
  let A = map (\i -> map (\j -> if i < j then 0.0 else A[i, j]) (0..<n)) (0..<n)
  -- Process the remaining rows
  let A =
    loop A for j in 0..<n - 1 do
      let A[j, j] = f64.sqrt (A[j, j] - reduce (+) 0.0 (map (\x -> x * x) A[j, :j]))
      let Ajp1_j = A[j + 1:, :j]
      let A[j + 1:, j] = map (/ A[j, j]) (map2 (-) A[j + 1:, j] (linalg.matvecmul_row Ajp1_j A[j, :j]))
      in A
  -- Process the last row
  let A[n - 1, n - 1] = f64.sqrt (A[n - 1, n - 1] - reduce (+) 0.0 (map (\x -> x * x) A[n - 1, :n - 1]))
  in A

-- def cholesky_banachiewicz_flat(A):
--     n = A.shape[0]
--     # Create a 1D array to store only the lower-triangular values
--     L = np.zeros((n*(n+1))//2)
--     # 2D index to 1D index
--     idx = lambda i, j: i*(i+1)//2 + j
--     for j in range(0, n-1):
--         # Diagonal element
--         a = idx(j, 0)
--         b = idx(j, j)
--         Ljj = L[a:b]  # Pre-compute for multiple uses
--         L[idx(j, j)] = np.sqrt(A[j, j] - np.sum(Ljj**2))
--         vals = []
--         for i in range(j+1, n):
--             a = idx(i, 0)
--             b = idx(i, j)
--             Lji = L[a:b]  # Pre-compute for multiple uses
--             c = idx(j, 0)
--             d = idx(j, j)
--             Ljj = L[c:d]  # Pre-compute for multiple uses
--             sumval = Lji @ Ljj
--             val = (A[i, j] - sumval) / L[idx(j, j)]
--             vals.append(val)
--             # L[idx(i, j)] 
--         indices = []
--         for i in range(j+1, n):
--             indices.append(idx(i, j))
--         print(indices)
--         L[indices] = vals
--             # L[idx(i, j)] 
--     # Last diagonal element
--     Ljj = L[idx(n-1, 0):idx(n-1, n-1)]  # Pre-compute for multiple uses
--     L[idx(n-1, n-1)] = np.sqrt(A[n-1, n-1] - np.sum(Ljj**2))
--     return L
def cholesky_flat [n] (A: [n][n]f64) =
  let tri_num n = n * (n + 1) / 2
  let L = replicate (tri_num n) 0.0
  let L =
    loop L for j in 0..<n - 1 do
      let a = tri_num j
      let b = a + j
      let Ljj = L[a:b]
      let L[b] = f64.sqrt (A[j, j] - linalg.dotprod Ljj Ljj)
      let is = j + 1..<n
      let vals =
        map (\i ->
                let c = tri_num i
                let d = c + j
                let l = b - a
                let Ljj = L[a:b] :> [l]f64
                let Lji = L[c:d] :> [l]f64
                let sumval = linalg.dotprod Lji Ljj
                in (A[i, j] - sumval) / L[b])
            is
      let indices = map (\i -> tri_num i + j) is
      in scatter L indices vals
  let n1 = n - 1
  let a = tri_num n1
  let b = a + n1
  let Ljj = L[a:b]
  let L[b] = f64.sqrt (A[n1, n1] - reduce (+) 0.0 (map (\x -> x * x) Ljj))
  in L

def tri_num (n: i64) : i64 = n * (n + 1) / 2

def chole [n] (A: [n][n]f64) : [n][n]f64 =
  let L = cholesky_flat A
  in map (\i -> map (\j -> if i < j then 0.0 else L[tri_num i + j]) (0..<n)) (0..<n)

-- def cholesky_decomposition(A):
--     dimensionSize = len(A)
--     L = np.zeros_like(A)
--     for i in range(dimensionSize):
--         for j in range(i+1):
--             sum = 0
--             for k in range(j):
--                 sum += L[i, k] * L[j, k]
--             if i == j:
--                 # Diagonal elements
--                 L[i, j] = np.sqrt(A[i, i] - sum)
--             else:
--                 # Off-diagonal elements
--                 L[i, j] = (1.0 / L[j, j] * (A[i, j] - sum))
--     return L

def cholesky_banachiewicz [n] (A: [n][n]f64) : [n][n]f64 =
  let L = replicate n (replicate n 0.0)
  in loop L for i in 0..<n do
       loop L for j in 0...i do
         let sum =
           loop sum = 0.0 for k in 0..<j do
             L[i, k] * L[j, k] + sum
         in if i == j
            then let L[i, j] = f64.sqrt (A[i, i] - sum)
                 in L
            else let L[i, j] = (1.0 / L[j, j] * (A[i, j] - sum))
                 in L

def cholesky_crout [n] (A: [n][n]f64) : [n][n]f64 =
  let L = replicate n (replicate n 0.0)
  in loop L for j in 0..<n do
       let sum = loop sum = 0.0 for k in 0..<j do L[j, k] * L[j, k] + sum
       let L[j, j] = f64.sqrt (A[j, j] - sum)
       in loop L for i in j + 1..<n do
            let sum = loop sum = 0.0 for k in 0..<j do L[i, k] * L[j, k] + sum
            let L[i, j] = (1.0 / L[j, j]) * (A[i, j] - sum)
            in L

def tril [n] (A: [n][n]f64) : *[n][n]f64 =
  tabulate_2d n n (\i j -> if j <= i then A[i, j] else 0.0)

def outer [n] [m] (xs: [n]f64) (ys: [m]f64) : *[n][m]f64 =
  map (\x -> map (\y -> x * y) ys) xs

def matsub [n] [m] (xss: [n][m]f64) (yss: [n][m]f64) : *[n][m]f64 =
  map2 (map2 (-)) xss yss

def cho [n] (A: *[n][n]f64) : [n][n]f64 =
  tril
  <| loop A for j in 0..<n do
    let k = n - j - 1
    let A[j, j] = f64.sqrt A[j, j]
    let A[j + 1:, j] = map (/ A[j, j]) A[j + 1:n, j]
    let v = A[j + 1:n, j] :> *[k]f64
    let op = outer v v :> [k][k]f64
    let A[j + 1:n, j + 1:n] = matsub (A[j + 1:n, j + 1:n] :> [k][k]f64) op
    in A

entry flat_cho_bench_batch [n] (As: [][n][n]f64) =
  map (\A -> (chole A)) As

-- ==
-- entry: flat_cho_bench_batch
-- input @ 10000x10

def main =
  let A =
    [ [4.0, 12.0, -16.0]
    , [12.0, 37.0, -43.0]
    , [-16.0, -43.0, 98.0]
    ]
  in cholesky_banachiewicz A
