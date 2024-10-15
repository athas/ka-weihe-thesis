import "lib/github.com/diku-dk/linalg/linalg"
import "lib/github.com/diku-dk/linalg/lu"

local module type linear = {
    type t
    val forward_substitution     [n] : [n][n]t -> [n]t -> [n]t
    val seq_forward_substitution [n] : [n][n]t -> [n]t -> [n]t
    val back_substitution        [n] : [n][n]t -> [n]t -> [n]t
    val seq_back_substitution    [n] : [n][n]t -> [n]t -> [n]t
    val solveLUb                 [n] : [n][n]t -> [n][n]t -> [n]t -> [n]t
    val seq_solveLUb             [n] : [n][n]t -> [n][n]t -> [n]t -> [n]t
    val lud                      [n] : [n][n]t -> ([n][n]t, [n][n]t)
    val seq_lud                  [n] : [n][n]t -> ([n][n]t, [n][n]t)
    val blocked_lud              [n] : [n][n]t -> ([n][n]t, [n][n]t)
    val lu_inv                   [n] : [n][n]t -> [n][n]t
    val simple_lu_inv            [n] : [n][n]t -> [n][n]t
    val seq_lu_inv               [n] : [n][n]t -> [n][n]t
    val cholesky_crout           [n] : [n][n]t -> [n][n]t
    val cholesky_banachiewicz    [n] : [n][n]t -> [n][n]t
    val cholesky_outer                 [n] : [n][n]t -> [n][n]t
    val cholesky_flat            [n] : [n][n]t -> [n][n]t
    val cholesky_inv             [n] : [n][n]t -> [n][n]t
    val cholesky_dot      [n] : [n][n]t -> [n][n]t
    val gauss                    [n] : [n][n]t -> [n][n]t
    val gauss_np                 [n] : [n][n]t -> [n][n]t
    val gauss_inv                [n] : [n][n]t -> [n][n]t
    val inv_cholesky_inv         [n] : [n][n]t -> [n][n]t
    val cholesky_flat1d          [n] : [n][n]t -> []t
    val solve_Ab_gauss           [n] : [n][n]t -> [n]t -> [n]t
    val solve_Ab_cg              [n] : [n][n]t -> [n]t -> [n]t -> t -> i64 -> [n]t
    val solve_AB_lu_blocked      [n][m] : [n][n]t -> [m][n]t -> [n][m]t
    val solve_Ab_lu_blocked      [n] : [n][n]t -> [n]t -> [n]t
    val solve_Ab_lu_seq          [n] : [n][n]t -> [n]t -> [n]t
    val solve_AB_lu_seq          [n][m] : [n][n]t -> [m][n]t -> [n][m]t
    val solve_Ab_lu              [n] : [n][n]t -> [n]t -> [n]t
    val solve_AB_lu              [n][m] : [n][n]t -> [m][n]t -> [n][m]t
    val solve_Ab_cholesky_outer        [n] : [n][n]t -> [n]t -> [n]t
    val solve_AB_cholesky_outer        [n][m] : [n][n]t -> [m][n]t -> [n][m]t
    val solve_Ab_cholesky_seq    [n] : [n][n]t -> [n]t -> [n]t
    val solve_AB_cholesky_seq    [n][m] : [n][n]t -> [m][n]t -> [n][m]t
    val solve_Ab_cholesky_flat   [n] : [n][n]t -> [n]t -> [n]t
    val solve_Ab_gauss_np        [n] : [n][n]t -> [n]t -> [n]t
    val solve_AB_gauss           [n][m] : [m][m]t -> [m][n]t -> [m][n]t
    val solve_Ab_cholesky_dot    [n] : [n][n]t -> [n]t -> [n]t
}

module mk_linear (T: real) : linear with t = T.t = {
    type t = T.t
    module linalg = mk_linalg T
    module lu     = mk_lu T
    
    def seq_dotprod [n] (a: [n]t) (b: [n]t): t =
        loop sum = (T.f64 0.0) for i in 0..<n do
            a[i] T.* b[i] T.+ sum

    def forward_substitution [n] (L: [n][n]t) (b: [n]t): [n]t =
        let y = linalg.veczeros n
        in loop y for i in 0..<n do
            let sumy = linalg.dotprod L[i,:i] y[:i]
            let y[i] = copy (b[i] T.- sumy) T./ L[i,i]
            in y

    def seq_forward_substitution [n] (L: [n][n]t) (b: [n]t): [n]t =
        let y = linalg.veczeros n
        in loop y for i in 0..<n do
            let sumy = seq_dotprod L[i,:i] y[:i]
            let y[i] = copy (b[i] T.- sumy) T./ L[i,i]
            in y

    def back_substitution [n] (U: [n][n]t) (y: [n]t): [n]t =
        let x = linalg.veczeros n
        in loop (x) for j in 0..<n do
            let i = n - j - 1
            let sumx = linalg.dotprod U[i,i+1:n] x[i+1:n]
            let x[i] = copy (y[i] T.- sumx) T./ U[i,i]
            in x

    def seq_back_substitution [n] (U: [n][n]t) (y: [n]t): [n]t =
        let x = linalg.veczeros n
        in loop (x) for j in 0..<n do
            let i = n - j - 1
            let sumx = seq_dotprod U[i,i+1:n] x[i+1:n]
            let x[i] = copy (y[i] T.- sumx) T./ U[i,i]
            in x 

    def solveLUb [n] (L: [n][n]t) (U: [n][n]t) (b: [n]t) =
        forward_substitution L b |> back_substitution U 

    def seq_solveLUb [n] (L: [n][n]t) (U: [n][n]t) (b: [n]t) =
        seq_forward_substitution L b |> seq_back_substitution U

    def seq_lud [n] (A: [n][n]t): ([n][n]t, [n][n]t) =
        let L = linalg.eye n
        let U = copy A
        in loop (L, U) for i in 0..<n do
            let (L, U) = loop (L, U) for j in (i+1)..<n do
                let Lji = copy (U[j,i] T./ U[i,i])
                let L[j,i] = Lji
                let (L, U) = loop (L, U) for k in i..<n do
                    let Ujk = U[j,k] T.- (Lji T.* U[i,k])
                    let U[j,k] = copy Ujk
                    in (L, U)
                in (L, U)
            in (L, U)                

    def lud [n] (A: [n][n]t): ([n][n]t, [n][n]t) =
        let L = linalg.eye n
        let U = copy A
        in loop (L, U) for j in 0..<(n-1) do
            let factor = map (T./ U[j, j]) U[j+1:, j]
            let L[j+1:, j] = factor
            let U[j+1:] = map (\i -> 
                map2 (T.-) U[j+1+i] (map (T.*factor[i]) U[j])
            ) (0..<(n - j - 1))
        in (L, U)

    def blocked_lud [n] (A: [n][n]t): ([n][n]t, [n][n]t) =
        let block_size = i64.max 1 (i64.min 32 (n / 4))
        in lu.lu2 block_size A

    def solve_Ab_lu_blocked [n] (A: [n][n]t) (b: [n]t): [n]t =
        let (L, U) = blocked_lud A
        in solveLUb L U b

    def solve_AB_lu_blocked [n][m] (A: [n][n]t) (B: [m][n]t) : [n][m]t =
        let (L, U) = blocked_lud A
        in map (solveLUb L U) B |> transpose

    def lu_inv [n] (A: [n][n]t): [n][n]t =
        solve_AB_lu_blocked A (linalg.eye n)

    def solve_Ab_lu [n] (A: [n][n]t) (b: [n]t): [n]t =
        let (L, U) = lud A
        in solveLUb L U b

    def solve_AB_lu [n][m] (A: [n][n]t) (B: [m][n]t) : [n][m]t =
        let (L, U) = lud A
        in map (solveLUb L U) B |> transpose

    def simple_lu_inv [n] (A: [n][n]t): [n][n]t =
        solve_AB_lu A (linalg.eye n)

    def solve_Ab_lu_seq [n] (A: [n][n]t) (b: [n]t): [n]t =
        let (L, U) = seq_lud A
        in seq_solveLUb L U b
    
    def solve_AB_lu_seq [n][m] (A: [n][n]t) (B: [m][n]t) : [n][m]t =
        let (L, U) = seq_lud A
        in map (seq_solveLUb L U) B |> transpose
    
    def seq_lu_inv [n] (A: [n][n]t): [n][n]t =
        solve_AB_lu_seq A (linalg.eye n)

    def cholesky_crout [n] (A: [n][n]t): [n][n]t =
        let L = linalg.matzeros n n
        in loop L for j in 0..<n do
            let sum = seq_dotprod L[j,:j] L[j,:j]
            let L[j,j] = T.sqrt (copy (A[j,j] T.- sum))
            in loop L for i in j+1..<n do
                let sum = seq_dotprod L[i,:j] L[j,:j]
                let L[i,j] = (copy ((T.f64 1.0) T./ L[j,j])) T.* (copy (A[i,j] T.- sum))
                in L

    def cholesky_banachiewicz [n] (A: [n][n]t): [n][n]t =
        let L = linalg.matzeros n n
        in loop L for i in 0..<n do
            loop L for j in 0...i do
                let sum = seq_dotprod L[i,:j] L[j,:j] 
                in if i == j then
                    let L[i,j] = T.sqrt (copy (A[i,i] T.- sum))
                    in L
                else
                    let L[i,j] = (copy ((T.f64 1.0) T./ L[j,j]) T.* (copy (A[i,j] T.- sum)))
                    in L

    def cholesky_dot [n] (A: [n][n]t): [n][n]t =
        -- Set upper triangular part to zero
        let A = map (\i -> map (\j -> if i < j then (T.f64 0.0) else A[i, j]) (0..<n)) (0..<n) 
        -- Process the remaining rows
        let A = loop A for j in 0..<n-1 do 
            let A[j, j] = T.sqrt(copy (A[j, j] T.- reduce (T.+) (T.f64 0.0) (map (\x -> x T.* x)  A[j, :j])))
            let Ajp1_j = A[j+1:, :j]
            let A[j+1:, j] = map (T./A[j, j]) (map2 (T.-) A[j+1:, j] (linalg.matvecmul_row Ajp1_j A[j, :j]))
            in A
        -- Process the last row
        let A[n-1, n-1] = T.sqrt(copy(A[n-1, n-1] T.- reduce (T.+) (T.f64 0.0) (map (\x -> x T.* x) A[n-1, :n-1])))
        in A

    def tril [n] (A: [n][n]t): [n][n]t =
        tabulate_2d n n (\i j -> if j <= i then A[i, j] else (T.f64 0.0))

    def outer [n][m] (xs: [n]t) (ys: [m]t): [n][m]t =
        map (\x -> map (\y -> x T.* y) ys) xs 

    let cholesky_outer [n] (A: [n][n]t): [n][n]t = 
        let A = loop A = copy A for j in 0..<n do
            let m = n - j - 1
            let A[j,j] = T.sqrt (copy A[j,j])
            let A[j+1:,j] = map (T./A[j,j]) A[j+1:n,j]
            let v = A[j+1:n,j] :> [m]t
            let outer_product = outer v v :> [m][m]t
            let mat = A[j+1:n,j+1:n] :> [m][m]t
            let A[j+1:n,j+1:n] = linalg.matsub mat outer_product
            in A
        in tril A

    def tri_num (n: i64): i64 = n * (n + 1) / 2
    def index_to_ij (k: i64) (n: i64): (i16, i16) =
        let kp = n * (n + 1) / 2 - k - 1
        let p = i64.f64 ((f64.sqrt (f64.i64 (1 + 8 * kp)) - 1) / 2)
        let i = n - (kp - p * (p + 1) / 2) - 1
        let j = n - p - 1
        in (i16.i64 i, i16.i64 j)

    def tril_indices (n: i64): *[](i16, i16) =
        let l = (tri_num n)
        in map (\k -> index_to_ij k n) (iota l)

    def low_outer (v: []t) (is: [](i16, i16)): *[]t =
        map (\ x -> 
            let (i, j) = x
            in #[unsafe] v[i] T.* v[j]
        ) is

    def cholesky_flat1d [n] (A: [n][n]t): []t =
        let t_i = tril_indices n
        let l = length t_i 
        let A = map (\(i, j) -> A[i, j]) t_i 
        let m = tri_num n
        let A = loop A for j in 0..<n-1 do  
            let di = m - (tri_num (n - j))
            let sdi = T.sqrt (copy A[di])
            let A[di] = sdi
            let st = n - j - 1 
            let st_i = di + 1
            let end = st_i + st
            let A[st_i:end] = map (T./sdi) A[st_i:end]
            let s_c = j + 1
            let r = n - s_c 
            let k = tri_num r
            let o_i = t_i[l-k:]
            let o_i = map (\(a, b) -> let (c, d) = o_i[0] in (a - c, b - d)) o_i
            let c = A[st_i:end] :> *[st]t
            let o = (low_outer c o_i) :> [k]t
            let A[m-k:] = map2 (\x y -> x T.- y) (A[m-k:] :> [k]t) o 
            in A
        let di = m - 1
        let A[di] = T.sqrt (copy A[di])
        in A

    -- def cholesky_flat [n] (A: [n][n]t) =
    --     let L = linalg.veczeros (tri_num n)
    --     let L = loop L for j in 0..<n-1 do
    --         let a = tri_num j
    --         let b = a + j
    --         let Ljj = L[a:b]
    --         let L[b] = T.sqrt (copy (A[j,j] T.- linalg.dotprod Ljj Ljj))
    --         let is = j+1..<n
    --         let vals = map (\i -> 
    --             let c = tri_num i
    --             let d = c + j
    --             let l = b - a
    --             let Ljj = L[a:b] :> [l]t
    --             let Lji = L[c:d] :> [l]t
    --             let sumval = linalg.dotprod Lji Ljj
    --             in (A[i,j] T.- sumval) T./ L[b]
    --         ) is
    --         let indices = map (\i -> tri_num i + j) is
    --         in scatter L indices vals
    --     let n1 = n-1
    --     let a = tri_num n1
    --     let b = a + n1
    --     let Ljj = L[a:b]
    --     let L[b] = T.sqrt (A[n1,n1] T.- reduce (T.+) (T.f64 0.0) (map (\x -> x T.* x) Ljj))
    --     in L

    def i_j_to_index (i: i64) (j: i64) (n: i64): i64 =
        let p = n - j - 1
        let kp = p * (p + 1) / 2 + (n - i - 1)
        let k = n * (n + 1) / 2 - kp
        in k - 1

    def cholesky_flat [n] A: [n][n]t =
        let L = cholesky_flat1d A
        in map (\i ->
            map (\j ->
                if i < j then (T.f64 0.0) else L[i_j_to_index i j n]
            ) (0..<n)
        ) (0..<n)

    -- def cholesky [n] (A:[n][n]t): [n][n]t =
    --     let L = cholesky_flat A
    --     in map (\i -> map (\j -> 
    --         if i < j then (T.f64 0.0) else L[tri_num i + j]
    --     ) (0..<n)) (0..<n)

    def solve_Ab_cholesky_flat [n] (A: [n][n]t) (b: [n]t): [n]t =
        let L = cholesky_flat A
        let U = transpose L
        in solveLUb L U b

    def solve_Ab_cholesky_outer [n] (A: [n][n]t) (b: [n]t): [n]t =
        let L = cholesky_outer A
        let U = transpose L
        in solveLUb L U b

    def solve_Ab_cholesky_dot [n] (A: [n][n]t) (b: [n]t): [n]t =
        let L = cholesky_outer A
        let U = transpose L
        in solveLUb L U b


    def solve_AB_cholesky_outer [n][m] (A: [n][n]t) (B: [m][n]t) : [n][m]t =
        let L = cholesky_outer A
        let U = transpose L
        in map (solveLUb L U) B |> transpose

    def solve_AB_cholesky_flat [n][m] (A: [n][n]t) (B: [m][n]t) : [n][m]t =
        let L = cholesky_flat A
        let U = transpose L
        in map (solveLUb L U) B |> transpose

    def cholesky_inv [n] (A: [n][n]t): [n][n]t =
        solve_AB_cholesky_outer A (linalg.eye n)


    def solve_Ab_cholesky_seq [n] (A: [n][n]t) (b: [n]t): [n]t =
        let L = cholesky_crout A
        let U = transpose L
        in seq_solveLUb L U b

    def solve_AB_cholesky_seq [n][m] (A: [n][n]t) (B: [m][n]t) : [n][m]t =
        let L = cholesky_crout A
        let U = transpose L
        in map (seq_solveLUb L U) B |> transpose

    def inv_cholesky_inv [n] (A: [n][n]t): [n][n]t =
        solve_AB_cholesky_seq A (linalg.eye n)

    def argmax arr =
        reduce_comm (\(a,i) (b,j) ->
            if a T.< b
            then (b,j)
            else if b T.< a then (a,i)
            else if j < i then (b, j)
            else (a, i)
        ) (T.i64 0, 0) (zip arr (indices arr))

    def swap [n] 't (i: i64) (j: i64) (xs: *[n]t) =
        let xs_i = copy xs[i]
        let xs_j = copy xs[j]
        let xs[i] = xs_j
        let xs[j] = xs_i
        in xs

    def gauss [m] [n] (A:[m][n]t) =
        loop A = copy A for i < i64.min m n do
        -- Find largest pivot
        let p = A[i:,i] |> map T.abs |> argmax |> (.1) |> (+i)
            let A = if p != i then swap i p A else A
            let irow = map (T./A[i,i]) A[i]
            in tabulate m (\j ->
                let scale = A[j,i]
                in map2 (\x y -> if j != i then y T.- scale T.* x else x) irow A[j]
            )


    def gauss_np [m][n] (A:[m][n]t): [m][n]t =
        loop A = copy A for i < i64.min m n do
            let irow = map (T./A[i,i]) A[i]
            in tabulate m (\j ->
                let scale = A[j,i]
                in map2 (\x y -> if j != i then y T.- scale T.* x else x) irow A[j]
            )

    def hstack [m][n][l] (A:[m][n]t) (B:[m][l]t) = 
        map2 (concat_to (n+l)) A B

    def solve_AB_gauss [m][n] (A:[m][m]t) (B:[m][n]t) : [m][n]t =
        let AB = hstack A B |> gauss
        in AB[:m, m:] :> [m][n]t

    def solve_Ab_gauss [m] (A:[m][m]t) (b:[m]t) =
        unflatten m 1 b |> solve_AB_gauss A |> flatten_to m

    def gauss_inv [n] (A: [n][n]t): [n][n]t =
        solve_AB_gauss A (linalg.eye n)

    def gauss_np_solveAB [m][n] (A:[m][m]t) (B:[m][n]t) : [m][n]t =
        let AB = hstack A B |> gauss_np
        in AB[:m, m:] :> [m][n]t

    def solve_Ab_gauss_np [m] (A:[m][m]t) (b:[m]t) =
        unflatten m 1 b |> gauss_np_solveAB A |> flatten_to m

    def gauss_np_inv [n] (A: [n][n]t): [n][n]t =
        gauss_np_solveAB A (linalg.eye n)

    def solve_Ab_cg [n] (A: [n][n]t) (b: [n]t) (x0: [n]t) (tol: t) (max_iter: i64) =
        let r = map2 (T.-) b (linalg.matvecmul_row A x0)
        let rs_old = linalg.dotprod r r
        let p = r
        let x = x0
        let i = 0
        let running = true
        let res = loop (x, p, r, rs_old, i, running) while running && i < max_iter do
            let i = i + 1
            let Ap = linalg.matvecmul_row A p
            let alpha = rs_old T./ linalg.dotprod p Ap
            let x = map2 (T.+) x (map (alpha T.*) p)
            let r = map2 (T.-) r (map (alpha T.*) Ap)
            let rs_new = linalg.dotprod r r
            in if T.sqrt rs_new T.< tol then
                (x, p, r, rs_old, i, false)
            else 
                let p = map2 (T.+) r (map ((rs_new T./ rs_old)T.*) p)
                in (x, p, r, rs_new, i, running)
        in res.0
}
