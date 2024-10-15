import "lib/github.com/diku-dk/linalg/linalg"
import "linear"

module linalg = mk_linalg f64
module linear = mk_linear f64


let all_close [n] (xs: [n]f64) (ys: [n]f64) (eps: f64): bool =
  let diffs = map2 (-) xs ys
  in reduce (&&) true (map (\x -> f64.abs x < eps) diffs)

let all_close2d [n][m] (xss: [n][m]f64) (yss: [n][m]f64) (eps: f64): bool =
    reduce (&&) true (map2 (\xs ys -> all_close xs ys eps) xss yss)

entry test_gauss_np = linear.gauss_np
-- ==
-- entry: test_gauss_np
-- input {[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]]}
-- output {[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]}

entry test_gauss = linear.gauss
-- ==
-- entry: test_gauss
-- input {[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]]}
-- output {[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]}

entry seq_lu_test = 
    let A = [
        [2, -3, 4],
        [6, -8, 7],
        [1, 1, 1]]
    let (L, U) = linear.seq_lud A
    let L_expected = [
        [1.0, 0.0, 0.0],
        [3.0, 1.0, 0.0],
        [0.5, 2.5, 1.0]]
    let U_expected = [
        [2.0, -3.0, 4.0],
        [0.0, 1.0, -5.0],
        [0.0, 0.0, 11.5]]
    let l_check = all_close2d L L_expected 1e-10
    let u_check = all_close2d U U_expected 1e-10
    in l_check && u_check

-- ==
-- entry: seq_lu_test
-- input {}
-- output {true}

entry lu_test = 
    let A = [
        [2, -3, 4],
        [6, -8, 7],
        [1, 1, 1]]
    let (L, U) = linear.lud A
    let L_expected = [
        [1.0, 0.0, 0.0],
        [3.0, 1.0, 0.0],
        [0.5, 2.5, 1.0]]
    let U_expected = [
        [2.0, -3.0, 4.0],
        [0.0, 1.0, -5.0],
        [0.0, 0.0, 11.5]]
    let l_check = all_close2d L L_expected 1e-10
    let u_check = all_close2d U U_expected 1e-10
    in l_check && u_check

-- ==
-- entry: lu_test
-- input {}
-- output {true}

entry solve_lu_test = linear.solve_Ab_lu
-- ==
-- entry: solve_lu_test
-- input {[[2f64, -3f64, 4f64], [6f64, -8f64, 7f64], [1f64, 1f64, 1f64]] [1f64, 2f64, 3f64]}
-- output {[1.39130435f64, 1.17391304f64, 0.43478261f64]}

entry solve_lu_seq = linear.solve_Ab_lu_seq
-- ==
-- entry: solve_lu_seq
-- input {[[2f64, -3f64, 4f64], [6f64, -8f64, 7f64], [1f64, 1f64, 1f64]] [1f64, 2f64, 3f64]}
-- output {[1.39130435f64, 1.17391304f64, 0.43478261f64]}

entry solve_lu_blocked_test = linear.solve_Ab_lu_blocked
-- ==
-- entry: solve_lu_blocked_test
-- input {[[4f64, 12f64, -16f64], [12f64, 37f64, -43f64], [-16f64, -43f64, 98f64]] [1f64, 2f64, 3f64]}
-- output {[28.58333333f64, -7.66666667f64,  1.33333333f64]}

entry solve_cholesky_outer_test = linear.solve_Ab_cholesky_outer
-- ==
-- entry: solve_cholesky_outer_test
-- input {[[4f64, 12f64, -16f64], [12f64, 37f64, -43f64], [-16f64, -43f64, 98f64]] [1f64, 2f64, 3f64]}
-- output {[28.58333333f64, -7.66666667f64,  1.33333333f64]}

entry solve_cholesky_dot_test = linear.solve_Ab_cholesky_dot
-- ==
-- entry: solve_cholesky_dot_test
-- input {[[4f64, 12f64, -16f64], [12f64, 37f64, -43f64], [-16f64, -43f64, 98f64]] [1f64, 2f64, 3f64]}
-- output {[28.58333333f64, -7.66666667f64,  1.33333333f64]}

entry solve_cholesky_seq_test = linear.solve_Ab_cholesky_seq
-- ==
-- entry: solve_cholesky_seq_test
-- input {[[4f64, 12f64, -16f64], [12f64, 37f64, -43f64], [-16f64, -43f64, 98f64]] [1f64, 2f64, 3f64]}
-- output {[28.58333333f64, -7.66666667f64,  1.33333333f64]}

entry solve_cholesky_flat_test = linear.solve_Ab_cholesky_flat
-- ==
-- entry: solve_cholesky_flat_test
-- input {[[4f64, 12f64, -16f64], [12f64, 37f64, -43f64], [-16f64, -43f64, 98f64]] [1f64, 2f64, 3f64]}
-- output {[28.58333333f64, -7.66666667f64,  1.33333333f64]}

entry solve_Ab_gauss_test = linear.solve_Ab_gauss
-- ==
-- entry: solve_Ab_gauss_test
-- input {[[4f64, 12f64, -16f64], [12f64, 37f64, -43f64], [-16f64, -43f64, 98f64]] [1f64, 2f64, 3f64]}
-- output {[28.58333333f64, -7.66666667f64,  1.33333333f64]}

entry solve_Ab_gauss_np_test = linear.solve_Ab_gauss_np
-- ==
-- entry: solve_Ab_gauss_np_test
-- input {[[4f64, 12f64, -16f64], [12f64, 37f64, -43f64], [-16f64, -43f64, 98f64]] [1f64, 2f64, 3f64]}
-- output {[28.58333333f64, -7.66666667f64,  1.33333333f64]}