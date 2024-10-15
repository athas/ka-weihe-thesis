import "linear"
import "lib/github.com/diku-dk/linalg/linalg"

module linear_f64 = mk_linear f64
module linalg = mk_linalg f64


entry bench_cholesky_outer = linear_f64.cholesky_outer
-- ==
-- entry: bench_cholesky_outer
-- input @ data/2d_100x100f64
-- input @ data/2d_500x500f64
-- input @ data/2d_1000x1000f64
-- input @ data/2d_5000x5000f64
-- input @ data/2d_10000x10000f64

entry bench_cholesky_crout = linear_f64.cholesky_crout
-- ==
-- entry: bench_cholesky_crout
-- input @ data/2d_100x100f64
-- input @ data/2d_500x500f64

-- (These are too slow)
-- input @ data/2d_1000x1000f64
-- input @ data/2d_5000x5000f64
-- input @ data/2d_10000x10000f64

entry bench_cholesky_banachiewicz = linear_f64.cholesky_banachiewicz
-- ==
-- entry: bench_cholesky_banachiewicz
-- input @ data/2d_100x100f64
-- input @ data/2d_500x500f64

-- (These are too slow)
-- input @ data/2d_1000x1000f64
-- input @ data/2d_5000x5000f64
-- input @ data/2d_10000x10000f64

entry bench_cholesky_flat = linear_f64.cholesky_flat
-- ==
-- entry: bench_cholesky_flat
-- input @ data/2d_100x100f64
-- input @ data/2d_500x500f64
-- input @ data/2d_1000x1000f64
-- input @ data/2d_5000x5000f64
-- input @ data/2d_10000x10000f64


entry bench_cholesky_dot = linear_f64.cholesky_dot
-- ==
-- entry: bench_cholesky_dot
-- input @ data/2d_100x100f64
-- input @ data/2d_500x500f64
-- input @ data/2d_1000x1000f64
-- input @ data/2d_5000x5000f64
-- input @ data/2d_10000x10000f64

-- Cholesky flat
-- linear_bench.fut:bench_cholesky_flat (no tuning file):
-- data/2d_100x100f64:                           2251μs (95% CI: [    2228.5,     2274.3])
-- data/2d_500x500f64:                          13116μs (95% CI: [   13079.6,    13139.6])
-- data/2d_1000x1000f64:                        28696μs (95% CI: [   28495.8,    29414.4])
-- data/2d_5000x5000f64:                       649778μs (95% CI: [  648471.9,   651037.2])
-- data/2d_10000x10000f64:                    4720386μs (95% CI: [ 4719673.6,  4721443.4])

-- Cholesky Crout
-- data/2d_100x100f64:         562549μs (95% CI: [  561570.4,   563715.6])
-- data/2d_500x500f64:       68469887μs (95% CI: [68467887.0, 68471887.0])

-- Cholesky Banachiewicz 
-- data/2d_100x100f64:         550646μs (95% CI: [  549729.9,   551555.2])
-- data/2d_500x500f64:       67851051μs (95% CI: [67849051.0, 67853051.0])

-- Cholesky Outer
-- linear_bench.fut:bench_cholesky (no tuning file):
-- data/2d_100x100f64:                           2295μs (95% CI: [    2289.2,     2301.1])
-- data/2d_500x500f64:                          11501μs (95% CI: [   11388.7,    11848.5])
-- data/2d_1000x1000f64:                        23247μs (95% CI: [   23244.0,    23249.2])
-- data/2d_5000x5000f64:                      1128661μs (95% CI: [ 1127855.8,  1129477.1])
-- data/2d_10000x10000f64:                    9725639μs (95% CI: [ 9712957.2,  9735290.8])

-- Cholesky Dot
-- data/2d_100x100f64:           2578μs (95% CI: [    2576.0,     2579.4])
-- data/2d_500x500f64:          13024μs (95% CI: [   12998.0,    13041.2])
-- data/2d_1000x1000f64:        28017μs (95% CI: [   28001.6,    28031.8])
-- data/2d_5000x5000f64:       267419μs (95% CI: [  267363.8,   267480.1])
-- data/2d_10000x10000f64:    1323244μs (95% CI: [ 1323166.1,  1323312.5])

-- NumPy:
-- n = 100, time = 63 us
-- n = 500, time = 1693 us
-- n = 1000, time = 7939 us
-- n = 5000, time = 499338 us
-- n = 10000, time = 2788463 us

entry bench_cholesky_flat_batched [n] (As: [][n][n]f64) =
    map (linear_f64.cholesky_flat) As
-- ==
-- entry: bench_cholesky_flat_batched
-- input @ data/2d_1000000x5x5f64
-- input @ data/2d_1000000x10x10f64
-- input @ data/2d_100000x20x20f64
-- input @ data/2d_10000x50x50f64
-- input @ data/2d_1000x100x100f64
-- input @ data/2d_100x500x500f64

entry bench_cholesky_crout_batched [n] (As: [][n][n]f64) =
    map (linear_f64.cholesky_crout) As
-- ==
-- entry: bench_cholesky_crout_batched
-- input @ data/2d_1000000x5x5f64
-- input @ data/2d_1000000x10x10f64
-- input @ data/2d_100000x20x20f64
-- input @ data/2d_10000x50x50f64
-- input @ data/2d_1000x100x100f64
-- input @ data/2d_100x500x500f64

entry bench_cholesky_banachiewicz_batched [n] (As: [][n][n]f64) =
    map (linear_f64.cholesky_banachiewicz) As
-- ==
-- entry: bench_cholesky_banachiewicz_batched
-- input @ data/2d_1000000x5x5f64
-- input @ data/2d_1000000x10x10f64
-- input @ data/2d_100000x20x20f64
-- input @ data/2d_10000x50x50f64
-- input @ data/2d_1000x100x100f64
-- input @ data/2d_100x500x500f64

entry bench_cholesky_outer_batched [n] (As: [][n][n]f64) =
    map (linear_f64.cholesky_outer) As
-- ==
-- entry: bench_cholesky_outer_batched
-- input @ data/2d_1000000x5x5f64
-- input @ data/2d_1000000x10x10f64
-- input @ data/2d_100000x20x20f64
-- input @ data/2d_10000x50x50f64
-- input @ data/2d_1000x100x100f64
-- input @ data/2d_100x500x500f64

--  Flat
-- linear_bench.fut:bench_cholesky_flat_batched (no tuning file):
-- data/2d_1000000x5x5f64:                       5265μs (95% CI: [    5263.4,     5268.3])
-- data/2d_1000000x10x10f64:                    45702μs (95% CI: [   45689.4,    45710.1])
-- data/2d_100000x20x20f64:                     35286μs (95% CI: [   35273.4,    35297.7])
-- data/2d_10000x50x50f64:                      56711μs (95% CI: [   56689.0,    56744.4])
-- data/2d_1000x100x100f64:                    240236μs (95% CI: [  240099.9,   240449.2])
-- data/2d_100x500x500f64:                   29102880μs (95% CI: [29100000.0, 29110000.0])

-- Crout
-- linear_bench.fut:bench_cholesky_crout_batched (no tuning file):
-- data/2d_1000000x5x5f64:         1942μs (95% CI: [    1940.9,     1943.7])
-- data/2d_1000000x10x10f64:      10400μs (95% CI: [   10396.6,    10402.7])
-- data/2d_100000x20x20f64:        4565μs (95% CI: [    4563.6,     4567.0])
-- data/2d_10000x50x50f64:        11628μs (95% CI: [   11622.9,    11632.6])
-- data/2d_1000x100x100f64:       43709μs (95% CI: [   43157.4,    44256.2])
-- data/2d_100x500x500f64:      7973984μs (95% CI: [ 7972406.6,  7975910.2])

-- Banachiewicz
-- linear_bench.fut:bench_cholesky_banachiewicz_batched (no tuning file):
-- data/2d_1000000x5x5f64:         1800μs (95% CI: [    1799.0,     1801.3])
-- data/2d_1000000x10x10f64:       9672μs (95% CI: [    9668.3,     9676.1])
-- data/2d_100000x20x20f64:        4254μs (95% CI: [    4253.0,     4255.7])
-- data/2d_10000x50x50f64:         6238μs (95% CI: [    6236.4,     6239.8])
-- data/2d_1000x100x100f64:       22206μs (95% CI: [   22177.7,    22233.2])
-- data/2d_100x500x500f64:      2105505μs (95% CI: [ 2104408.4,  2106600.5])

-- Outer
-- linear_bench.fut:bench_cholesky_batched (no tuning file):
-- data/2d_1000000x5x5f64:                       6495μs (95% CI: [    6494.5,     6496.4])
-- data/2d_1000000x10x10f64:                    60511μs (95% CI: [   60500.3,    60520.7])
-- data/2d_100000x20x20f64:                     51839μs (95% CI: [   51816.8,    51861.9])
-- data/2d_10000x50x50f64:                     131924μs (95% CI: [  131872.2,   131991.5])
-- data/2d_1000x100x100f64:                    537358μs (95% CI: [  537277.7,   537525.5])
-- data/2d_100x500x500f64:                   58620970μs (95% CI: [58619880.0, 58622060.0])

-- Dot
-- linear_bench.fut:bench_cholesky_vectorized_batched (no tuning file):
-- data/2d_1000000x5x5f64:         7152μs (95% CI: [    7150.1,     7153.9])
-- data/2d_1000000x10x10f64:      56095μs (95% CI: [   56078.3,    56114.2])
-- data/2d_100000x20x20f64:       42474μs (95% CI: [   42439.7,    42506.5])
-- data/2d_10000x50x50f64:       125089μs (95% CI: [  125007.9,   125184.7])
-- data/2d_1000x100x100f64:      533273μs (95% CI: [  533132.6,   533405.5])
-- data/2d_100x500x500f64:     58408621μs (95% CI: [58407621.0, 58409621.0])

-- NumPy:
-- n = 5x5, time = 1841378 us
-- n = 10x10, time = 2033527 us
-- n = 20x20, time = 325007 us
-- n = 50x50, time = 98811 us
-- n = 100x100, time = 85831 us
-- n = 500x500, time = 165973 us

entry bench_lud = linear_f64.lud
-- ==
-- entry: bench_lud
-- input @ data/2d_100x100f64
-- input @ data/2d_500x500f64
-- input @ data/2d_1000x1000f64
-- input @ data/2d_5000x5000f64
-- input @ data/2d_10000x10000f64

entry bench_seq_lud = linear_f64.seq_lud
-- ==
-- entry: bench_seq_lud
-- input @ data/2d_100x100f64
-- input @ data/2d_500x500f64
-- input @ data/2d_1000x1000f64
-- input @ data/2d_5000x5000f64
-- input @ data/2d_10000x10000f64

entry bench_blocked_lud = linear_f64.blocked_lud
-- ==
-- entry: bench_blocked_lud
-- input @ data/2d_100x100f64
-- input @ data/2d_500x500f64
-- input @ data/2d_1000x1000f64
-- input @ data/2d_5000x5000f64
-- input @ data/2d_10000x10000f64


-- linear_bench.fut:bench_lud (using linear_bench.fut.tuning):
-- data/2d_100x100f64:             1705μs (95% CI: [    1699.1,     1711.3])
-- data/2d_500x500f64:             9165μs (95% CI: [    9155.2,     9176.6])
-- data/2d_1000x1000f64:          20163μs (95% CI: [   20157.4,    20179.9])
-- data/2d_5000x5000f64:        1584932μs (95% CI: [ 1584587.0,  1585287.6])
-- data/2d_10000x10000f64:     11885985μs (95% CI: [11884785.0, 11887185.0])

-- linear_bench.fut:bench_seq_lud (no tuning file):
-- data/2d_100x100f64:           960931μs (95% CI: [  959647.0,   961962.5])
-- data/2d_500x500f64:        111392020µs (95% CI: [111381020.0, 111403020.0]) 

-- linear_bench.fut:bench_blocked_lud (no tuning file):
-- data/2d_100x100f64:             2356μs (95% CI: [    2334.5,     2432.8])
-- data/2d_500x500f64:             5958μs (95% CI: [    5956.7,     5960.1])
-- data/2d_1000x1000f64:          12995μs (95% CI: [   12993.1,    12997.7])
-- data/2d_5000x5000f64:         216774μs (95% CI: [  216744.9,   216798.1])
-- data/2d_10000x10000f64:      1112225μs (95% CI: [ 1112112.6,  1112392.2])

-- "Scipy"
-- n = 100x100, time = 27320 us
-- n = 500x500, time = 29859 us
-- n = 1000x1000, time = 41966 us
-- n = 5000x5000, time = 717490 us
-- n = 10000x10000, time = 4905548 us

entry bench_lud_batched [n] (As: [][n][n]f64) =
    map (linear_f64.lud) As
-- ==
-- entry: bench_lud_batched
-- input @ data/2d_1000000x5x5f64
-- input @ data/2d_1000000x10x10f64
-- input @ data/2d_100000x20x20f64
-- input @ data/2d_10000x50x50f64
-- input @ data/2d_1000x100x100f64
-- input @ data/2d_100x500x500f64

entry bench_seq_lud_batched [n] (As: [][n][n]f64) =
    map (linear_f64.seq_lud) As
-- ==
-- entry: bench_seq_lud_batched
-- input @ data/2d_1000000x5x5f64
-- input @ data/2d_1000000x10x10f64
-- input @ data/2d_100000x20x20f64
-- input @ data/2d_10000x50x50f64
-- input @ data/2d_1000x100x100f64
-- input @ data/2d_100x500x500f64

entry bench_blocked_lud_batched [n] (As: [][n][n]f64) =
    map (linear_f64.blocked_lud) As
-- ==
-- entry: bench_blocked_lud_batched
-- input @ data/2d_1000000x5x5f64
-- input @ data/2d_1000000x10x10f64
-- input @ data/2d_100000x20x20f64
-- input @ data/2d_10000x50x50f64
-- input @ data/2d_1000x100x100f64
-- input @ data/2d_100x500x500f64


-- linear_bench.fut:bench_lud_batched (no tuning file):
-- data/2d_1000000x5x5f64:         4237μs (95% CI: [    4234.6,     4239.3])
-- data/2d_1000000x10x10f64:      23685μs (95% CI: [   23671.9,    23698.3])
-- data/2d_100000x20x20f64:       15110μs (95% CI: [   15103.1,    15116.8])
-- data/2d_10000x50x50f64:        19187μs (95% CI: [   19166.7,    19198.0])
-- data/2d_1000x100x100f64:       14703μs (95% CI: [   14686.9,    14720.7])
-- data/2d_100x500x500f64:       161251μs (95% CI: [  161154.4,   161494.4])

-- linear_bench.fut:bench_seq_lud_batched (no tuning file):
-- data/2d_1000000x5x5f64:         2127μs (95% CI: [    2124.0,     2130.1])
-- data/2d_1000000x10x10f64:      12636μs (95% CI: [   12630.7,    12640.6])
-- data/2d_100000x20x20f64:        6033μs (95% CI: [    6028.4,     6036.7])
-- data/2d_10000x50x50f64:        19407μs (95% CI: [   19400.8,    19412.8])
-- data/2d_1000x100x100f64:      112546μs (95% CI: [  112345.2,   112783.1])
-- data/2d_100x500x500f64:     13716390μs

-- linear_bench.fut:bench_blocked_lud_batched (no tuning file):
-- data/2d_1000000x5x5f64:        16703μs (95% CI: [   16690.4,    16715.6])
-- data/2d_1000000x10x10f64:      42849μs (95% CI: [   42695.2,    43077.2])
-- data/2d_100000x20x20f64:       22863μs (95% CI: [   22765.5,    23223.5])
-- data/2d_10000x50x50f64:        26905μs (95% CI: [   26752.7,    27058.8])
-- data/2d_1000x100x100f64:        3654μs (95% CI: [    3652.9,     3655.3])
-- data/2d_100x500x500f64:        23886μs (95% CI: [   23873.3,    23906.8])

-- Scipy:
-- n = 5x5, time = 4980473 us
-- n = 10x10, time = 7842934 us
-- n = 20x20, time = 1141277 us
-- n = 50x50, time = 305864 us
-- n = 100x100, time = 415662 us
-- n = 500x500, time = 547019 us

entry bench_cholesky_solveAb = linear_f64.solve_Ab_cholesky_outer
-- ==
-- entry: bench_cholesky_solveAb
-- input @ data/2d_100x100_1d_100f64
-- input @ data/2d_500x500_1d_500f64
-- input @ data/2d_1000x1000_1d_1000f64
-- input @ data/2d_5000x5000_1d_5000f64
-- input @ data/2d_10000x10000_1d_10000f64

entry bench_seq_cholesky_solveAb = linear_f64.solve_Ab_cholesky_seq
-- ==
-- entry: bench_seq_cholesky_solveAb
-- input @ data/2d_100x100_1d_100f64
-- input @ data/2d_500x500_1d_500f64
-- input @ data/2d_1000x1000_1d_1000f64
-- input @ data/2d_5000x5000_1d_5000f64
-- input @ data/2d_10000x10000_1d_10000f64

entry bench_lu_solveAb = linear_f64.solve_Ab_lu
-- ==
-- entry: bench_lu_solveAb
-- input @ data/2d_100x100_1d_100f64
-- input @ data/2d_500x500_1d_500f64
-- input @ data/2d_1000x1000_1d_1000f64
-- input @ data/2d_5000x5000_1d_5000f64
-- input @ data/2d_10000x10000_1d_10000f64

entry bench_seq_lu_solveAb = linear_f64.solve_Ab_lu_seq
-- ==
-- entry: bench_seq_lu_solveAb
-- input @ data/2d_100x100_1d_100f64
-- input @ data/2d_500x500_1d_500f64
-- input @ data/2d_1000x1000_1d_1000f64
-- input @ data/2d_5000x5000_1d_5000f64
-- input @ data/2d_10000x10000_1d_10000f64

entry bench_gauss_solveAb = linear_f64.solve_Ab_gauss
-- ==
-- entry: bench_gauss_solveAb
-- input @ data/2d_100x100_1d_100f64
-- input @ data/2d_500x500_1d_500f64
-- input @ data/2d_1000x1000_1d_1000f64
-- input @ data/2d_5000x5000_1d_5000f64
-- input @ data/2d_10000x10000_1d_10000f64

entry bench_gauss_np_solveAb = linear_f64.solve_Ab_gauss_np
-- ==
-- entry: bench_gauss_np_solveAb
-- input @ data/2d_100x100_1d_100f64
-- input @ data/2d_500x500_1d_500f64
-- input @ data/2d_1000x1000_1d_1000f64
-- input @ data/2d_5000x5000_1d_5000f64
-- input @ data/2d_10000x10000_1d_10000f64

entry bench_conjugate_gradient [n] (A: [n][n]f64) (b: [n]f64) =
    -- let A = linalg.matmul A (transpose A)
    let x0 = replicate n 0.0
    in linear_f64.solve_Ab_cg A b x0 1e-6 n
-- ==
-- entry: bench_conjugate_gradient
-- input @ data/2d_100x100_1d_100f64
-- input @ data/2d_500x500_1d_500f64
-- input @ data/2d_1000x1000_1d_1000f64
-- input @ data/2d_5000x5000_1d_5000f64
-- input @ data/2d_10000x10000_1d_10000f64


-- linear_bench.fut:bench_seq_lu_solveAb (no tuning file):
-- data/2d_100x100_1d_100f64:           993593μs ⠼ (RSE of mean: 0.0394;    5 runs)


-- linear_bench.fut:bench_cholesky_solveAb (no tuning file):
-- data/2d_100x100_1d_100f64:             4019μs (95% CI: [    4009.4,     4033.0])
-- data/2d_500x500_1d_500f64:            20123μs (95% CI: [   20109.0,    20161.1])
-- data/2d_1000x1000_1d_1000f64:         43868μs (95% CI: [   43840.0,    43898.3])
-- data/2d_5000x5000_1d_5000f64:        351368μs (95% CI: [  351285.2,   351471.0])
-- data/2d_10000x10000_1d_10000f64:    1493065μs (95% CI: [ 1492814.6,  1493297.0])

-- linear_bench.fut:bench_lu_solveAb (no tuning file):
-- data/2d_100x100_1d_100f64:             4261μs (95% CI: [    4256.6,     4265.9])
-- data/2d_500x500_1d_500f64:            11726μs (95% CI: [   11721.7,    11734.1])
-- data/2d_1000x1000_1d_1000f64:         25118μs (95% CI: [   25095.2,    25138.6])
-- data/2d_5000x5000_1d_5000f64:        297169μs (95% CI: [  296887.6,   297351.4])
-- data/2d_10000x10000_1d_10000f64:    2543884μs (95% CI: [ 2543683.6,  2544077.1])

-- linear_bench.fut:bench_gauss_solveAb (no tuning file):
-- data/2d_100x100_1d_100f64:             5949μs (95% CI: [    5944.7,     5954.2])
-- data/2d_500x500_1d_500f64:            29988μs (95% CI: [   29737.1,    30226.6])
-- data/2d_1000x1000_1d_1000f64:         59787μs (95% CI: [   59710.9,    59881.5])
-- data/2d_5000x5000_1d_5000f64:       1687173μs (95% CI: [ 1686769.6,  1687768.9])
-- data/2d_10000x10000_1d_10000f64:   12111056μs (95% CI: [12108474.4, 12113856.0])

-- linear_bench.fut:bench_gauss_np_solveAb (no tuning file):
-- data/2d_100x100_1d_100f64:             1251μs (95% CI: [    1244.3,     1258.6])
-- data/2d_500x500_1d_500f64:             5717μs (95% CI: [    5707.8,     5730.5])
-- data/2d_1000x1000_1d_1000f64:         14226μs (95% CI: [   14221.6,    14241.7])
-- data/2d_5000x5000_1d_5000f64:       1508537μs (95% CI: [ 1508071.3,  1508964.1])
-- data/2d_10000x10000_1d_10000f64:   11739530μs

-- linear_bench.fut:bench_conjugate_gradient (no tuning file):
-- data/2d_100x100_1d_100f64:             5531μs (95% CI: [    5526.1,     5535.4])
-- data/2d_500x500_1d_500f64:            31212μs (95% CI: [   30766.8,    31434.6])
-- data/2d_1000x1000_1d_1000f64:         70790μs (95% CI: [   70499.7,    70967.4])
-- data/2d_5000x5000_1d_5000f64:       1090442μs (95% CI: [ 1088842.6,  1092523.6])
-- data/2d_10000x10000_1d_10000f64:    6805977μs (95% CI: [ 6803392.6,  6808561.6])

-- Numpy
-- n = 100x100, time = 414 us
-- n = 500x500, time = 1861 us
-- n = 1000x1000, time = 10192 us
-- n = 5000x5000, time = 595741 us
-- n = 10000x10000, time = 4100007 us

entry bench_cholesky_solveAb_batched = 
    map2 (linear_f64.solve_Ab_cholesky_outer)
-- == 
-- entry: bench_cholesky_solveAb_batched
-- input @ data/2d_1000000x5x5_1d_1000000x5f64
-- input @ data/2d_1000000x10x10_1d_1000000x10f64
-- input @ data/2d_100000x20x20_1d_100000x20f64
-- input @ data/2d_10000x50x50_1d_10000x50f64
-- input @ data/2d_1000x100x100_1d_1000x100f64
-- input @ data/2d_100x500x500_1d_100x500f64

entry bench_seq_cholesky_solveAb_batched = 
    map2 (linear_f64.solve_Ab_cholesky_seq)
-- == 
-- entry: bench_seq_cholesky_solveAb_batched
-- input @ data/2d_1000000x5x5_1d_1000000x5f64
-- input @ data/2d_1000000x10x10_1d_1000000x10f64
-- input @ data/2d_100000x20x20_1d_100000x20f64
-- input @ data/2d_10000x50x50_1d_10000x50f64
-- input @ data/2d_1000x100x100_1d_1000x100f64
-- input @ data/2d_100x500x500_1d_100x500f64

entry bench_lu_solveAb_batched = 
    map2 (linear_f64.solve_Ab_lu)
-- == 
-- entry: bench_lu_solveAb_batched
-- input @ data/2d_1000000x5x5_1d_1000000x5f64
-- input @ data/2d_1000000x10x10_1d_1000000x10f64
-- input @ data/2d_100000x20x20_1d_100000x20f64
-- input @ data/2d_10000x50x50_1d_10000x50f64
-- input @ data/2d_1000x100x100_1d_1000x100f64
-- input @ data/2d_100x500x500_1d_100x500f64

entry bench_seq_lu_solveAb_batched = 
    map2 (linear_f64.solve_Ab_lu_seq)
-- == 
-- entry: bench_seq_lu_solveAb_batched
-- input @ data/2d_1000000x5x5_1d_1000000x5f64
-- input @ data/2d_1000000x10x10_1d_1000000x10f64
-- input @ data/2d_100000x20x20_1d_100000x20f64
-- input @ data/2d_10000x50x50_1d_10000x50f64
-- input @ data/2d_1000x100x100_1d_1000x100f64
-- input @ data/2d_100x500x500_1d_100x500f64

entry bench_guass_solveAb_batched = 
    map2 (linear_f64.solve_Ab_gauss)
-- == 
-- entry: bench_guass_solveAb_batched
-- input @ data/2d_1000000x5x5_1d_1000000x5f64
-- input @ data/2d_1000000x10x10_1d_1000000x10f64
-- input @ data/2d_100000x20x20_1d_100000x20f64
-- input @ data/2d_10000x50x50_1d_10000x50f64
-- input @ data/2d_1000x100x100_1d_1000x100f64
-- input @ data/2d_100x500x500_1d_100x500f64

entry bench_guass_np_solveAb_batched = 
    map2 (linear_f64.solve_Ab_gauss_np)
-- == 
-- entry: bench_guass_np_solveAb_batched
-- input @ data/2d_1000000x5x5_1d_1000000x5f64
-- input @ data/2d_1000000x10x10_1d_1000000x10f64
-- input @ data/2d_100000x20x20_1d_100000x20f64
-- input @ data/2d_10000x50x50_1d_10000x50f64
-- input @ data/2d_1000x100x100_1d_1000x100f64
-- input @ data/2d_100x500x500_1d_100x500f64


entry bench_conjugate_gradient_batched [n] (As: [][n][n]f64) (bs: [][n]f64) =
    let x0 = replicate n 0.0
    in map2 (\A b -> linear_f64.solve_Ab_cg A b x0 1e-6 n) As bs
-- == 
-- entry: bench_conjugate_gradient_batched
-- input @ data/2d_1000000x5x5_1d_1000000x5f64
-- input @ data/2d_1000000x10x10_1d_1000000x10f64
-- input @ data/2d_100000x20x20_1d_100000x20f64
-- input @ data/2d_10000x50x50_1d_10000x50f64
-- input @ data/2d_1000x100x100_1d_1000x100f64
-- input @ data/2d_100x500x500_1d_100x500f64


-- linear_bench.fut:bench_cholesky_solveAb_batched (no tuning file):
-- data/2d_1000000x5x5_1d_1000000x5f64:         16490μs (95% CI: [   16485.8,    16494.3])
-- data/2d_1000000x10x10_1d_1000000x10f64:     134174μs (95% CI: [  134157.5,   134194.1])
-- data/2d_100000x20x20_1d_100000x20f64:       112757μs (95% CI: [  112737.9,   112781.5])
-- data/2d_10000x50x50_1d_10000x50f64:         128873μs (95% CI: [  128776.9,   128974.5])
-- data/2d_1000x100x100_1d_1000x100f64:        542807μs (95% CI: [  542563.9,   543033.2])
-- data/2d_100x500x500_1d_100x500f64:        76504121μs (95% CI: [76500000.0, 76510000.0])

-- linear_bench.fut:bench_seq_cholesky_solveAb_batched (no tuning file):
-- data/2d_1000000x5x5_1d_1000000x5f64:          3277μs (95% CI: [    3274.8,     3278.4])
-- data/2d_1000000x10x10_1d_1000000x10f64:      16538μs (95% CI: [   16530.4,    16546.3])
-- data/2d_100000x20x20_1d_100000x20f64:         6785μs (95% CI: [    6781.3,     6787.8])
-- data/2d_10000x50x50_1d_10000x50f64:          13612μs (95% CI: [   13601.7,    13623.4])
-- data/2d_1000x100x100_1d_1000x100f64:         47305μs (95% CI: [   47209.7,    47406.0])
-- data/2d_100x500x500_1d_100x500f64:         8005949μs (95% CI: [ 8005000.0,  8007000.0])

-- linear_bench.fut:bench_lu_solveAb_batched (no tuning file):
-- data/2d_1000000x5x5_1d_1000000x5f64:         22105μs (95% CI: [   22083.5,    22139.6])
-- data/2d_1000000x10x10_1d_1000000x10f64:     121299μs (95% CI: [  121097.4,   121510.9])
-- data/2d_100000x20x20_1d_100000x20f64:        55674μs (95% CI: [   55606.6,    55741.2])
-- data/2d_10000x50x50_1d_10000x50f64:          19744μs (95% CI: [   19735.2,    19751.0])
-- data/2d_1000x100x100_1d_1000x100f64:         16670μs (95% CI: [   16664.6,    16675.9])
-- data/2d_100x500x500_1d_100x500f64:          165624μs (95% CI: [  165481.6,   165752.4])

-- linear_bench.fut:bench_seq_lu_solveAb_batched (no tuning file):
-- data/2d_1000000x5x5_1d_1000000x5f64:          4287μs (95% CI: [    4275.3,     4297.6])
-- data/2d_1000000x10x10_1d_1000000x10f64:      23794μs (95% CI: [   23782.7,    23804.0])
-- data/2d_100000x20x20_1d_100000x20f64:        10142μs (95% CI: [   10137.6,    10145.9])
-- data/2d_10000x50x50_1d_10000x50f64:          22631μs (95% CI: [   22625.9,    22636.1])
-- data/2d_1000x100x100_1d_1000x100f64:        115883μs (95% CI: [  115746.2,   116078.2])
-- data/2d_100x500x500_1d_100x500f64:        13773200μs (95% CI: [13769290.1, 13778896.4])

-- linear_bench.fut:bench_guass_solveAb_batched (no tuning file):
-- data/2d_1000000x5x5_1d_1000000x5f64:          4136μs (95% CI: [    4134.2,     4137.2])
-- data/2d_1000000x10x10_1d_1000000x10f64:      30050μs (95% CI: [   30038.9,    30059.5])
-- data/2d_100000x20x20_1d_100000x20f64:        21799μs (95% CI: [   21792.0,    21807.3])
-- data/2d_10000x50x50_1d_10000x50f64:         134416μs (95% CI: [  134358.1,   134468.3])
-- data/2d_1000x100x100_1d_1000x100f64:        551759μs (95% CI: [  551638.9,   551908.5]) 
-- data/2d_100x500x500_1d_100x500f64:        59807991μs (95% CI: [59790000.0, 59820000.0])

-- linear_bench.fut:bench_guass_np_solveAb_batched (no tuning file):
-- data/2d_1000000x5x5_1d_1000000x5f64:          3498μs (95% CI: [    3497.6,     3499.4])
-- data/2d_1000000x10x10_1d_1000000x10f64:      18227μs (95% CI: [   18212.9,    18236.5])
-- data/2d_100000x20x20_1d_100000x20f64:        11890μs (95% CI: [   11880.0,    11906.3])
-- data/2d_10000x50x50_1d_10000x50f64:          16575μs (95% CI: [   16559.7,    16598.2])
-- data/2d_1000x100x100_1d_1000x100f64:         13325μs (95% CI: [   13304.8,    13347.5])
-- data/2d_100x500x500_1d_100x500f64:          155387μs (95% CI: [  155324.4,   155439.2])

-- linear_bench.fut:bench_conjugate_gradient_batched (no tuning file):
-- data/2d_1000000x5x5_1d_1000000x5f64:          2343μs (95% CI: [    2326.1,     2409.7])
-- data/2d_1000000x10x10_1d_1000000x10f64:      13164μs (95% CI: [   13161.9,    13167.6])
-- data/2d_100000x20x20_1d_100000x20f64:         9343μs (95% CI: [    9338.3,     9347.5])
-- data/2d_10000x50x50_1d_10000x50f64:           8986μs (95% CI: [    8978.8,     8991.9])
-- data/2d_1000x100x100_1d_1000x100f64:          9072μs (95% CI: [    9068.7,     9075.2])
-- data/2d_100x500x500_1d_100x500f64:          117261μs (95% CI: [  117208.0,   117370.0])

-- NumPy:
-- n = 5x5, time = 2397694 us
-- n = 10x10, time = 2734681 us
-- n = 20x20, time = 1036028 us
-- n = 50x50, time = 373783 us
-- n = 100x100, time = 140247 us
-- n = 500x500, time = 202333 us