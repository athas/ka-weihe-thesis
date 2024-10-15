#!/bin/bash

# Create directories if they don't exist
mkdir -p data

# Generate datasets
futhark dataset -b --f64-bounds=-1:1 -g [100][100]f64 > data/2d_100x100f64
futhark dataset -b --f64-bounds=-1:1 -g [500][500]f64 > data/2d_500x500f64
futhark dataset -b --f64-bounds=-1:1 -g [1000][1000]f64 > data/2d_1000x1000f64
futhark dataset -b --f64-bounds=-1:1 -g [5000][5000]f64 > data/2d_5000x5000f64
futhark dataset -b --f64-bounds=-1:1 -g [10000][10000]f64 > data/2d_10000x10000f64

futhark dataset -b --f64-bounds=-1:1 -g [1000000][5][5]f64 > data/2d_1000000x5x5f64
futhark dataset -b --f64-bounds=-1:1 -g [1000000][10][10]f64 > data/2d_1000000x10x10f64
futhark dataset -b --f64-bounds=-1:1 -g [100000][20][20]f64 > data/2d_100000x20x20f64
futhark dataset -b --f64-bounds=-1:1 -g [10000][50][50]f64 > data/2d_10000x50x50f64
futhark dataset -b --f64-bounds=-1:1 -g [1000][100][100]f64 > data/2d_1000x100x100f64
futhark dataset -b --f64-bounds=-1:1 -g [100][500][500]f64 > data/2d_100x500x500f64

futhark dataset -b --f64-bounds=-1:1 -g [100][100]f64 -g [100]f64 > data/2d_100x100_1d_100f64
futhark dataset -b --f64-bounds=-1:1 -g [500][500]f64 -g [500]f64 > data/2d_500x500_1d_500f64
futhark dataset -b --f64-bounds=-1:1 -g [1000][1000]f64 -g [1000]f64 > data/2d_1000x1000_1d_1000f64
futhark dataset -b --f64-bounds=-1:1 -g [5000][5000]f64 -g [5000]f64 > data/2d_5000x5000_1d_5000f64
futhark dataset -b --f64-bounds=-1:1 -g [10000][10000]f64 -g [10000]f64 > data/2d_10000x10000_1d_10000f64

futhark dataset -b --f64-bounds=-1:1 -g [1000000][5][5]f64 -g [1000000][5]f64 > data/2d_1000000x5x5_1d_1000000x5f64
futhark dataset -b --f64-bounds=-1:1 -g [1000000][10][10]f64 -g [1000000][10]f64 > data/2d_1000000x10x10_1d_1000000x10f64
futhark dataset -b --f64-bounds=-1:1 -g [100000][20][20]f64 -g [100000][20]f64 > data/2d_100000x20x20_1d_100000x20f64
futhark dataset -b --f64-bounds=-1:1 -g [10000][50][50]f64 -g [10000][50]f64 > data/2d_10000x50x50_1d_10000x50f64
futhark dataset -b --f64-bounds=-1:1 -g [1000][100][100]f64 -g [1000][100]f64 > data/2d_1000x100x100_1d_1000x100f64
futhark dataset -b --f64-bounds=-1:1 -g [100][500][500]f64 -g [100][500]f64 > data/2d_100x500x500f64_1d_100x500f64

futhark dataset -b --f64-bounds=0:1 -g [100][100]f64 -g [100]f64 -g [100]f64 > data/2x_100x100_1d_100_1d_100f64
futhark dataset -b --f64-bounds=0:1 -g [500][500]f64 -g [500]f64 -g [500]f64 > data/2x_500x500_1d_500_1d_500f64
futhark dataset -b --f64-bounds=0:1 -g [1000][1000]f64 -g [1000]f64 -g [1000]f64 > data/2x_1000x1000_1d_1000_1d_1000f64
futhark dataset -b --f64-bounds=0:1 -g [5000][5000]f64 -g [5000]f64 -g [5000]f64 > data/2x_5000x5000_1d_5000_1d_5000f64
futhark dataset -b --f64-bounds=0:1 -g [10000][10000]f64 -g [10000]f64 -g [10000]f64 > data/2x_10000x10000_1d_10000_1d_10000f64

futhark dataset -b --f64-bounds=0:1 -g [1000000][5][5]f64 -g [1000000][5]f64 -g [1000000][5]f64 > data/2d_1000000x5x5_1d_1000000x5_1d_1000000x5f64
futhark dataset -b --f64-bounds=0:1 -g [1000000][10][10]f64 -g [1000000][10]f64 -g [1000000][10]f64 > data/2d_1000000x10x10_1d_1000000x10_1d_1000000x10f64
futhark dataset -b --f64-bounds=0:1 -g [100000][20][20]f64 -g [100000][20]f64 -g [100000][20]f64 > data/2d_100000x20x20_1d_100000x20_1d_100000x20f64
futhark dataset -b --f64-bounds=0:1 -g [10000][50][50]f64 -g [10000][50]f64 -g [10000][50]f64 > data/2d_10000x50x50_1d_10000x50_1d_10000x50f64
futhark dataset -b --f64-bounds=0:1 -g [1000][100][100]f64 -g [1000][100]f64 -g [1000][100]f64 > data/2d_1000x100x100_1d_1000x100_1d_1000x100f64
futhark dataset -b --f64-bounds=0:1 -g [100][500][500]f64 -g [100][500]f64 -g [100][500]f64 > data/2d_100x500x500f64_1d_100x500_1d_100x500f64