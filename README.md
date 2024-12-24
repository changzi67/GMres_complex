# ILUTP预条件+GMRES求解

## 编译依赖
- [SuiteSparse软件包](https://github.com/DrTimothyAldenDavis/SuiteSparse) 中的AMD子包，用于矩阵重排序
- [Eigen3](https://eigen.tuxfamily.org/)，一个header-only的库，用于生成参考解和计算误差
- [fast_matrix_market](https://github.com/alugowski/fast_matrix_market/)，一个header-only的库，用于读取MTX格式的矩阵文件
- [Intel MKL](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2025-0/overview.html)，用于GMRES中的矩阵向量乘法和内积计算

## 编译与运行说明
程序使用CMake编译，MKL的路径由环境变量`MKLROOT`指定，其他的依赖库都放在了`/home/chengjw/.locals`下，所以CMakeLists.txt中的路径是这样写的：

```CMAKE
set(LIB_DIR /home/chengjw/.locals)
# ...
set(MKL_THREADING gnu_thread)
set(MKL_INTERFACE lp64)
LIST(APPEND CMAKE_PREFIX_PATH ${LIB_DIR}/SuiteSparse-7.4.0.beta10)
find_package(MKL CONFIG REQUIRED PATHS $ENV{MKLROOT})
find_package(SuiteSparse_config CONFIG REQUIRED)
find_package(AMD CONFIG REQUIRED)
# ...
target_include_directories(ilu_gmres PUBLIC
        ./third_party
        ${LIB_DIR}/eigen-3.4.0
        ${LIB_DIR}/fast_matrix_market/include
        ${LIB_DIR}/fast_matrix_market/include/dragonbox)
target_link_directories(ilu_gmres PUBLIC
        ${LIB_DIR}/fast_matrix_market/lib64)
```

如果你将依赖库放在了其他地方，则需要对应地修改这些路径。在配置好路径后，编译方法如下：

```shell
cmake -B build/
cmake --build build/ --parallel
```

运行方法如下：

```txt
Usage:
  ./build/ilu_gmres [OPTION...]

  -m, --mat arg           Coefficient matrix file (Matrix Market)
  -o, --optimization arg  MKL optimization level (default: 10)
  -p, --pivot-tol arg     ILUTP pivot tolerance (default: 0)
  -f, --fill-factor arg   ILUTP fill factor (default: 1)
  -e, --extra-fill arg    ILUTP extra fills (default: 0)
  -d, --drop-tol arg      ILUTP drop tolerance (default: 1e-3)
  -t, --tol arg           GMRES tolerance (default: 1e-6)
  -i, --maxit arg         GMRES max iterations (default: 10)
  -r, --restart arg       GMRES restart iterations (default: 100)
  -h, --help              Print usage
```

## 程序流程

1. 使用`fast_matrix_market`库以**CSR**格式读取稀疏矩阵文件
2. 以全1向量作为解`x`，使用`Eigen3`库生成右端项`b=Ax`
3. 使用ILUTP预条件+GMRES进行求解，其中包括：
   + 调用`solver.analyze`进行MC64静态选主元与amd重排序
   + 调用`solver.factorize_ilutp`进行ILUTP分解
   + 调用`solver.solve`进行求解
4. 保存解到`sol.txt`，并打印ILU因子的非零元数、相对残差、解的相对误差、运行时间等信息

## 编译控制
在`ilu_gmres.h`中，有一些类型定义可以控制矩阵索引和值的类型：
+ `idx_t`：矩阵的索引类型，目前为`int`
+ `val_t`：矩阵的值类型，目前为`double`

在`ilu_gmres.cpp`中，有一些宏定义可以控制程序的行为：
+ `DEBUG`：是否打印调试信息
