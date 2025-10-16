# How to build

Note: These build instructions are for the Linux examples only.

## Prerequisites

1. Refer to the top-level readme.md to compile onnxruntime with ZHOUYI EP.
2. cmake(version >=3.13)

## Build the example

### x86_64

1. use build.sh

```shell
$ ./build.sh   -c /path/to/gcc/  -n  /path/to/onnxruntime
```

2. use cmake

```shell
$ mkdir build && cd build
cmake -DCMAKE_C_COMPILER=/path/to/gcc/bin/gcc \
-DCMAKE_CXX_COMPILER=/path/to/gcc/bin/g++ \
-Donnxruntime_USE_ZHOUYI=ON \
-DONNXRUNTIME_ROOTDIR=path/to/onnxruntime \
-DONNXRUNTIME_LIB_DIR=/path/to/libonnxruntime.so \
..
```

### ARM64

1. use build.sh

```shell
$ ./build.sh -c /path/to/gcc-arm-10.3-x86_64-aarch64-none-linux-gnu  -n /path/to/onnxruntime  -l ../path/to/libonnxruntime.so --prefix aarch64-none-linux-gnu-
```

    for example:

./build.sh -c /arm/tools/gcc-arm-10.3-x86_64-aarch64-none-linux-gnu  -n ../onnxruntime  -l ../onnxruntime/build_arm64/Debug/ --prefix aarch64-none-linux-gnu-

2. use cmake

```shell
$ mkdir build && cd buildcmake -DCMAKE_C_COMPILER=/path/to/aarch64-none-linux-gnu-gcc \
-DCMAKE_CXX_COMPILER=/path/to/aarch64-none-linux-gnu-g++ \
-Donnxruntime_USE_ZHOUYI=ON \
-DONNXRUNTIME_ROOTDIR=/path/to/onnxruntime \
-DONNXRUNTIME_LIB_DIR=/path/to/libonnxruntime.so \
..
```

## How to run the example?

### X86_64

```shell
$ export OPERATOR_PATH=/path/to/operator_path/
$ ./zhouyi_example -m ./test_data/mnist-12-int8.onnx -i ./test_data/input_0.bin
```

The 'operator path is the directory where 'aiffcllib_x2.a' and 'tpccllib_x2.a' are stored.

### ARM64

1. Make sure the glibc version of the arm board matches.
2. Push the Compass_Midware_Runtime/lib/arm64-v8a library to the arm board.
   Assume that the arm board path is 'path/to/lib/arm64-v8a',execute the command on the arm board:

```shell
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:path/to/lib/arm64-v8a
```

3. If the compiler is inconsistent with the embedded Linux version, you need to push the compiler's library to the board.
   Assume that the arm board path is 'path/to/aarch64-none-linux-gnu/lib64',execute the command on the arm board:

```shell
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:path/to/aarch64-none-linux-gnu/lib64
```

4. Push 'libonnxruntime.so.1.17.0' to the arm64 board.
   Assume that the arm64 board path is 'path/to/libonnxruntime.so.1.17.0',execute the command on the arm64 board:
```shell
$ cd path/to
$ ln -s libonnxruntime.so.1.17.0 libonnxruntime.so
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:path/to
```
5. Push operator to the board.
   Assume that the arm board path is path/to/operator, execute the command on the arm board:

```shell
$ export OPERATOR_PATH=/path/to/operator/
```

6. Ensure that the Zhouyi NPU driver works normally.
7. Refer to README.md in example path to compile the arm64 version zhouyi_example, and then push zhouyi_example and test_data to the arm board.
8. Run the test program on the arm board:

```shell
$ ./zhouyi_example -m ./test_data/mnist-12-int8.onnx -i ./test_data/input_0.bin
```

## How to run example with cache?
1. Generate cache onnx model 
Note: zhouyi_example also supports switching embed_mode and specifying path of context onnx model, which refers by `./zhouyi_example -h`
```shell
$ ./zhouyi_example -m ./test_data/mnist-12-int8.onnx -i ./test_data/input_0.bin -f
```
2. Uses cached onnx model (default embed_mode=1)
```shell
$ ./zhouyi_example -m ./test_data/mnist-12-int8.onnx_ctx.onnx -i ./test_data/input_0.bin
```
