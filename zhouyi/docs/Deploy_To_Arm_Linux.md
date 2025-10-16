# Deploy the test program 'onnx_test_runner'

1. Make sure the glibc version of the arm64 board matches.
2. Push the Compass_Midware_Runtime/lib/arm64-v8a library to the arm board.
   Assume that the arm64 board path is 'path/to/lib/arm64-v8a',execute the command on the arm64 board:

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:path/to/lib/arm64-v8a
```

3. If the compiler is inconsistent with the embedded Linux version, you need to push the compiler's library to the arm64 board.
   Assume that the arm64 board path is 'path/to/aarch64-none-linux-gnu/lib64',execute the command on the arm64 board:

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:path/to/aarch64-none-linux-gnu/lib64
```

4. Push operator to the arm64 board.
   Assume that the arm64 board path is path/to/operator, execute the command on the arm64 board:

```
export OPERATOR_PATH=/path/to/operator/
```

5. Ensure that the Zhouyi NPU driver works normally.
6. Push the 'test case/QLinearAdd uint8' to the arm64 board.
7. Refer to <User_Guide.md> to compile the arm64 version onnx_test_runner, and then push onnx_test_runner to the board.
8. Run the test program on the arm64 board:

```
onnx_test_runner  -e zhouyi path/to/testcase/QLinearAdd_uint8/
```

# Deploy the test program 'zhouyi_example'

1. Make sure the glibc version of the arm64 board matches.
2. Push the Compass_Midware_Runtime/lib/arm64-v8a library to the arm64 board.
   Assume that the arm64 board path is 'path/to/lib/arm64-v8a',execute the command on the arm64 board:

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:path/to/lib/arm64-v8a
```

3. If the compiler is inconsistent with the embedded Linux version, you need to push the compiler's library to the board.
   Assume that the arm64 board path is 'path/to/aarch64-none-linux-gnu/lib64',execute the command on the arm64 board:

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:path/to/aarch64-none-linux-gnu/lib64
```

4. Push 'libonnxruntime.so.1.22.0' to the arm64 board.
   Assume that the arm64 board path is 'path/to/libonnxruntime.so.1.22.0',execute the command on the arm64 board:

```
cd path/to
ln -s libonnxruntime.so.1.22.0 libonnxruntime.so
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:path/to
```

5. Push operator to the arm64 board.
   Assume that the arm64 board path is path/to/operator, execute the command on the arm64 board:

```
export OPERATOR_PATH=/path/to/operator/
```

6. Ensure that the Zhouyi NPU driver works normally.
7. Refer to <README.md> in example path to compile the arm64 version zhouyi_example, and then push zhouyi_example and test_data to the arm64 board.
8. Run the test program on the arm64 board:

```
./zhouyi_example -m ./test_data/mnist-12-int8.onnx -i ./test_data/input_0.bin
```

# Deploy the python package

1. Refer to <User_Guide.md> to compile the arm64 version file 'onnxruntime_zhouyi-1.22.0-cp310-cp310-linux_aarch64.whl'.
2. Push the following 2 packages to the arm64 board and install:
   onnxruntime_zhouyi-1.22.0-cp310-cp310-linux_aarch64.whl
   ZhouyiOperators-xxxx-py3-none-any.whl
   Dependencies :

   | Module      | Minimum version |
   | ----------- | --------------- |
   | coloredlogs | 15.0.1          |
   | flatbuffers | 24.3.25         |
   | numpy       | 10.0            |

   The aarch64 versions of the above modules can be downloaded from the website https://pypi.org.
   If there are other missing modules for the arm64 development board, please install them yourself.
3. Push The testcase/model  folder to arm64 board.
4. Ensure that the Zhouyi NPU driver works normally.
5. Run python example:

   ```

   python3 run_model.py -m mnist-12-int8/mnist-12-int8.onnx -i mnist-12-int8/input_0.pb

   ```
