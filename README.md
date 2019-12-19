# TVM benchmark

## Notes

* Versions

    * tvm, github 2019.11.26 master branch 
    * pytorch, github 2019.11.22 master branch
    * pytorch-tvm, github 2019.11.06 master branch
    * tensorflow 1.11.0
    * mxnet with cuda, mkl support, `pip install mxnet-cu101mkl`

* [pytorch-tvm](https://github.com/pytorch/tvm)

    * 使用 pytorch 版本: 2019.11.22 github master branch [commit 183aa1534f9e199e1e67e453dfb94dc855dabc0d]
    * 需要手动编译编译最新版本 torch，不然在 `import torch-tvm` 时会出现 undefined symbol 问题。 [ref issue](https://github.com/pytorch/tvm/issues/77)
    * g++5 版本不行，可以在 `setup.py` 中指定 g++, gcc 版本 `'-DCMAKE_CXX_COMPILER=g++-8','-DCMAKE_C_COMPILER=gcc-8'`

* The code of `pytorch_resnet` is from [pytorch-tvm](https://github.com/pytorch/tvm/test)

* tvm 暂不支持 tensorflow 2.0 ([ref issue](https://github.com/apache/incubator-tvm/issues/4102))，目前使用 tensorflow 1.11 进行测试。

## Results

### Pytorch + TVM

* LSTM (pytorch_lstm.py)

    使用 profiler 获取各个步骤时间发现，并没有 tvm compile 的时间。猜测 TVM 暂时不知道 LSTM 中用到的一些操作。

    ```
    TORCH_JIT, timing: 9.202501773834229 ms
    TVM_opt_level_0, timing: 8.887653350830078 ms
    TVM was not able to optimize this trace.
    TVM_opt_level_3, timing: 8.944499492645264 ms
    TVM was not able to optimize this trace.
    ```

* RESNET (pytorch_resnet/benchmark.py)

    使用了 TVM 性能反而下降了，在官方给的性能数据里应该有 2 倍左右的性能提升。

    ```
    Tracing model with JIT
    Warming JIT up with 10 runs
    Running JIT 100 times
    Done benchmarking JIT
    Tracing model with TVM
    Warming TVM up with 10 iters
    WARNING: reshape with -1 as the first value has known incompatibility with PyTorch semantics.
    Running TVM 100 times
    Done benchmarking TVM, which compiled 0.27% of compute
    JIT: 39.00197515715197 iter/s
    TVM: 28.652232198238856 iter/s
    ```

### Tensorflow + TVM

* LSTM (tf_lstm.py)

    ~~会出现 `KeyError: ‘rnn/TensorArray:1’` Error，暂时没找到修复的办法。尝试过社区提供的[一些办法](https://discuss.tvm.ai/t/tensorarray-globalvar-and-globaltypevar-confusion/4567/23)，仍然有问题。~~ fixed by [branch](https://github.com/soiferj/tvm/commit/b9d14c59fcc069f122461a21270d51bb4825ae66)

    目前仍然会出现 `AttributeError: relay.Call object has no attributed type_annotation`

### MXNet + TVM

* LSTM

    target = "llvm target=x86_64-linux-gnu"

    ```
    mxnet lstm_cell: 24.87 ms
    mxnet lstm fuse: 1.89 ms
    tvm compiling ...
    WARNING:autotvm:Cannot find config for target=llvm -target=x86_64-linux-gnu, workload=('dense', (1, 2, 'float32'), (8, 2, 'float32'), 0, 'float32'). A fallback configuration is used, which may bring great performance regression.
    WARNING:autotvm:Cannot find config for target=llvm -target=x86_64-linux-gnu, workload=('dense', (1, 800, 'float32'), (8, 800, 'float32'), 0, 'float32'). A fallback configuration is used, which may bring great performance regression.
    tvm compiling completed. spent 2.058202 seconds
    tvm lstm_cell opt=0  2.62 ms             (1.21 ms)
    tvm compiling ...
    tvm compiling completed. spent 16.477742 seconds
    tvm lstm_cell opt=3  2.00 ms             (0.47 ms)
    tvm compiling ...
    tvm compiling completed. spent 1.502558 seconds
    tvm lstm opt=0       2.06 ms             (0.46 ms)
    tvm compiling ...
    tvm compiling completed. spent 3.496346 seconds
    tvm lstm opt=3       1.99 ms             (1.54 ms)
    ```

    target = "llvm -mcpu=core-avx2"
    ```
    mxnet lstm_cell: 26.27 ms
    mxnet lstm fuse: 2.11 ms
    tvm compiling ...
    WARNING:autotvm:Cannot find config for target=llvm -mcpu=core-avx2, workload=('dense', (1, 2, 'float32'), (8, 2, 'float32'), 0, 'float32'). A fallback configuration is used, which may bring great performance regression.
    WARNING:autotvm:Cannot find config for target=llvm -mcpu=core-avx2, workload=('dense', (1, 800, 'float32'), (8, 800, 'float32'), 0, 'float32'). A fallback configuration is used, which may bring great performance regression.
    tvm compiling completed. spent 2.367452 seconds
    tvm lstm_cell opt=0  1.99 ms             (0.25 ms)
    tvm compiling ...
    tvm compiling completed. spent 18.370862 seconds
    tvm lstm_cell opt=3  1.75 ms             (0.22 ms)
    tvm compiling ...
    tvm compiling completed. spent 1.780173 seconds
    tvm lstm opt=0       2.11 ms             (0.37 ms)
    tvm compiling ...
    tvm compiling completed. spent 4.781243 seconds
    tvm lstm opt=3       1.68 ms             (0.16 ms)
    ```

    target = "cuda", GPU: Nvidia 1080Ti
    ```
    mxnet lstm_cell: 17.68 ms
    mxnet lstm fuse: 3.28 ms
    tvm compiling ...
    WARNING:autotvm:Cannot find config for target=cuda, workload=('dense', (1, 2, 'float32'), (8, 2, 'float32'), 0, 'float32'). A fallback configuration is used, which may bring great performance regression.
    WARNING:autotvm:Cannot find config for target=cuda, workload=('dense', (1, 800, 'float32'), (8, 800, 'float32'), 0, 'float32'). A fallback configuration is used, which may bring great performance regression.
    tvm compiling completed. spent 3.917630 seconds
    tvm lstm_cell opt=0  12.16 ms            (0.32 ms)
    tvm compiling ...
    tvm compiling completed. spent 19.705491 seconds
    tvm lstm_cell opt=3  5.07 ms             (0.16 ms)
    tvm compiling ...
    tvm compiling completed. spent 2.337069 seconds
    tvm lstm opt=0       11.02 ms            (0.24 ms)
    tvm compiling ...
    tvm compiling completed. spent 5.595142 seconds
    tvm lstm opt=3       4.75 ms             (0.15 ms)
    ```
