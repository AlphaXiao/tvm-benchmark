import numpy as np
import time, sys
from os import path
import tvm
import torch_tvm
import torch

data_shape = (30, 1, 800)

def get_pytorch_model():
    model = torch.nn.LSTM(data_shape[2], hidden_size=2, num_layers=6, bidirectional=False)
    for name, param in model.named_parameters():
        torch.nn.init.constant_(param, 1.0)
    return model
    
def model_inputs():
    return torch.ones(data_shape)
    
def get_benchmark_name(is_tvm, opt_level, prefix=""):
    if is_tvm:
        return prefix + "TVM_" + "opt_level_" + str(opt_level)
    return prefix + "TORCH_JIT"

def benmarch_pytorch_lstm(is_tvm=False, opt_level=0):
    torch_tvm.disable()
    # Ensure your model is in eval mode and also turn off gradients.
    with torch.no_grad():
        inputs = model_inputs()
        model = get_pytorch_model()
        if is_tvm:
            torch_tvm.enable(opt_level=opt_level, device_type="cpu", device="llvm -mcpu=core-avx2", host="llvm -mcpu=core-avx2")
            # torch_tvm.enable(opt_level=opt_level)
        
        # This is where all the compilation happens.
        mod = torch.jit.trace(model, inputs)
    
        dry_run = 10  # use 10 iterations to warm up
        run = 100
        for i in range(dry_run+run):
            if i == dry_run:
                tic = time.time()
            _ = mod(inputs)
        time_iter = (time.time() - tic) * 1000 / run
        print(f"{get_benchmark_name(is_tvm, opt_level)}, timing: {time_iter} ms")

        if is_tvm:
            if "tvm::CompilationGroup" not in str(mod.graph_for(inputs)):
                print("TVM was not able to optimize this trace.")
            else:
                with torch.autograd.profiler.profile() as prof:
                    _ = mod(inputs)
                tvm_profiled_time = 0
                total_profiled_time = 0
                for p in prof.key_averages():
                    total_profiled_time += int(p.cpu_time)
                    if p.key == "TVM":
                        tvm_profiled_time += int(p.cpu_time)
                print("{} TVM compiling costs {:.2f}%".format(get_benchmark_name(is_tvm, opt_level), 100 * tvm_profiled_time / total_profiled_time))


if __name__ == '__main__':
    benmarch_pytorch_lstm(is_tvm=False)
    benmarch_pytorch_lstm(is_tvm=True, opt_level=0)
    benmarch_pytorch_lstm(is_tvm=True, opt_level=3)