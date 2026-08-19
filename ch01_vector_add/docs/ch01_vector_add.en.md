# Chapter 1 · Your First Compute Shader: Vector Addition

> Goal: get a full "GPU operator" loop running end to end. **Don't try to understand every single Vulkan call**—first burn the main pipeline "create buffer → upload → build pipeline → dispatch → read back" into your memory.

---

## 1.1 Why Start with Vector Addition

Almost every operation in a neural network ultimately boils down to "parallel computation over a large blob of numbers": matrix multiplication is nested parallelism, convolution is offset parallelism, and activation functions are per-element parallelism. Vector addition (element-wise parallelism) is the simplest of these, enough to reveal the skeleton of GPU computation without being distracted by the operator's math details.

## 1.2 The Lifecycle of an Operator in Vulkan

```
CPU 数据 ──上传──▶ GPU buffer（storage buffer）
                        │
            绑定管线 + 描述符 + push constant
                        │
                 vkCmdDispatch（启动 N 个线程）
                        │
            GPU 并行执行 compute shader（每个线程算一个元素）
                        │
            结果留在 GPU buffer ──读回──▶ CPU 校验
```

Mapped to the steps in `main.cpp`:

1. **Create buffer**: `vkCreateBuffer` three buffers (a, b, c), each with usage `eStorageBuffer` (for the shader to read/write) and `eTransferSrc/Dst` (for upload/readback). Memory is chosen as `eDeviceLocal` (device-local video memory, fastest).
2. **Upload**: `vkUpload` copies the CPU's `a`, `b` into the GPU. device-local memory cannot be written directly by the CPU, so the common library automatically uses a host-visible staging buffer to mediate internally (this mediating mechanism is explained in detail in **ch03**).
3. **Build pipeline**: `vkLoadShader` reads the `.spv` → `createShaderModule` → `createComputePipeline`. The pipeline contains the "descriptor layout" (states how many buffers the shader needs) and the "push constant range" (states which constants can be pushed in).
4. **Bind descriptors**: `updateDescriptorSets` binds the real buffers to the shader's `binding 0/1/2`. This step is equivalent to "telling the shader: a is at slot 0, b at slot 1, c at slot 2".
5. **Dispatch**: number of workgroups = `ceil(N/64)`, because one workgroup has 64 threads (`local_size_x = 64` in `vec_add.comp`).
6. **Read back + verify**: `vkDownload` copies the result back to the CPU and compares `c[i] == a[i]+b[i]` element by element.

## 1.3 Two Key Concepts (Just Memorize for Now, Dig Deeper Later)

- **Workgroup**: `dispatch(groups,1,1)` launches `groups` workgroups, each with 64 threads. The global thread index `gl_GlobalInvocationID.x` ranges over 0..N-1 and naturally serves as the element index. The GPU schedules by "workgroup"; threads within the same workgroup can share shared memory (only needed from **ch04**).
- **push constant vs descriptor**: Shape parameters (here a single `n`) are lighter weight via push constant; the real bulk data (buffers) uses descriptors. ncnn / llama.cpp both use push constants heavily to pass shapes.

## 1.4 Hands-on Experiments

1. Build and run with Validation Layers enabled, and confirm "错误数=0" (error count = 0).
2. Change `N` to a value that is not a multiple of 64 (e.g. 1000), and verify that the boundary guard `if (i >= n) return;` is necessary—what happens if you remove it? (It would write to an out-of-bounds address; Validation Layers may report it, or it may silently compute wrong results.)
3. Change `local_size_x` to 32 or 256, adjust the `groups` formula `(N+local-1)/local` accordingly, and rerun. Feel how the "workgroup size" is tunable.
4. Try changing addition to multiplication or `a[i]*2 + 1`—you only need to modify the `.comp` and recompile, no C++ changes required (this is the benefit of declarative shaders).

## 1.5 Compared to Industrial Grade

- ncnn's vector addition corresponds to its "element-wise operator" (`Elementwise` series of GPU kernels), but ncnn also does **elempack=4 packing** (processing 4 elements at once) and fuses it into adjacent operators.
- llama.cpp has no standalone "vector addition"; similar element-wise operations are fused into kernels like `mul_f32` / `add_f32`, and shapes are passed via `vk_op_binary_push_constants`.

In the next chapter (ch02) we won't rush to add new operators; instead we will **explain synchronization (fence / barrier) thoroughly**—it is the most easily overlooked and the most bug-prone part of GPU computation.
