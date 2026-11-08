# Chapter 2 · The Synchronization Model: fences and pipeline barriers

> Goal: understand why the GPU is "asynchronous", and what the two synchronization primitives — **fence (CPU↔GPU)** and **barrier (between operators inside the GPU)** — each solve.

---

## 2.1 The GPU is asynchronous

When `vkQueueSubmit` returns, the operators have most likely not started computing yet. The commands are merely pushed into the queue, and the GPU executes them on a separate timeline. This leads to a counter-intuitive fact:

- Timing around `submit` with the CPU clock only measures the "submission cost" (tens of microseconds), not the "computation cost".
- Reading the result buffer immediately after `submit` may read garbage data from the previous frame.

So synchronization is mandatory.

## 2.2 Two synchronization primitives

| Primitive | Scope | What it solves |
|-----------|-------|----------------|
| **fence** | CPU ↔ GPU | Lets the CPU wait until "the whole batch of commands has finished" before continuing (e.g. reading results, freeing resources) |
| **semaphore** | queue ↔ queue | Ordering across queues / across submissions (not covered in this course) |
| **pipeline barrier** | inside the GPU | Within the same command buffer, the next operator may only read after the previous one has finished writing (avoiding hazards) |

In this chapter the two kernels are chained in the same command buffer: first `b=a+1`, then `c=b*2`. The second kernel reads `b` written by the first — this is the classic **Read-After-Write (RAW) hazard** (the inverse of Write-After-Read). A `pipelineBarrier` must be inserted in between to tell the driver: "before the second dispatch starts, all writes to b must be globally visible".

## 2.3 What happens in the code

Key section of `main.cpp`:

```cpp
cmd.bindPipeline(..., p1); cmd.pushConstants(...); cmd.dispatch(...);   // write b
// ↓ insert barrier
vk::BufferMemoryBarrier bmb{eShaderWrite, eShaderRead, 0, 0, bufB.buf, 0, VK_WHOLE_SIZE};
cmd.pipelineBarrier(eComputeShader, eComputeShader, {}, 0,nullptr, 1,&bmb, 0,nullptr);
cmd.bindPipeline(..., p2); cmd.pushConstants(...); cmd.dispatch(...);   // read b, write c
```

Parameters of `pipelineBarrier`:
- `srcStageMask = eComputeShader`: the "producer" stage before it is the compute shader.
- `dstStageMask = eComputeShader`: the "consumer" stage after it is also the compute shader.
- `bmb`: `b`'s access changes from `ShaderWrite` to `ShaderRead`, and the driver inserts the correct wait based on this.

Finally `submit` + `waitForFences`: this is the fence taking effect — it guarantees that when we `vkDownload` and read `c`, the GPU has actually finished computing.

## 2.4 Hands-on experiments (required)

1. **Verify the necessity of the barrier**: comment out the `cmd.pipelineBarrier(...)` line, then recompile and run. On some GPUs / drivers you will see "errors > 0" or random results — that is the consequence of missing synchronization. Add the barrier back and the error count drops to zero.
2. **Verify the necessity of the fence**: insert `vkDownload` before `waitForFences` and immediately print the first few `c` values; compare before vs after the wait — before the wait they may be uninitialized data.
3. **Think about it**: if you split the two dispatches into two separate `submit`s (each with its own fence in between), is the barrier still needed? (Hint: does a fence across submits imply a barrier? Answer: in Vulkan there is an **implicit barrier** between submits on the same queue, but writing the barrier explicitly is clearer and avoids the extra submit overhead within a single submission.)

## 2.5 Compared with industry-grade

- **ncnn**'s `VkCompute` maintains `delayed_records` and `pending_dispatch_total()`: `record_pipeline()` only records, while `submit_and_wait()` submits once and **inserts dependency barriers automatically** — the caller never hand-writes `pipelineBarrier`. The "hand-written barrier" here exists so you understand what ncnn does for you. (The `VkCompute::dispatch()` name from the first release is not a public API; dispatch only appears as internal `vkCmdDispatch` calls in `command.cpp`.)
- **llama.cpp** once specifically added a `buffer ownership transfer` barrier on AMD devices (because queue / memory ownership switches require extra synchronization). Synchronization is the easiest pitfall in Vulkan inference, so make sure you fully grasp it here.

The next chapter (ch03) covers **memory management**: why does `vkUpload` go through a staging buffer when device-local? What exactly differs between `HOST_VISIBLE / DEVICE_LOCAL / COHERENT`? And how is FP16 stored into a buffer.
