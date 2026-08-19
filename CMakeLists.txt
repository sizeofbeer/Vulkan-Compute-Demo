cmake_minimum_required(VERSION 3.16)
project(vulkan_nn_course CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 顶层引入通用函数（add_chapter）
include(cmake/helpers.cmake)

# 共享 Vulkan 封装库：所有章节都链接它，避免重复样板
add_subdirectory(common)

# 已实现的章节（后续章节在此追加 add_subdirectory）
add_subdirectory(ch00_environment)
add_subdirectory(ch01_vector_add)
add_subdirectory(ch02_sync)
add_subdirectory(ch03_memory)
add_subdirectory(ch04_gemm)
add_subdirectory(ch05_conv)
add_subdirectory(ch06_elemwise)
add_subdirectory(ch07_norm)
add_subdirectory(ch08_attention)
add_subdirectory(ch09_profiling)
add_subdirectory(ch10_arena)
add_subdirectory(ch11_quant)
add_subdirectory(ch12_graph)
add_subdirectory(ch13_mlp)

# ch14：完整 LLM 推理引擎（阶段一：tokenizer 闭环）
add_subdirectory(ch14_llm)

# ch15：Cooperative Matrix 编程（Tensor Core / Matrix Core 硬件加速）
add_subdirectory(ch15_coopmat)

# ch16：Subgroup 编程（Warp 级并行）
add_subdirectory(ch16_subgroup)

# ch17：FlashAttention（融合注意力）
add_subdirectory(ch17_flash_attn)

# ch18：深度量化（Q4_0 / Q8_0）
add_subdirectory(ch18_quant_deep)

# ch20：前沿量化（K-quants / MXFP4）
add_subdirectory(ch20_frontier_quant)

# ch19：MoE 路由（Mixture of Experts）
add_subdirectory(ch19_moe)
