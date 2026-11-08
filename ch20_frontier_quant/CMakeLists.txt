add_chapter(ch20_frontier_quant)

# 进阶 demo：NVFP4（E4M3 两级缩放）vs 本章 MXFP4（fp16 单级缩放）
# 独立可执行目标，复用本章编译出的 .spv，不改动 main.cpp 的既有验证流程。
add_executable(ch20_nvfp4 main_nvfp4.cpp)
target_link_libraries(ch20_nvfp4 PRIVATE vk_common Vulkan::Vulkan)
target_include_directories(ch20_nvfp4 PRIVATE ${CMAKE_SOURCE_DIR}/common)
target_compile_definitions(ch20_nvfp4 PRIVATE SHADER_DIR="${CMAKE_CURRENT_BINARY_DIR}/shaders/")
add_dependencies(ch20_nvfp4 ch20_frontier_quant_shaders)
