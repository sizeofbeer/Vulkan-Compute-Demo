add_chapter(ch19_moe)

# 进阶 demo：稀疏 Gather/Scatter MoE（练习 #3 的参考实现）
# 独立可执行目标，复用本章编译出的 .spv，不改动 main.cpp 的既有验证流程。
add_executable(ch19_moe_sparse main_sparse.cpp)
target_link_libraries(ch19_moe_sparse PRIVATE vk_common Vulkan::Vulkan)
target_include_directories(ch19_moe_sparse PRIVATE ${CMAKE_SOURCE_DIR}/common)
target_compile_definitions(ch19_moe_sparse PRIVATE SHADER_DIR="${CMAKE_CURRENT_BINARY_DIR}/shaders/")
add_dependencies(ch19_moe_sparse ch19_moe_shaders)
