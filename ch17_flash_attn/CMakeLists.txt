add_chapter(ch17_flash_attn)

# 进阶 demo：KV cache 增量解码 + decode 两阶段（split / reduce）
# 独立可执行目标，复用本章编译出的 .spv，不改动 main.cpp 的既有验证流程。
add_executable(ch17_kvcache main_kvcache.cpp)
target_link_libraries(ch17_kvcache PRIVATE vk_common Vulkan::Vulkan)
target_include_directories(ch17_kvcache PRIVATE ${CMAKE_SOURCE_DIR}/common)
target_compile_definitions(ch17_kvcache PRIVATE SHADER_DIR="${CMAKE_CURRENT_BINARY_DIR}/shaders/")
add_dependencies(ch17_kvcache ch17_flash_attn_shaders)
