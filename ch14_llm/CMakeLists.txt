# ch14_llm/CMakeLists.txt
# 完整 LLM 推理引擎（阶段二：加载真实 ncnn 权重 + CPU 解码 + 贪心生成）
# 本阶段为 CPU 参考实现，逐算子与后续 Vulkan 内核一一对应。
cmake_minimum_required(VERSION 3.16)

add_executable(ch14_llm
    main.cpp
    tokenizer/bpe_tokenizer.cpp
    llm_model.cpp
    llm_engine.cpp)

target_include_directories(ch14_llm PRIVATE
    ${CMAKE_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR})

# 链接共享封装库（含 i18n 双语支持与 Vulkan 辅助）
target_link_libraries(ch14_llm PRIVATE vk_common)

# 模型目录（含 vocab.txt/merges.txt/.param/.bin）传给代码，避免硬编码路径
target_compile_definitions(ch14_llm PRIVATE
    TOKENIZER_DIR="${CMAKE_SOURCE_DIR}/../model/qwen3_asr_0.6b/")
