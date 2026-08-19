# ch04_gemm/CMakeLists.txt
# 手动管理着色器编译，因为 gemm_cm.comp 需要 -DUSE_COOPMAT 标志
# gemm_tiled.comp 使用普通编译

find_package(Vulkan REQUIRED)
find_program(GLSLC glslc DOC "Vulkan GLSL compiler (from Vulkan SDK)")

# 可执行文件
add_executable(ch04_gemm main.cpp)
target_link_libraries(ch04_gemm PRIVATE vk_common Vulkan::Vulkan)
target_include_directories(ch04_gemm PRIVATE ${CMAKE_SOURCE_DIR}/common)

set(SPV_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
file(MAKE_DIRECTORY ${SPV_DIR})

# ---- 1. 编译 gemm_tiled.comp（普通分块 GEMM） ----
set(TILED_SRC "${CMAKE_CURRENT_SOURCE_DIR}/shaders/gemm_tiled.comp")
set(TILED_SPV "${SPV_DIR}/gemm_tiled.spv")
if(GLSLC)
    add_custom_command(
        OUTPUT ${TILED_SPV}
        COMMAND ${GLSLC} ${TILED_SRC} -o ${TILED_SPV}
        DEPENDS ${TILED_SRC}
        COMMENT "glslc: gemm_tiled.comp -> gemm_tiled.spv")
else()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/gemm_tiled.spv")
        configure_file("${CMAKE_CURRENT_SOURCE_DIR}/shaders/gemm_tiled.spv" ${TILED_SPV} COPYONLY)
    else()
        message(WARNING "glslc 未找到，且没有预编译的 gemm_tiled.spv")
    endif()
endif()

# ---- 2. 编译 gemm_cm.comp（Cooperative Matrix GEMM，带 -DUSE_COOPMAT） ----
# 仅当 glslc 支持 cooperative matrix 内建时编译；否则跳过（运行时 hasCM 守卫）。
# 部分 glslc 只支持 coopmat 类型而不支持 coopmatLoad/Store 内建，无法编译。
set(CM_SRC "${CMAKE_CURRENT_SOURCE_DIR}/shaders/gemm_cm.comp")
set(CM_SPV "${SPV_DIR}/gemm_cm.spv")
check_coopmat_support(_COOPMAT_OK)
if(_COOPMAT_OK)
    add_custom_command(
        OUTPUT ${CM_SPV}
        COMMAND ${GLSLC} -DUSE_COOPMAT ${CM_SRC} -o ${CM_SPV}
        DEPENDS ${CM_SRC}
        COMMENT "glslc (CM): gemm_cm.comp -> gemm_cm.spv (with -DUSE_COOPMAT)")
    set(_CM_TARGETS ${CM_SPV})
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/gemm_cm.spv")
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/shaders/gemm_cm.spv" ${CM_SPV} COPYONLY)
    set(_CM_TARGETS ${CM_SPV})
else()
    # 无 glslc 支持且无预编译产物：不生成 CM 着色器，运行时由 hasCM + 文件存在性守卫
    message(STATUS "[ch04] glslc 不支持 Cooperative Matrix，跳过 gemm_cm.spv 编译")
    set(_CM_TARGETS "")
endif()

# 着色器依赖目标
add_custom_target(ch04_gemm_shaders DEPENDS ${TILED_SPV} ${_CM_TARGETS})
add_dependencies(ch04_gemm ch04_gemm_shaders)

# 把着色器输出目录通过宏传给代码
target_compile_definitions(ch04_gemm PRIVATE SHADER_DIR="${SPV_DIR}/")