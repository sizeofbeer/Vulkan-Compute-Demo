# ch15_coopmat: Cooperative Matrix 编程
# 使用 VK_KHR_cooperative_matrix 扩展实现硬件加速矩阵乘，
# 以及普通分块 GEMM 回退路径。
#
# 编译方式：
#   1. gemm_tiled.comp → gemm_tiled.spv（由 add_chapter 自动编译）
#   2. cm_src/gemm_cm.comp → gemm_cm.spv（手动编译，带 -DUSE_COOPMAT）
#
# 注意：gemm_cm.comp 放在 cm_src/ 目录下而不是 shaders/，以避免被
# add_chapter 的自动 glob 匹配到。它需要 -DUSE_COOPMAT 编译标志来启用
# cooperative matrix 类型和指令。

add_chapter(ch15_coopmat)

# ---- 额外编译：Cooperative Matrix 着色器（带 -DUSE_COOPMAT） ----
# 仅当 glslc 支持 cooperative matrix 内建时编译；否则跳过（运行时 hasCM 守卫）。
set(SPV_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
set(CM_SRC "${CMAKE_CURRENT_SOURCE_DIR}/cm_src/gemm_cm.comp")
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
    message(STATUS "[ch15] glslc 不支持 Cooperative Matrix，跳过 gemm_cm.spv 编译")
    set(_CM_TARGETS "")
endif()

# 将 CM 着色器加入依赖，确保它在可执行文件构建前就绪
if(_CM_TARGETS)
    add_custom_target(ch15_coopmat_cm_shaders DEPENDS ${_CM_TARGETS})
    add_dependencies(ch15_coopmat ch15_coopmat_cm_shaders)
endif()