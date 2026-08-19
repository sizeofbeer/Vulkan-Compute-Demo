# helpers.cmake —— 课程通用的 CMake 辅助
# 提供 add_chapter(NAME)：自动编译本章 shaders/*.comp -> .spv，并链接 common + Vulkan

find_package(Vulkan REQUIRED)

# glslc 来自 Vulkan SDK；找不到时退化为使用预编译的 .spv
find_program(GLSLC glslc DOC "Vulkan GLSL compiler (from Vulkan SDK)")

# ------------------------------------------------------------
# GLSL_COOPMAT_SUPPORTED：检测当前 glslc 是否支持 GL_KHR_cooperative_matrix
# 的 coopmatLoad/coopmatMulAdd/coopmatStore 内建函数。部分 glslc 版本只支持
# coopmat 类型声明、但不支持其加载/存储内建（toolchain 半实现），此时 CM 着色器
# 无法编译，应跳过（运行时由 features.cooperativeMatrixKHR 守卫）。
# 用法：check_coopmat_support(<out_var>)
# ------------------------------------------------------------
function(check_coopmat_support OUT)
    set(${OUT} FALSE PARENT_SCOPE)
    if(NOT GLSLC)
        return()
    endif()
    # 生成一个最小 coopmat 着色器并尝试编译；失败即认为不支持
    set(_t "${CMAKE_CURRENT_BINARY_DIR}/_coopmat_probe.comp")
    file(WRITE ${_t} "#version 450\n"
        "#extension GL_KHR_cooperative_matrix : require\n"
        "#extension GL_KHR_memory_scope_semantics : enable\n"
        "layout(local_size_x=16) in;\n"
        "layout(std430,binding=0) buffer Buf { float a[]; };\n"
        "void main(){\n"
        "  coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> f;\n"
        "  coopmatLoad(f, a, 16, gl_CooperativeMatrixLayoutRowMajor, 0);\n"
        "  coopmatStore(a, f, 16, gl_CooperativeMatrixLayoutRowMajor, 0);\n"
        "}\n")
    execute_process(
        COMMAND ${GLSLC} --target-env=vulkan1.3 ${_t} -o ${_t}.spv
        RESULT_VARIABLE _res
        OUTPUT_QUIET ERROR_QUIET)
    if(_res EQUAL 0)
        set(${OUT} TRUE PARENT_SCOPE)
    endif()
endfunction()

function(add_chapter NAME)
    # 本章可执行文件（源码固定为 main.cpp）
    add_executable(${NAME} main.cpp)
    target_link_libraries(${NAME} PRIVATE vk_common Vulkan::Vulkan)
    target_include_directories(${NAME} PRIVATE ${CMAKE_SOURCE_DIR}/common)

    # 编译本章所有 .comp 着色器到 build 目录下的 shaders/
    file(GLOB SHADER_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.comp")
    set(SPV_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
    file(MAKE_DIRECTORY ${SPV_DIR})

    set(SPV_TARGETS "")
    foreach(sh ${SHADER_SOURCES})
        get_filename_component(sn ${sh} NAME_WE)
        set(spv "${SPV_DIR}/${sn}.spv")
        if(GLSLC)
            add_custom_command(
                OUTPUT ${spv}
                COMMAND ${GLSLC} --target-env=vulkan1.1 ${sh} -o ${spv}
                DEPENDS ${sh}
                COMMENT "glslc: ${sn}.comp -> ${sn}.spv")
        else()
            # 没有 glslc：尝试拷贝源码目录里已有的 .spv；否则告警
            if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${sn}.spv")
                configure_file("${CMAKE_CURRENT_SOURCE_DIR}/shaders/${sn}.spv"
                               ${spv} COPYONLY)
            else()
                message(WARNING "glslc 未找到，且没有预编译的 ${sn}.spv，请手动用 glslc 编译着色器")
            endif()
        endif()
        list(APPEND SPV_TARGETS ${spv})
    endforeach()

    # 用一个 custom target 保证着色器先于可执行构建
    if(SPV_TARGETS)
        add_custom_target(${NAME}_shaders DEPENDS ${SPV_TARGETS})
        add_dependencies(${NAME} ${NAME}_shaders)
    endif()

    # 把着色器输出目录通过宏传给代码，main 里用 SHADER_DIR "xxx.spv"
    target_compile_definitions(${NAME} PRIVATE SHADER_DIR="${SPV_DIR}/")
endfunction()
