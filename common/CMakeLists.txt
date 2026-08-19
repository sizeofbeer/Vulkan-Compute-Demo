# common/CMakeLists.txt —— 共享封装库
add_library(vk_common STATIC vk_base.cpp i18n.cpp)
target_include_directories(vk_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(vk_common PUBLIC Vulkan::Vulkan)
