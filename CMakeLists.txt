cmake_minimum_required(VERSION 3.14)
project(RainTorch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 头文件搜索路径
include_directories(${PROJECT_SOURCE_DIR}/src)

# 收集框架源码，编译静态库
file(GLOB CORE_SRC
    src/autograd/node.cc
    src/autograd/ops.cc
)
add_library(raintorch STATIC ${CORE_SRC})

# 编译示例可执行文件
add_executable(node_demo examples/cpp/main.cc)
target_link_libraries(node_demo raintorch)

# 指定可执行文件输出到 bin 目录
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

add_executable(demo_node examples/cpp/demo_node.cc)
target_link_libraries(demo_node raintorch)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
