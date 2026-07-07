cmake_minimum_required(VERSION 3.16)
project(real_robot_interface_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Sonic 现成 C++ 依赖库与头文件路径
set(UNITREE_SDK2_ROOT "/home/feng/GR00T-WholeBodyControl/gear_sonic_deploy/thirdparty/unitree_sdk2")

include_directories(
    ${UNITREE_SDK2_ROOT}/include
    ${UNITREE_SDK2_ROOT}/thirdparty/include
    ${UNITREE_SDK2_ROOT}/thirdparty/include/ddscxx
)

link_directories(
    ${UNITREE_SDK2_ROOT}/lib/x86_64
    ${UNITREE_SDK2_ROOT}/thirdparty/lib/x86_64
)

# 确保运行时能够自动找到 libddsc.so 和 libddscxx.so
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "${UNITREE_SDK2_ROOT}/thirdparty/lib/x86_64")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

add_library(real_robot_interface_cpp SHARED
    real_robot_interface_cpp.cpp
)

target_compile_options(real_robot_interface_cpp PRIVATE -O3 -ffast-math -Wall -DNDEBUG)

target_link_libraries(real_robot_interface_cpp PRIVATE
    ${UNITREE_SDK2_ROOT}/lib/x86_64/libunitree_sdk2.a
    ddsc
    ddscxx
    pthread
    dl
    rt
)

target_link_options(real_robot_interface_cpp PRIVATE
    "-Wl,-rpath,${UNITREE_SDK2_ROOT}/thirdparty/lib/x86_64"
)
