cmake_minimum_required (VERSION 3.0)
project (myproject LANGUAGES CUDA CXX)
enable_language(CUDA)
include_directories ("${PROJECT_SOURCE_DIR}/third_party/eigen")
# 查找当前目录下的所有源文件
# 并将名称保存到 DIR_SRCS 变量
aux_source_directory(. DIR_SRCS)
# 添加 math 子目录
add_subdirectory(src)
# 指定生成目标 
add_executable(train train.cc)
# 添加链接库
target_link_libraries(train MiniDNNLib)
# 指定生成目标 
add_executable(test_cpu test_cpu.cc)
# 添加链接库
target_link_libraries(test_cpu MiniDNNLib)

add_executable(test_gpu test_gpu.cc)
# 添加链接库
target_link_libraries(test_gpu MiniDNNLib)

add_executable(test_gpu_cmem test_gpu_cmem.cc)
# 添加链接库
target_link_libraries(test_gpu_cmem MiniDNNLib)

add_executable(test_gpu_cmem_smem test_gpu_cmem_smem.cc)
# 添加链接库
target_link_libraries(test_gpu_cmem_smem MiniDNNLib)
