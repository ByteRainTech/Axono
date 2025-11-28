#!/usr/bin/env bash

CMAKE_ARGS="-Dpybind11_DIR=$(python3 -m pybind11 --cmakedir)"
BUILD_ARGS="-j4"

cat <<-'EOF'
       db         8b        d8  ,ad8888ba,    888b      88    ,ad8888ba,    
      d88b         Y8,    ,8P  d8"'    `"8b   8888b     88   d8"'    `"8b   
     d8'`8b         `8b  d8'  d8'        `8b  88 `8b    88  d8'        `8b  
    d8'  `8b          Y88P    88          88  88  `8b   88  88          88  
   d8YaaaaY8b         d88b    88          88  88   `8b  88  88          88  
  d8""""""""8b      ,8P  Y8,  Y8,        ,8P  88    `8b 88  Y8,        ,8P  
 d8'        `8b    d8'    `8b  Y8a.    .a8P   88     `8888   Y8a.    .a8P   
d8'          `8b  8P        Y8  `"Y8888Y"'    88      `888    `"Y8888Y"'    
                                                                           
Axono 安装向导
Github: https://github.com/ByteRainTech/Axono
EOF
read -n 1 -p "是否安装基础环境? (y/n):" is_build_env
echo
echo
[[ $is_build_env =~ [Yy] ]] && pip3 install -r requirements.txt
read -n 1 -p "是否安装CUDA支持? (y/n): " is_cuda
[[ $is_cuda =~ [Yy] ]] && CMAKE_ARGS+=" -DWITH_CUDA=ON"
echo
echo
echo "CMAKE 清单 参数"
echo $CMAKE_ARGS
echo "Build 参数"
echo $BUILD_ARGS
echo
echo "顷刻间，安装即可完成，确认开始 (y/n)"
read -n 1 -p "> " confirmation
echo
echo
echo '(1/4) 创建 Build 文件夹'
mkdir -p build && cd build
echo '(2/4) 执行清单...'
cmake .. $CMAKE_ARGS
echo '(3/4) 运行构建'
make -j$(nproc)
echo '(4/4) 安装Python端'
cd .. && python3 setup.py install

cat <<-'EOF'
 _       __    ______    __    ______   ____     __  ___    ______
| |     / /   / ____/   / /   / ____/  / __ \   /  |/  /   / ____/
| | /| / /   / __/     / /   / /      / / / /  / /|_/ /   / __/   
| |/ |/ /   / /___    / /___/ /___   / /_/ /  / /  / /   / /___   
|__/|__/   /_____/   /_____/\____/   \____/  /_/  /_/   /_____/   
                                                                  
安装成功！感谢您使用Axono，我们诚邀您加入我们，您可以

1. 成为我们的 Contributor
> 现在出发！https://github.com/ByteRainTech/Axono/

2. 加入我们的QQ群（同样，您甚至可以成为我们官方的 Reviewer）
> 1014082546

TIP: 我们编译的文件会在Axono Python端的 /library/ 出现。

EOF
echo "这是一些安全性的询问，您可以按下Ctrl+C结束安装向导，也可以完成完整的安装流程"
read -n 1 -p "是否执行单元测试-CPU (y/n): " is_cpu_unittest
if [ "$is_cpu_unittest" = "y" ] || [ "$is_cpu_unittest" = "Y" ]; then
    cd ./python/tests/ && python3 run.py
fi
echo
echo
if [ "$is_cuda" = "y" ] || [ "$is_cuda" = "Y" ]; then
    read -n 1 -p "是否执行单元测试-CUDA (y/n): " is_cuda_unittest
    if [ "$is_cuda_unittest" = "y" ] || [ "$is_cuda_unittest" = "Y" ]; then
        export axono_default_device=cuda
        python3 run.py
    fi
    echo
    echo
fi
cat <<-'EOF'
完成！感谢您的耐心完成整个向导~喵！
您可以通过:
----------------
import axono
axono.welcome()
----------------
完成你在Axono的第一步

EOF
