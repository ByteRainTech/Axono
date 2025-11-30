#!/usr/bin/env bash
set -e

PYTHON_ENV=${PYTHON_ENV:-"python3"}
AUTO_CI=${AUTO_CI:-""}
SKIP_ENV=${SKIP_ENV:-""}
COLOR=${COLOR:-"true"}

echo "若您的终端显示乱码，可执行 export COLOR=false 关闭颜色显示"
echo

if [ "$COLOR" = "true" ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    NC='\033[0m' # No Color
    BLUE='\033[0;34m'
    GREY='\033[1;30m'
    YELLOW='\033[1;33m'
else
    RED=''
    GREEN=''
    NC=''
    BLUE=''
    GREY=''
    YELLOW=''
fi

# 打印 logo
echo -ne "$BLUE"
cat <<-'EOF'
       db         8b        d8  ,ad8888ba,    888b      88    ,ad8888ba,    
      d88b         Y8,    ,8P  d8"'    `"8b   8888b     88   d8"'    `"8b   
     d8'`8b         `8b  d8'  d8'        `8b  88 `8b    88  d8'        `8b  
    d8'  `8b          Y88P    88          88  88  `8b   88  88          88  
   d8YaaaaY8b         d88b    88          88  88   `8b  88  88          88  
  d8""""""""8b      ,8P  Y8,  Y8,        ,8P  88    `8b 88  Y8,        ,8P  
 d8'        `8b    d8'    `8b  Y8a.    .a8P   88     `8888   Y8a.    .a8P   
d8'          `8b  8P        Y8  `"Y8888Y"'    88      `888    `"Y8888Y"'    
                            
EOF
echo -ne "$NC"
echo -e "${NC}Axono 安装向导"
echo -e "${GREEN}Github${NC}: ${YELLOW}https://github.com/ByteRainTech/Axono${NC}"


# 安装环境询问
echo
echo -e "${BLUE}使用的 Python 环境${NC}: ${YELLOW}$PYTHON_ENV${NC}"
echo -e "${GREY}> $(whereis $PYTHON_ENV)${NC}"
echo
if [ "$AUTO_CI" = "" ]; then
    # 安装基础环境
    echo -e "${GREY}Tips: 使用 AUTO_CI 可以启用非交互式安装（值为 cpu 只安装 cpu 支持，值为 cuda 可安装 cuda 支持），同时使用 SKIP_ENV 参数跳过基础环境安装${NC}"
    echo
    echo -e "${BLUE}是否安装基础环境?${NC}"
    echo -e "${GREY}如果安装时提示 ${RED}error: externally-managed-environment${GREY}，则可输入 ${GREEN}a${GREY} 启用 ${NC}--break-system-packages${GREY} 选项以安装到系统${NC}"
    echo -ne "${GREY}(${YELLOW}Y${GREY}/a/n):${NC} "
    read -n 1 -p "" is_build_env
else
    # 自动安装基础环境
    echo "当前处于非交互式安装模式。模式: $AUTO_CI"
    echo
    if [ "$SKIP_ENV" = "" ]; then
        echo "自动安装基础环境"
        is_build_env="a"
    else
        echo "跳过安装基础环境"
        is_build_env="n"
    fi
fi
echo
echo

# 安装基础环境
if [ "$is_build_env" = "y" ] || [ "$is_build_env" = "Y" ] || [ "$is_build_env" = "" ]; then
    $PYTHON_ENV -m pip install -r requirements.txt
    $PYTHON_ENV -m pip install setuptools wheel build # venv 构建避免缺包
fi
if [ "$is_build_env" = "A" ] || [ "$is_build_env" = "a" ]; then
    # --break-system-packages 选项强制安装
    $PYTHON_ENV -m pip install -r requirements.txt --break-system-packages
    $PYTHON_ENV -m pip install setuptools wheel build --break-system-packages # venv 构建避免缺包
fi

# 获取编译参数
CMAKE_ARGS=${CMAKE_ARGS:-"-Dpybind11_DIR=$($PYTHON_ENV -m pybind11 --cmakedir)"}
CMAKE_EXT_ARGS=${CMAKE_EXT_ARGS:-""}
CMAKE_ARGS+=" $CMAKE_EXT_ARGS"
MAKE_ARGS=${MAKE_ARGS:-"-j$(nproc)"}
BUILD_ARGS=${BUILD_ARGS:-"-O3"}

if [ "$AUTO_CI" = "" ]; then
    echo -ne "${BLUE}是否安装 CUDA 支持? ${GREY}(y/${YELLOW}N${GREY}):${NC} "
    read -n 1 -p "" is_cuda
else
    if [ "$AUTO_CI" = "cuda" ]; then
        echo "自动安装 CUDA 支持"
        is_cuda="y"
    else
        echo "跳过安装 CUDA 支持"
    fi
fi
[[ $is_cuda =~ [Yy] ]] && CMAKE_ARGS+=" -DWITH_CUDA=ON"
echo
echo
echo -e "${GREEN}CMake 参数${GREY} (环境变量 CMAKE_EXT_ARGS)${NC}"
echo -ne "${GREY}"
echo $CMAKE_ARGS
echo -ne "${NC}"
echo -e "${GREEN}Make 参数${GREY} (环境变量 MAKE_ARGS)${NC}"
echo -ne "${GREY}"
echo $MAKE_ARGS
echo -ne "${NC}"
echo -e "${GREEN}Build 参数${GREY} (环境变量 BUILD_ARGS)${NC}"
echo -ne "${GREY}"
echo $BUILD_ARGS
echo -ne "${NC}"
echo

if [ "$AUTO_CI" = "" ]; then
    echo -ne "${BLUE}顷刻间，安装即可完成，确认开始 ${GREY}(${YELLOW}Y${GREY}/n):${NC} "
    read -n 1 -p "" confirmation
    if [ "$confirmation" = "n" ] || [ "$is_build_env" = "N" ]; then
        echo
        echo "安装已取消"
        exit 1
    fi
    echo
fi
echo -e "${GREY}(${GREEN}1${GREY}/${GREEN}4${GREY}) ${BLUE}创建 Build 文件夹${NC}"
mkdir -p build && cd build
echo -e "${GREY}(${GREEN}2${GREY}/${GREEN}4${GREY}) ${BLUE}执行清单...${NC}"
cmake .. $CMAKE_ARGS
echo -e "${GREY}(${GREEN}3${GREY}/${GREEN}4${GREY}) ${BLUE}运行构建${NC}"
make $MAKE_ARGS EXTRA_CXXFLAGS="$BUILD_ARGS"
echo -e "${GREY}(${GREEN}4${GREY}/${GREEN}4${GREY}) ${BLUE}安装 Python 端${NC}"
# cd .. && python3 setup.py install
cd .. && python3 -m pip install .

echo -ne "${BLUE}"
cat <<-'EOF'
 _       __    ______    __    ______   ____     __  ___    ______
| |     / /   / ____/   / /   / ____/  / __ \   /  |/  /   / ____/
| | /| / /   / __/     / /   / /      / / / /  / /|_/ /   / __/   
| |/ |/ /   / /___    / /___/ /___   / /_/ /  / /  / /   / /___   
|__/|__/   /_____/   /_____/\____/   \____/  /_/  /_/   /_____/   
                                                          
EOF
echo -ne "${GREY}"
cat <<-'EOF'
安装成功！感谢您使用Axono，我们诚邀您加入我们，您可以

1. 成为我们的 Contributor
> 现在出发！https://github.com/ByteRainTech/Axono/

2. 加入我们的QQ群（同样，您甚至可以成为我们官方的 Reviewer）
> 1014082546

Tips: 我们编译的文件会在 Axono Python 端的 /library/ 出现。

EOF
echo -ne "${NC}"


if [ "$AUTO_CI" = "" ]; then
    echo -e"${BLUE}这是一些安全性的询问，您可以按下 ${YELLOW}Ctrl+C${BLUE} 结束安装向导，也可以完成完整的安装流程${NC}"
    echo -ne "${BLUE}是否执行单元测试-CPU? ${GREY}(y/${YELLOW}N${GREY}):${NC} "
    read -n 1 -p "" is_cpu_unittest
else
    echo "自动执行单元测试"
    is_cpu_unittest="y"
    is_cuda_unittest="y"
fi
if [ "$is_cpu_unittest" = "y" ] || [ "$is_cpu_unittest" = "Y" ]; then
    cd ./python/tests/ && $PYTHON_ENV run.py
fi
echo
echo
if [ "$is_cuda" = "y" ] || [ "$is_cuda" = "Y" ]; then
    echo -ne "${BLUE}是否执行单元测试-CUDA? ${GREY}(y/${YELLOW}N${GREY}):${NC} "
    read -n 1 -p "" is_cuda_unittest
    if [ "$is_cuda_unittest" = "y" ] || [ "$is_cuda_unittest" = "Y" ]; then
        export axono_default_device=cuda
        $PYTHON_ENV run.py
    fi
    echo
    echo
fi
echo -e "${GREEN}
完成！感谢您的耐心完成整个向导~喵！${BLUE}"
cat <<-'EOF'
您可以通过:
----------------
import axono
axono.welcome()
----------------
完成你在Axono的第一步

EOF
echo -ne "${NC}"
