```
                                                                                                             
       db         8b        d8  ,ad8888ba,    888b      88    ,ad8888ba,                        ,a8888a,     
      d88b         Y8,    ,8P  d8"'    `"8b   8888b     88   d8"'    `"8b                     ,8P"'  `"Y8,   
     d8'`8b         `8b  d8'  d8'        `8b  88 `8b    88  d8'        `8b                   ,8P        Y8,  
    d8'  `8b          Y88P    88          88  88  `8b   88  88          88      8b       d8  88          88  
   d8YaaaaY8b         d88b    88          88  88   `8b  88  88          88      `8b     d8'  88          88  
  d8""""""""8b      ,8P  Y8,  Y8,        ,8P  88    `8b 88  Y8,        ,8P       `8b   d8'   `8b        d8'  
 d8'        `8b    d8'    `8b  Y8a.    .a8P   88     `8888   Y8a.    .a8P         `8b,d8'     `8ba,  ,ad8'   
d8'          `8b  8P        Y8  `"Y8888Y"'    88      `888    `"Y8888Y"'            "8"         "Y8888P"     

                                                                                                                                                                      
  ,ad8888ba,      ,ad8888ba,            88     ,a8888a,         88          ,d8       ,a8888a,      ad88888ba    ad888888b,  8888888888           ,d8      ad8888ba,  
 d8"'    `"8b    d8"'    `"8b         ,d88   ,8P"'  `"Y8,     ,d88        ,d888     ,8P"'  `"Y8,   d8"     "8b  d8"     "88  88                 ,d888     8P'    "Y8  
d8'        `8b  d8'        `8b      888888  ,8P        Y8,  888888      ,d8" 88    ,8P        Y8,  Y8a     a8P          a8P  88  ____         ,d8" 88    d8           
88          88  88          88          88  88          88      88    ,d8"   88    88          88   "Y8aaa8P"        ,d8P"   88a8PPPP8b,    ,d8"   88    88,dd888bb,  
88          88  88          88          88  88          88      88  ,d8"     88    88          88   ,d8"""8b,      a8P"      PP"     `8b  ,d8"     88    88P'    `8b  
Y8,    "88,,8P  Y8,    "88,,8P          88  `8b        d8'      88  8888888888888  `8b        d8'  d8"     "8b   a8P'                 d8  8888888888888  88       d8  
 Y8a.    Y88P    Y8a.    Y88P           88   `8ba,  ,ad8'       88           88     `8ba,  ,ad8'   Y8a     a8P  d8"          Y8a     a8P           88    88a     a8P  
  `"Y8888Y"Y8a    `"Y8888Y"Y8a          88     "Y8888P"         88           88       "Y8888P"      "Y88888P"   88888888888   "Y88888P"            88     "Y88888P"   
                                                                                                                                                                      
                                                                                                                                                                      
                                                                                                             
```

<img src="logo.png" width="230">

Axono 是一个轻量级的人工智能算法库，旨在为教学、研究与原型开发提供简洁可扩展的张量与算子接口。

> [查看Benchmark](benchmark.md)

## 主要特性
- 支持的数据精度：
  - int8、int16、int32、int64、float32、float64  
  （精度接口见 axono.core -> DataType）
- 张量抽象：axono.core -> Tensor
- 常用运算与算子（见 axono.core.operators / axono.core.ops）：
  - 矩阵乘法（matmul），支持 `@` 运算符
  - 加法（add），支持 `+` 运算符
  - 激活函数：`relu(x, inplace: bool=False)`
- NumPy 互操作：
  - Tensor.to_numpy()
  - axono.core.from_numpy(...)
- 设备支持：
  - CPU: `cpu`
  - NVIDIA GPU: `cuda:<id>`

## 安装（Linux）
```bash
# 编译安装
bash build.sh
```
```python
# 已构建安装
# cuda 11.8
pip install axono --index-url=https://download.axono.org/whl/cu118/
# cuda 12.6
pip install axono --index-url=https://download.axono.org/whl/cu126/
# cuda 12.5
pip install axono --index-url=https://download.axono.org/whl/cu125/
# cpu
pip install axono --index-url=https://download.axono.org/whl/cpu/
```
> Windows 系列暂未测试设备，故不提供安装方法。

## 单元测试
```bash
cd python/tests
python run.py
```
