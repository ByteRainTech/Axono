```
                                                                                                             
       db         8b        d8  ,ad8888ba,    888b      88    ,ad8888ba,                        ,a8888a,     
      d88b         Y8,    ,8P  d8"'    `"8b   8888b     88   d8"'    `"8b                     ,8P"'  `"Y8,   
     d8'`8b         `8b  d8'  d8'        `8b  88 `8b    88  d8'        `8b                   ,8P        Y8,  
    d8'  `8b          Y88P    88          88  88  `8b   88  88          88      8b       d8  88          88  
   d8YaaaaY8b         d88b    88          88  88   `8b  88  88          88      `8b     d8'  88          88  
  d8""""""""8b      ,8P  Y8,  Y8,        ,8P  88    `8b 88  Y8,        ,8P       `8b   d8'   `8b        d8'  
 d8'        `8b    d8'    `8b  Y8a.    .a8P   88     `8888   Y8a.    .a8P         `8b,d8'     `8ba,  ,ad8'   
d8'          `8b  8P        Y8  `"Y8888Y"'    88      `888    `"Y8888Y"'            "8"         "Y8888P"     
                                                                                                             
                                                                                                             
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

> 请注意：当前为早期开发版（0.1.0），请在正式 release 后再用于生产环境。

## 安装（Linux）
```bash
sh build.sh
```

## 快速示例
### 天气预报
```python
import numpy as np
from axono.core import Tensor
from axono.train.optimizer import Adam
import pandas as pd

class WeatherMLP:
    def __init__(self, input_size=4, hidden_size=8, output_size=1):
        # 初始化权重
        self.W1 = Tensor(np.random.randn(input_size, hidden_size) * 0.01)
        self.b1 = Tensor(np.zeros(hidden_size))
        self.W2 = Tensor(np.random.randn(hidden_size, output_size) * 0.01)
        self.b2 = Tensor(np.zeros(output_size))
        
    def forward(self, x):
        # 前向传播
        self.x = Tensor(x)
        self.h = (self.x @ self.W1 + self.b1).relu()
        self.y_pred = self.h @ self.W2 + self.b2
        return self.y_pred
    
    def backward(self, grad):
        # 反向传播
        self.y_pred.backward(grad)

def main():
    # 模拟气象数据 (温度、湿度、气压、风速)
    np.random.seed(42)
    n_samples = 1000
    X = np.random.randn(n_samples, 4)
    # 模拟天气预测 (0: 晴天, 1: 雨天)
    y = (0.3 * X[:, 0] + 0.2 * X[:, 1] + 0.1 * X[:, 2] + 0.4 * X[:, 3] > 0).astype(float)
    y = y.reshape(-1, 1)

    # 创建模型
    model = WeatherMLP()
    optimizer = Adam([model.W1, model.b1, model.W2, model.b2], lr=0.01)

    # 训练模型
    batch_size = 32
    epochs = 100
    
    for epoch in range(epochs):
        total_loss = 0
        for i in range(0, n_samples, batch_size):
            batch_X = X[i:i+batch_size]
            batch_y = y[i:i+batch_size]
            
            # 前向传播
            y_pred = model.forward(batch_X)
            
            # 计算loss (MSE)
            loss = ((y_pred - Tensor(batch_y)) ** 2).mean()
            total_loss += loss.data
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss/n_samples:.4f}")

    # 测试模型
    test_X = np.array([[25.0, 0.8, 1013.0, 15.0]])  # 示例：温度25℃，湿度80%，气压1013hPa，风速15m/s
    pred = model.forward(test_X)
    print(f"预测结果: {'有可能下雨' if pred.data[0, 0] > 0.5 else '可能晴天'}")

if __name__ == "__main__":
    main()

```

## 单元测试
```bash
cd python/tests
python run.py
```
