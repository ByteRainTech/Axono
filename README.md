# Axono
> 版本 `0.1.0`

<a href="https://cla-assistant.io/ByteRainTech/Axono"><img src="https://cla-assistant.io/readme/badge/ByteRainTech/Axono" alt="CLA assistant" /></a>

Axono——轻量级的人工智能算法库喵~

## 现已支持
精度:
- int8
- int16
- int32
- int64
- float32
- float64
> 精度接口见 `axono.core -> DataType`
> Tensor接口见 `axono.core -> Tensor`

#### 运算支持 `axono.core.operators`
- `matmul` (也可使用@运算)
```python
  c = a @ b
```
- `add` (也可使用+运算)
```python
  c = a + b
```
#### 算子支持 `axono.core.ops`
- `relu(x, inplace: bool=False)`

#### 通用转换支持 `axono.core`
- `to_numpy()` and `from_numpy` (与大多框架用法一致)

## 安装
> Linux
```bash
sh build_env.sh
sh build.sh
python setup.py install
```

## 单元测试

执行:
```bash
cd python/tests
python run.py
```
