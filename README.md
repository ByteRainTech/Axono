# Axono
> 版本 `0.1.0`

Axono——轻量级的大数据库喵~

## 现已支持
精度:
- int8
- int16
- int32
- int64
- float32
- float64
> 精度接口见 `axono.core -> DataType`
> Tensor接口见 `axono.core -> tensor.Tensor`
运算支持
- `matmul` (也可使用@运算)

通用转换支持
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
