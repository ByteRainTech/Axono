import numpy as np
from sklearn.preprocessing import MinMaxScaler

import axono


def create_sequences(data, seq_length):
    """创建时序序列"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : (i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


def main():
    # 这里使用随机数据模拟股票价格，实际使用时需要真实数据
    np.random.seed(42)
    time_steps = 1000
    stock_prices = np.random.randn(time_steps).cumsum()

    # 数据预处理
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(stock_prices.reshape(-1, 1))

    # 创建序列数据
    seq_length = 10
    X, y = create_sequences(scaled_data, seq_length)

    # 创建数据集和加载器
    class TimeSeriesDataset(axono.data.Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = y

        def __getitem__(self, index):
            return {
                "inputs": axono.core.Tensor.from_numpy(self.X[index]),
                "targets": axono.core.Tensor.from_numpy(self.y[index]),
            }

        def __len__(self):
            return len(self.X)

    dataset = TimeSeriesDataset(X, y)
    data_loader = axono.data.DataLoader(dataset, batch_size=32, shuffle=True)

    # 创建LSTM模型
    model = axono.models.LSTM(
        input_size=1, hidden_size=64, num_layers=2, dropout=0.1, device="cuda:0"
    )

    # 创建优化器和训练器
    optimizer = axono.train.Adam(model.parameters(), lr=0.01)

    trainer = axono.train.Trainer(
        model=model, optimizer=optimizer, loss_fn="mse", device="cuda:0"
    )

    # 创建可视化工具
    viz = axono.viz.TrainingVisualizer()

    # 训练模型
    trainer.fit(
        train_loader=data_loader, epochs=50, callbacks={"on_epoch_end": viz.update}
    )

    # 可视化训练过程
    viz.plot_metrics()

    # 预测
    model.eval()
    with axono.no_grad():
        # 获取最后一个序列作为输入
        last_seq = X[-1:]
        last_seq_tensor = axono.core.Tensor.from_numpy(last_seq).to("cuda:0")

        # 预测下一个值
        pred = model(last_seq_tensor)
        pred = pred.to("cpu").numpy()

        # 反归一化
        pred = scaler.inverse_transform(pred)
        print(f"预测的下一个股票价格: {pred[0][0]:.2f}")


if __name__ == "__main__":
    main()
