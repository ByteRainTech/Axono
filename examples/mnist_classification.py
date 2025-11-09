import axono
from datasets import load_dataset
import numpy as np
from tqdm import tqdm

def main():
    print("正在加载 MNIST 数据集...")
    dataset = load_dataset("mnist")
    
    transform = axono.data.transforms.Compose([
        axono.data.transforms.ToTensor(),
        axono.data.transforms.Normalize(
            mean=[0.1307],  # MNIST 数据集的均值
            std=[0.3081]    # MNIST 数据集的标准差
        )
    ])
    
    class MNISTDataset(axono.data.Dataset):
        def __init__(self, hf_dataset, transform=None):
            self.dataset = hf_dataset
            self.transform = transform
        
        def __getitem__(self, index):
            item = self.dataset[index]
            image = np.array(item['image']).astype(np.float32)
            image = image[None, :, :]  # 添加通道维度 [1, 28, 28]
            
            if self.transform:
                image = self.transform(image)
            
            return {
                'inputs': image,
                'targets': item['label']
            }
        
        def __len__(self):
            return len(self.dataset)
    
    # 创建数据加载器
    train_dataset = MNISTDataset(dataset['train'], transform=transform)
    test_dataset = MNISTDataset(dataset['test'], transform=transform)
    
    train_loader = axono.data.DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True
    )
    
    test_loader = axono.data.DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False
    )
    
    model = axono.models.CNN(
        input_channels=1,    # MNIST 是灰度图像
        num_classes=10,      # 数字 0-9
        hidden_channels=[32, 64],
        device="cuda:0"
    )
    
    # 创建优化器和训练器
    optimizer = axono.train.Adam(
        model.parameters(),
        lr=0.001
    )
    
    trainer = axono.train.Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn="cross_entropy",
        device="cuda:0"
    )
    
    # 创建可视化工具
    viz = axono.viz.TrainingVisualizer()
    model_viz = axono.viz.ModelVisualizer(model)
    feature_viz = axono.viz.FeatureVisualizer(model)
    
    # 显示模型结构
    print("\n模型结构:")
    model_viz.summary()
    
    # 8. 训练模型
    print("\n开始训练...")
    trainer.fit(
        train_loader=train_loader,
        valid_loader=test_loader,
        epochs=5,
        callbacks={
            "on_epoch_end": viz.update
        }
    )
    
    # 可视化训练过程
    viz.plot_metrics()
    
    # 在测试集上评估
    print("\n在测试集上评估...")
    model.eval()
    correct = 0
    total = 0
    
    with axono.no_grad():
        for batch in tqdm(test_loader):
            inputs = batch['inputs'].to("cuda:0")
            targets = batch['targets'].to("cuda:0")
            
            outputs = model(inputs)
            predictions = outputs.argmax(dim=1)
            
            correct += (predictions == targets).sum().item()
            total += targets.size(0)
    
    accuracy = 100.0 * correct / total
    print(f"\n测试集准确率: {accuracy:.2f}%")
    
    # 可视化特征图
    print("\n可视化第一个卷积层的特征图...")
    sample_batch = next(iter(test_loader))
    feature_viz.register_hooks(["features.0"])  # 第一个卷积层
    feature_viz.plot_feature_maps(
        sample_batch['inputs'][:4],  # 只显示4个样本的特征图
        "features.0",
        num_features=16
    )

if __name__ == "__main__":
    main()