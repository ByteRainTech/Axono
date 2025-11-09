import axono
import numpy as np
from typing import List, Dict

def tokenize_text(text: str, vocab: Dict[str, int], max_len: int) -> np.ndarray:
    """简单的分词函数"""
    tokens = text.lower().split()
    token_ids = [vocab.get(token, vocab['<unk>']) for token in tokens]
    
    # 截断或填充到固定长度
    if len(token_ids) > max_len:
        token_ids = token_ids[:max_len]
    else:
        token_ids = token_ids + [vocab['<pad>']] * (max_len - len(token_ids))
    
    return np.array(token_ids)

def main():
    # 创建示例数据
    categories = ['politics', 'sports', 'technology', 'entertainment']
    vocab = {
        '<pad>': 0,
        '<unk>': 1,
        'the': 2, 'is': 3, 'in': 4, 'a': 5, 'to': 6,
        'politics': 7, 'sports': 8, 'tech': 9, 'news': 10
    }
    
    # 生成示例数据
    num_samples = 1000
    max_len = 50
    X = []
    y = []
    
    for _ in range(num_samples):
        category = np.random.choice(len(categories))
        # 生成随机文本
        text = f"this is {categories[category]} news " * 5
        X.append(tokenize_text(text, vocab, max_len))
        y.append(category)
    
    X = np.array(X)
    y = np.array(y)
    
    # 创建数据集
    class TextDataset(axono.data.Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = y
        
        def __getitem__(self, index):
            return {
                'inputs': axono.core.Tensor.from_numpy(self.X[index]),
                'targets': axono.core.Tensor.from_numpy(np.array(self.y[index]))
            }
        
        def __len__(self):
            return len(self.X)
    
    dataset = TextDataset(X, y)
    data_loader = axono.data.DataLoader(
        dataset,
        batch_size=16,
        shuffle=True
    )
    
    # 创建Transformer模型
    class TextClassifier(axono.nn.Module):
        def __init__(
            self,
            vocab_size: int,
            num_classes: int,
            d_model: int = 256,
            nhead: int = 4,
            num_encoder_layers: int = 3,
            device: str = "cuda:0"
        ):
            super().__init__()
            
            self.embedding = axono.nn.Embedding(
                vocab_size,
                d_model,
                device=device
            )
            
            self.transformer = axono.models.Transformer(
                d_model=d_model,
                nhead=nhead,
                num_encoder_layers=num_encoder_layers,
                device=device
            )
            
            self.classifier = axono.nn.Linear(
                d_model,
                num_classes,
                device=device
            )
        
        def forward(self, x):
            # x shape: [batch_size, seq_len]
            x = self.embedding(x)  # [batch_size, seq_len, d_model]
            x = self.transformer(x)  # [batch_size, seq_len, d_model]
            # 使用序列的平均池化作为特征
            x = x.mean(dim=1)  # [batch_size, d_model]
            x = self.classifier(x)  # [batch_size, num_classes]
            return x
    
    model = TextClassifier(
        vocab_size=len(vocab),
        num_classes=len(categories),
        device="cuda:0"
    )
    
    # 4. 创建优化器和训练器
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
    
    # 5. 创建可视化工具
    viz = axono.viz.TrainingVisualizer()
    model_viz = axono.viz.ModelVisualizer(model)
    
    # 6. 显示模型结构
    model_viz.plot()
    model_viz.summary()
    
    # 7. 训练模型
    trainer.fit(
        train_loader=data_loader,
        epochs=10,
        callbacks={
            "on_epoch_end": viz.update
        }
    )
    
    # 8. 可视化训练过程
    viz.plot_metrics()
    
    # 9. 测试预测
    model.eval()
    test_text = "this is technology news about artificial intelligence"
    test_tokens = tokenize_text(test_text, vocab, max_len)
    test_tensor = axono.core.Tensor.from_numpy(
        test_tokens[None, :]
    ).to("cuda:0")
    
    with axono.no_grad():
        pred = model(test_tensor)
        pred = pred.to("cpu").numpy()
        predicted_category = categories[pred.argmax()]
        print(f"预测类别: {predicted_category}")

if __name__ == "__main__":
    main()