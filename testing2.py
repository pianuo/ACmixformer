import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random


class AudioDataset(Dataset):
    def __init__(self, num_samples, max_length):
        self.num_samples = num_samples
        self.max_length = max_length

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        length = random.randint(1, self.max_length)
        sample = torch.randn(length, 20)
        target = torch.randint(0, 2, (length,))
        return sample, target


class AudioCollate:
    def __call__(self, batch):
        # 获取 batch 中每个样本的长度
        lengths = [len(sample) for sample, _ in batch]

        # 将 batch 中的样本填充到相同的长度，并记录填充部分的位置
        data = torch.zeros(len(batch), max(lengths), 20)
        targets = torch.zeros(len(batch), max(lengths)).long()
        for i, (sample, target) in enumerate(batch):
            data[i, :len(sample)] = sample
            targets[i, :len(sample)] = target
        padding_mask = data.sum(dim=2) != 0

        return data, targets, padding_mask


class TransformerClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, num_heads, dropout):
        super().__init__()
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                input_size,
                num_heads,
                hidden_size,
                dropout
            ),
            num_layers
        )
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x 的 shape 是 [batch_size, seq_length, input_size]
        x = x.permute(1, 0, 2)  # 将 seq_length 移到第一维
        x = self.transformer_encoder(x)
        x = x[-1, :, :]  # 取最后一个时间步的输出作为整个序列的表示
        x = self.output_layer(x)
        return x


# 定义一些超参数
num_epochs = 10
batch_size = 5
num_samples_train = 30
num_samples_test = 20
max_length = 10
input_size = 20
hidden_size = 64
output_size = 2
num_layers = 2
num_heads = 4
dropout = 0.1

# 创建训练和测试数据集
train_dataset = AudioDataset(num_samples_train, max_length)
test_dataset = AudioDataset(num_samples_test, max_length)

# 创建 DataLoader
train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=AudioCollate(),
    drop_last=True
)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=AudioCollate(),
    drop_last=True
)

# 创建模型、优化器和损失函数
model = TransformerClassifier(input_size, hidden_size, output_size, num_layers, num_heads, dropout)
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# 训练模型
for epoch in range(num_epochs):
    for batch_idx, (data, target, padding_mask) in enumerate(train_dataloader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target.flatten())
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx}/{len(train_dataloader)}, Loss: {loss.item()}")

    # 在测试集上测试模型
    with torch.no_grad():
        total_correct = 0
        total_samples = 0
        for data, target, padding_mask in test_dataloader:
            output = model(data)
            predicted = output.argmax(dim=1)
            total_correct += (predicted == target.flatten()).sum().item()
            total_samples += predicted.shape[0] * predicted.shape[1]
        accuracy = total_correct / total_samples
        print(f"Epoch {epoch + 1}/{num_epochs}, Test accuracy: {accuracy}")

print("Training finished!")

