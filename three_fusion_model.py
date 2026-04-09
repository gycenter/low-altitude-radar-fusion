import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 路径(仅展示)
# df = pd.read_csv("train_fusion_csv/fusion_trainset_logits.csv")
# best_model_path = "model/fusion_mlp_model.pth"

class FusionMLP(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=64, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# 1. 读取数据
df = pd.read_csv("train_fusion_csv/fusion_trainset_logits.csv")
X = df.drop(columns=['label']).values.astype(np.float32)
y = df['label'].values.astype(np.int64)

# 2. 划分训练和验证集
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

# 3. 构造DataLoader
train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FusionMLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 4. 训练模型
num_epochs = 30
best_val_acc = 0.0
best_model_path = "model/fusion_mlp_model.pth"

for epoch in range(num_epochs):
    model.train()
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

    # 验证
    model.eval()
    val_preds = []
    val_labels = []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            preds = outputs.argmax(dim=1)
            val_preds.append(preds.cpu().numpy())
            val_labels.append(batch_y.cpu().numpy())
    val_preds = np.concatenate(val_preds)
    val_labels = np.concatenate(val_labels)
    val_acc = accuracy_score(val_labels, val_preds)

    print(f"Epoch {epoch + 1}/{num_epochs}, Val Accuracy: {val_acc:.4f}")

    # 保存最优模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)
        print(f"保存新最优模型，准确率: {best_val_acc:.4f}")
