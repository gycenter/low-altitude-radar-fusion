import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import glob
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence,pad_sequence
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from sklearn.metrics import classification_report
import random
import joblib

# 训练集航迹文件路径
data_dir = "dataset/航迹"
# 模型保存路径(仅展示)
# joblib.dump(dataset.scaler, "model/lstm_scaler.pkl")
# torch.save(model.state_dict(), "model/lstm_model_weights.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

def extract_motion_features(df):
    """
    输入：
      df: pd.DataFrame,包含航迹序列,必须含有列:
          '滤波距离', '滤波方位', '滤波俯仰',
          '全速度', 'X向速度', 'Y向速度', 'Z向速度', '航向'

    输出：
      features: np.ndarray,形状 (seq_len, feature_dim)
      新增运动特征如下：
        - 各速度分量一阶差分（加速度近似）
        - 速度大小一阶差分
        - 各速度分量的滑动窗口均值和标准差(窗口大小3)
        - 速度方向变化角度
        - 速度模长
    """

    # 复制数据，避免修改原df
    data = df.copy()

    # 计算速度模长
    data['speed_norm'] = np.sqrt(
        data['X向速度']**2 + data['Y向速度']**2 + data['Z向速度']**2
    )

    # 计算速度分量一阶差分(加速度近似)
    for axis in ['X向速度', 'Y向速度', 'Z向速度', 'speed_norm']:
        data[f'{axis}_diff'] = data[axis].diff().fillna(0)

    # 计算滑动窗口统计量（窗口大小=3）
    window = 3
    for axis in ['X向速度', 'Y向速度', 'Z向速度', 'speed_norm']:
        data[f'{axis}_mean'] = data[axis].rolling(window, min_periods=1).mean()
        data[f'{axis}_std'] = data[axis].rolling(window, min_periods=1).std().fillna(0)

    # 计算速度方向变化角度（利用航向角速度差分）
    data['航向_diff'] = data['航向'].diff().fillna(0).abs()
    # 如果航向角差大于180度，则调整为360 - 差值
    data['航向_diff'] = data['航向_diff'].apply(lambda x: x if x <= 180 else 360 - x)

    # 最终保留的特征列（数值）
    feature_cols = [
        '全速度',
        'X向速度', 'Y向速度', 'Z向速度',
        'speed_norm',
        'X向速度_diff', 'Y向速度_diff', 'Z向速度_diff', 'speed_norm_diff',
        'X向速度_mean', 'Y向速度_mean', 'Z向速度_mean', 'speed_norm_mean',
        'X向速度_std', 'Y向速度_std', 'Z向速度_std', 'speed_norm_std',
        '航向_diff',
    ]
    features = data[feature_cols].fillna(0).values.astype(np.float32)
    return features

class TrackSequenceDataset(Dataset):
    def __init__(self, data_dir):
        self.feature_cols = ['全速度', 'X向速度', 'Y向速度', 'Z向速度', '航向']
        self.samples = []
        all_features = []
        # 加载所有轨迹数据
        for fname in glob.glob(os.path.join(data_dir, "Tracks_*.txt")):
            if "_5_" in fname or "_6_" in fname:
                continue
            df = pd.read_csv(fname, encoding='gbk')
            if df[self.feature_cols].isnull().values.any():
                continue
            features = extract_motion_features(df)
            basename = os.path.basename(fname)
            parts = basename.split('_')
            label = int(parts[2]) - 1
            self.samples.append((features, label))
            all_features.append(features)

        # 标准化器 fit 所有样本的拼接
        all_stack = np.vstack(all_features)
        self.scaler = StandardScaler().fit(all_stack)

        # 缓存标准化结果
        self.samples = [
            (self.scaler.transform(x).astype(np.float32), y)
            for x, y in self.samples
        ]
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y), x.shape[0]  # 返回序列长度

class TrackLSTM(nn.Module):
    def __init__(self, input_dim=18, hidden_dim=64, num_classes=4):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, (h_n, _) = self.lstm(packed)
        last_hidden = h_n[-1]  # 取最后一层的输出
        return self.classifier(last_hidden)
def collate_fn(batch):
    batch.sort(key=lambda x: x[2], reverse=True)  # 按长度降序排序
    sequences, labels, lengths = zip(*batch)
    sequences_padded = pad_sequence(sequences, batch_first=True)
    return sequences_padded, torch.tensor(labels), torch.tensor(lengths)

dataset = TrackSequenceDataset(data_dir)
if __name__ == "__main__":
    # 保存 scaler 以备部署
    joblib.dump(dataset.scaler, "model/lstm_scaler.pkl")

    train_size = int(0.8 * len(dataset))
    train_set, val_set = random_split(dataset, [train_size, len(dataset) - train_size])

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=16, collate_fn=collate_fn)
        
    model = TrackLSTM().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(20):
        model.train()
        total_loss = 0
        for x, y, lengths in train_loader:
            x, y, lengths = x.to(device), y.to(device), lengths.to(device)
            out = model(x, lengths)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[Epoch {epoch+1}] Train Loss: {total_loss / len(train_loader):.4f}")

        # 验证
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for x, y, lengths in val_loader:
                x = x.to(device)
                out = model(x, lengths)
                pred = out.argmax(dim=1).cpu().numpy()
                preds.extend(pred)
                targets.extend(y.numpy())
        print(classification_report(targets, preds, digits=4))

    torch.save(model.state_dict(), "model/lstm_model_weights.pth")
