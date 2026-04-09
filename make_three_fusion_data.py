import os
import numpy as np
import pandas as pd
import torch
from keras.models import load_model
import joblib
from get_radar_info import *  # get_echo_logits 函数
import torch.nn as nn
import csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 路径配置
train_point_dir = "dataset/点迹"
train_track_dir = "dataset/航迹"
train_echo_dir = "dataset/原始回波"

point_model_path  = 'model/point_mlp_filtered.h5'
point_scaler_path = 'model/point_mlp_scaler_filtered.pkl'
lstm_weights_path = 'model/lstm_model_weights.pth'
lstm_scaler_path  = 'model/lstm_scaler.pkl'
echo_model_path   = "model/complex_variable_width_model.pth"

# 融合训练集保存路径
out_csv = "train_fusion_csv/fusion_trainset_logits.csv"

# 点迹特征顺序
point_feature_order = [
    "doppler_mean", "doppler_std", "doppler_max",
    "doppler_acc_mean", "doppler_acc_std",
    "snr_min", "snr_range", "snr_peak_ratio", "snr_jump_std", "snr_slope_mean",
    "pointnum_mean", "pointnum_max",
    "power_std", "power_mean"
]

# 加载模型和标准化器
point_model = load_model(point_model_path)
point_scaler = joblib.load(point_scaler_path)

class TrackLSTMClassifier(nn.Module):
    def __init__(self, input_size=18, hidden_size=64, num_classes=4, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.classifier(out)

lstm_model = TrackLSTMClassifier()
lstm_model.load_state_dict(torch.load(lstm_weights_path, map_location=device,weights_only=False))
lstm_model.to(device)
lstm_model.eval()
lstm_scaler = joblib.load(lstm_scaler_path)

def extract_point_features(df):
    df = df.dropna()
    f = {}
    f["doppler_mean"] = df["多普勒速度"].mean()
    f["doppler_std"] = df["多普勒速度"].std()
    f["doppler_max"] = df["多普勒速度"].max()

    # 加速度特征（多普勒速度一阶差分）
    doppler_diff = df["多普勒速度"].diff().dropna()
    f["doppler_acc_mean"] = doppler_diff.mean()
    f["doppler_acc_std"] = doppler_diff.std()

    snr_value = df["信噪比"]
    snr = 10 * np.log10(snr_value)   # 以db为单位
    f["snr_min"] = snr.min()
    f["snr_range"] = snr.max() - snr.min()
    f["snr_peak_ratio"] = snr.max() / (snr.mean() + 1e-5)
    f["snr_jump_std"] = np.std(np.diff(snr))
    f["snr_slope_mean"] = np.mean(np.diff(snr))

    f["pointnum_mean"] = df["原始点数量"].mean()
    f["pointnum_max"] = df["原始点数量"].max()

    f["power_std"] = df["和幅度"].std()
    f["power_mean"] = df["和幅度"].mean()
    return f

def extract_motion_features(df):
    """
    输入：
      df: pd.DataFrame,包含航迹序列,必须含有列:
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

class TrackLSTMClassifier(nn.Module):
    def __init__(self, input_size=18, hidden_size=64, num_classes=4, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.classifier(out)
    
# ------------------ 加载点迹模型 -------------------
point_model = load_model(point_model_path)
point_scaler = joblib.load(point_scaler_path)

def get_point_logits_from_df(df):
    features = extract_point_features(df)
    X = np.array([[features[f] for f in point_feature_order]], dtype=np.float32)
    X_scaled = point_scaler.transform(X)
    logits = point_model.predict(X_scaled, verbose=0)
    return logits.flatten()

def get_track_logits_from_df(df):
    X = extract_motion_features(df)
    X_scaled = lstm_scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = lstm_model(X_tensor)
    return logits.cpu().numpy().flatten()

fusion_dataset = []

for fname in os.listdir(train_point_dir):
    print("正在处理文件：", fname)
    if not fname.endswith(".txt") or not fname.startswith("PointTracks_"):
        continue
    try:
        parts = fname.replace(".txt", "").split("_")
        batch_id = parts[1]
        label = int(parts[2])
        index = parts[3]
    except Exception as e:
        print(f"无法解析文件名：{fname}, 错误: {e}")
        continue

    if label > 4:
        continue

    point_path = os.path.join(train_point_dir, fname)
    track_path = os.path.join(train_track_dir, f"Tracks_{batch_id}_{label}_{index}.txt")
    echo_path  = os.path.join(train_echo_dir, f"{batch_id}_Label_{label}.dat")

    if not (os.path.exists(point_path) and os.path.exists(track_path) and os.path.exists(echo_path)):
        print(f"缺少文件：{point_path} 或 {track_path} 或 {echo_path}")
        continue

    try:
        point_df = pd.read_csv(point_path, encoding="gbk")
        track_df = pd.read_csv(track_path, encoding="gbk")

        echo_logits_dict = get_echo_logits(echo_model_path, echo_path, device, num_classes=4)

        N = len(track_df)

        for i in range(5, N):
            frame_id = i + 1
            if frame_id not in echo_logits_dict:
                continue
            echo_logits = echo_logits_dict[frame_id]
            if echo_logits.shape[0] != 4:
                continue

            point_logits = get_point_logits_from_df(point_df.iloc[:i+1])
            track_logits = get_track_logits_from_df(track_df.iloc[:i+1])

            if point_logits.shape[0] !=4 or track_logits.shape[0] !=4:
                continue

            fused_logits = np.concatenate([point_logits, track_logits, echo_logits])  # 12维
            row = np.append(fused_logits, label - 1)  # 标签从0开始
            fusion_dataset.append(row)

    except Exception as e:
        print(f"处理文件 {fname} 出错: {e}")
        continue

fusion_array = np.array(fusion_dataset)

os.makedirs("train_fusion_csv", exist_ok=True)
out_csv = "train_fusion_csv/fusion_trainset_logits.csv"

with open(out_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    header = [f'point_logit_{i}' for i in range(4)] + \
             [f'track_logit_{i}' for i in range(4)] + \
             [f'echo_logit_{i}' for i in range(4)] + ['label']
    writer.writerow(header)
    writer.writerows(fusion_array)

print(f"融合训练集已保存至 {out_csv}")
