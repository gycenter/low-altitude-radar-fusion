project_final/
├── MLP.py                                # MLP模型    
├── LSTM.py                               # LSTM模型    
├── init_file.py                          # 初始化dataset/data文件为雷达回波数据处理所需结构    
├── 雷达回波数据处理.ipynb                  # 雷达数据处理，回波模型
├── get_radar_info.py                     # 根据回波模型得到预测信息
├── make_three_fusion_data.py             # 生成融合数据
├── three_fusion_model.py                 # 三通道融合模型
├── predictor_three_fusion.py             # 根据融合模型预测
├── dataset                               # 数据集    
│   ├── data                              # 雷达处理所需回波数据集
│   ├── data_RAW                          # 雷达预处理数据集
│   ├── 原始回波  
│   ├── 点迹  
│   └── 航迹  
├── model                                 # 模型  
│   ├── complex_variable_width_model.pth  # 回波模型
│   ├── fusion_mlp_model.pth              # 融合模型
│   ├── lstm_model_weights.pth            # LSTM模型
│   ├── lstm_scaler.pkl                   # 航迹数据归一化    
│   ├── point_mlp_filtered.h5             # 点迹模型
│   └── point_mlp_scaler_filtered.pkl     # 点迹数据归一化
├── train_fusion_csv                      # 融合数据集
│   └── fusion_trainset_logits.csv