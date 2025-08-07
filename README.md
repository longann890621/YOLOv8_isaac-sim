# 🥤 YOLOv8 飲料物件偵測專案（含合成資料生成）

本專案實作了完整的物件偵測訓練流程，從 Isaac Sim 合成資料生成，到 YOLOv8 訓練與測試評估，皆自動化完成。訓練資料為五類常見飲料物件，包含：can、coffee、glasses、milk、water_bottle，並使用自製桌面環境與隨機位置擺放方式提升模型泛化能力。

---


---

## 🧪 1. 合成資料生成（Isaac Sim）

### 📝 功能摘要

- 隨機生成五種飲料物件於桌面場景
- 自動擺放避免重疊
- 使用 KittiWriter 拍攝資料
- 自動轉換 KITTI ➜ YOLOv8 格式
- 切分成 train/val/test 並輸出 `data.yaml`

### 🚀 執行方式

```bash
cd ~/AI_Nvidia/Isaac_Sim
./python.sh ~/isaacsim_usddrink_sticker_table.py

```

執行完成後，將在 ~/Desktop/usddrink_sticker_table_output 中看到：

- Camera/rgb/：合成影像
- labels_yolo/：YOLO 格式標註
- yolov8_dataset/：train/val/test 切分資料夾與 data.yaml

---
# 🏋️ 2. 訓練 YOLOv8 模型

### 📝 功能摘要
- 使用 ultralytics 官方 YOLOv8 介面
- 支援 yolov8s/m/l 等變體選擇
- 設定批次大小、訓練輪數、模型儲存位置等

### 🚀 執行方式

```bash
# 建議先進入虛擬環境
source ~/Downloads/.venv/bin/activate
python ~/isaacsim_drink_yolov8_train.py

```

訓練結果會儲存在 runs/usddrink_sticker_final/，包含 best.pt 權重檔。

---

## 🧠 3. 測試與推論分析

### 📝 功能摘要
- 載入訓練完成模型進行測試集推論
- 自動比對 ground truth 與預測結果
- 產出混淆矩陣（含 IoU 門檻比對）
- 輸出分類報告（precision / recall / F1-score）

### 🚀 執行方式

```bash
python ~/isaacsim_drink_test_predict.py

```
---

## 🛠 開發環境
- 🧠 Isaac Sim 4.5.0
- 🐍 Python 3.10+
- ⚡ YOLOv8 (ultralytics==8.x)
- 🧪 sklearn / matplotlib / seaborn


## 🚧 TODO / 可優化方向
- 支援多視角、多光源場景
- 引入 domain randomization
- 導入自動增強與資料平衡機制
- 模型量化與部署測試
