from ultralytics import YOLO

# 載入模型，可改成 yolov8n.pt / yolov8m.pt / yolov8l.pt 依需求
model = YOLO("yolov8s.pt")

# 開始訓練
model.train(
    data="/home/robertlo/Desktop/usddrink_sticker_table_output/yolov8_dataset/data.yaml",  # 訓練資料設定
    epochs=100,                      # 訓練 100 輪
    imgsz=640,                       # 輸入圖片大小
    batch=4,                         # 批次大小（依顯卡記憶體可再放大）
    name="usddrink_sticker_final",       # 訓練任務名稱
    project="/home/robertlo/Desktop/usddrink_sticker_table_output/runs",  # 儲存資料夾
    device=0,                        # ✅ 使用 GPU 編號 0
    workers=0,                       # ✅ 禁用多工（防止 dataloader 問題）
    amp=False,                       # ✅ 關掉自動混合精度，避免不穩定
    cache=False,                     # ✅ 不使用圖片快取
    deterministic=True,             # ✅ 訓練過程 deterministic（可重複性）
    verbose=True                     # ✅ 顯示詳細資訊
)


# ✅ 執行方式（在 VSCode 的終端機）：
# source ~/Downloads/.venv/bin/activate
# python ~/Downloads/VScode/isaacsim_fourth_stage/isaacsim_drink_yolov8_train.py