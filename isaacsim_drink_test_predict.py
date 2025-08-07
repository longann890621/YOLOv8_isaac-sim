import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from ultralytics import YOLO
from glob import glob

# === Setup ===
model_path = "/home/robertlo/Desktop/usddrink_sticker_table_output/runs/usddrink_sticker_final/weights/best.pt"
data_dir = "/home/robertlo/Desktop/usddrink_sticker_table_output/yolov8_dataset"
output_dir = "/home/robertlo/Desktop/usddrink_sticker_table_output/runs/detect/test_predictions"
test_img_dir = os.path.join(data_dir, "images", "test")
test_lbl_dir = os.path.join(data_dir, "labels", "test")

# === Load model ===
model = YOLO(model_path)
results = model.predict(source=test_img_dir, save=True, save_txt=True, project=output_dir, name="predict")

# === Parse Predictions and Ground Truths ===
y_true = []
y_pred = []
iou_threshold = 0.5
pred_label_dir = os.path.join(output_dir, "predict", "labels")
image_files = sorted(glob(os.path.join(test_img_dir, "*.png")))

for img_path in image_files:
    name = os.path.splitext(os.path.basename(img_path))[0]
    gt_file = os.path.join(test_lbl_dir, f"{name}.txt")
    pred_file = os.path.join(pred_label_dir, f"{name}.txt")

    if not os.path.exists(gt_file):
        continue

    gt_lines = open(gt_file).read().strip().splitlines()
    gt_boxes = [list(map(float, line.strip().split())) for line in gt_lines]

    pred_boxes = []
    if os.path.exists(pred_file):
        pred_lines = open(pred_file).read().strip().splitlines()
        pred_boxes = [list(map(float, line.strip().split())) for line in pred_lines]

    matched_gt = set()
    for pb in pred_boxes:
        p_cls, px, py, pw, ph = pb
        p_x1, p_y1 = px - pw / 2, py - ph / 2
        p_x2, p_y2 = px + pw / 2, py + ph / 2

        best_iou = 0
        best_gt_idx = -1
        for idx, gb in enumerate(gt_boxes):
            if idx in matched_gt:
                continue
            g_cls, gx, gy, gw, gh = gb
            g_x1, g_y1 = gx - gw / 2, gy - gh / 2
            g_x2, g_y2 = gx + gw / 2, gy + gh / 2

            inter_x1 = max(p_x1, g_x1)
            inter_y1 = max(p_y1, g_y1)
            inter_x2 = min(p_x2, g_x2)
            inter_y2 = min(p_y2, g_y2)
            inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

            pred_area = pw * ph
            gt_area = gw * gh
            union_area = pred_area + gt_area - inter_area

            iou = inter_area / union_area if union_area > 0 else 0
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            y_true.append(int(gt_boxes[best_gt_idx][0]))
            y_pred.append(int(p_cls))
            matched_gt.add(best_gt_idx)

# === 防呆：若無資料就提前結束 ===
if not y_true or not y_pred:
    print("⚠️ 沒有收集到任何預測或標註資料，請確認模型是否成功預測，或標註是否對應正確。")
    exit()

# === Confusion Matrix ===
labels = ["can", "coffee", "glasses", "milk", "water_bottle"]
cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
cm_percent = np.divide(cm.astype(float), cm.sum(axis=1, keepdims=True), where=cm.sum(axis=1, keepdims=True) != 0) * 100

plt.figure(figsize=(7, 6))
sns.set(font_scale=1.1)
ax = sns.heatmap(
    cm_percent,
    annot=[[f"{int(cm[i][j])}\n({cm_percent[i][j]:.0f}%)" for j in range(len(labels))] for i in range(len(labels))],
    fmt="", cmap="Blues", xticklabels=labels, yticklabels=labels,
    linewidths=.5, cbar_kws={'label': 'Prediction %'}
)
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("YOLOv8 Confusion Matrix (with IoU Match)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "predict", "confusion_matrix.png"))
plt.close()

# === Classification Report ===
report = classification_report(y_true, y_pred, target_names=labels)
print(report)
with open(os.path.join(output_dir, "predict", "classification_report.txt"), "w") as f:
    f.write(report)

print("\n✅ Done. Confusion matrix and classification report saved in:", os.path.join(output_dir, "predict"))
