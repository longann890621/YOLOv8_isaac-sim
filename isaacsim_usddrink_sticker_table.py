from omni.isaac.kit import SimulationApp
import os
import shutil
import random
import yaml

# === 設定參數 ===
NUM_FRAMES = 2000
WIDTH = 960
HEIGHT = 544
DATA_DIR = os.path.expanduser("~/Desktop/usddrink_sticker_table_output")
YOLO_DIR = os.path.join(DATA_DIR, "labels_yolo")
YOLOV8_DIR = os.path.join(DATA_DIR, "yolov8_dataset")

print("[🟢] 啟動 SimulationApp...")
simulation_app = SimulationApp({"headless": True, "width": WIDTH, "height": HEIGHT})
for _ in range(10):
    simulation_app.update()

import omni.usd
import omni.replicator.core as rep
from omni.isaac.core.utils.stage import open_stage, get_current_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from pxr import Semantics

rep.settings.carb_settings("/omni/replicator/RTSubframes", 4)

# === 飲料模型與縮放 ===
DRINK_MODELS = {
    "can": "/home/robertlo/Downloads/fbx_usd/usd2/Canmodel_fbx.usd",
    "coffee": "/home/robertlo/Downloads/fbx_usd/usd2/Coffee Cup.usd",
    "glasses": "/home/robertlo/Downloads/fbx_usd/usd2/Glasses.usd",
    "milk": "/home/robertlo/Downloads/fbx_usd/usd2/Milk Carton.usd",
    "water_bottle": "/home/robertlo/Downloads/fbx_usd/usd2/Water bottle.usd"
}
SCALE_MAP = {
    "can": 1.8,
    "coffee": 0.07,
    "glasses": 1.0,
    "milk": 1,
    "water_bottle": 0.8
}
CLASS_NAMES = list(DRINK_MODELS.keys())
CLASS_NAME_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}

# === 背景與桌子 ===
def prefix_with_isaac(path):
    root = get_assets_root_path()
    if root is None:
        raise RuntimeError("❌ 找不到 Nucleus Assets")
    return root + path

ENV_URL = prefix_with_isaac("/Isaac/Environments/Simple_Warehouse/warehouse.usd")
TABLE_URL = "file:///home/robertlo/Downloads/fbx_usd/usd2/old_table.usd"
TABLE_TOP_Z = 0.82
TABLE_SCALE = 1.4

# === 語意標註過濾 ===
def update_semantics(stage, keep_classes):
    for prim in stage.Traverse():
        if prim.HasAPI(Semantics.SemanticsAPI):
            for prop in prim.GetProperties():
                if Semantics.SemanticsAPI.IsSemanticsAPIPath(prop.GetPath()):
                    instance_name = prop.SplitName()[1]
                    sem = Semantics.SemanticsAPI.Get(prim, instance_name)
                    if sem.GetSemanticDataAttr().Get() not in keep_classes:
                        prim.RemoveProperty(sem.GetSemanticTypeAttr().GetName())
                        prim.RemoveProperty(sem.GetSemanticDataAttr().GetName())
                        prim.RemoveAPI(Semantics.SemanticsAPI, instance_name)

# === 加入桌子（單獨一次或每幀都加）===
def add_table():
    try:
        table = rep.create.from_usd(TABLE_URL)
        with table:
            rep.modify.pose(position=(0, 0, 0), rotation=(0, 0, 0), scale=TABLE_SCALE)
        print("[🪑] 桌子成功加入場景")
    except Exception as e:
        print(f"[❌] 桌子載入失敗：{e}")

# === 加入飲料（每幀呼叫）===
def add_drinks():
    objs = []
    placed_positions = []

    def is_too_close(new_pos, existing_positions, min_dist=0.25):
        for pos in existing_positions:
            dx = new_pos[0] - pos[0]
            dy = new_pos[1] - pos[1]
            if (dx**2 + dy**2)**0.5 < min_dist:
                return True
        return False

    for cls_name, usd_path in DRINK_MODELS.items():
        scale = SCALE_MAP.get(cls_name, 1.0)
        for _ in range(10):
            pos_x = random.uniform(-0.6, 0.6)
            pos_y = random.uniform(-0.3, 0.3)
            if not is_too_close((pos_x, pos_y), placed_positions):
                try:
                    obj = rep.create.from_usd(f"file://{usd_path}", semantics=[("class", cls_name)])
                    with obj:
                        rep.modify.pose(position=(pos_x, pos_y, TABLE_TOP_Z), rotation=(0, 0, 0), scale=scale)
                    objs.append(obj)
                    placed_positions.append((pos_x, pos_y))
                    break
                except Exception as e:
                    print(f"[❌] 載入失敗：{cls_name}，錯誤：{e}")
    return rep.create.group(objs)

# === orchestrator 拍照 ===
def run_orchestrator():
    rep.orchestrator.run()
    while not rep.orchestrator.get_is_started():
        simulation_app.update()
    while rep.orchestrator.get_is_started():
        simulation_app.update()
    rep.BackendDispatch.wait_until_done()
    rep.orchestrator.stop()

# === KITTI ➜ YOLO 轉換 ===
def convert_kitti_to_yolo(kitti_dir, yolo_dir, img_width, img_height):
    os.makedirs(yolo_dir, exist_ok=True)
    for fname in os.listdir(kitti_dir):
        if not fname.endswith(".txt"):
            continue
        with open(os.path.join(kitti_dir, fname), "r") as f:
            lines = f.readlines()
        yolo_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            class_name = parts[0]
            class_id = CLASS_NAME_TO_ID.get(class_name, 0)
            left, top, right, bottom = map(float, parts[4:8])
            x_center = (left + right) / 2.0 / img_width
            y_center = (top + bottom) / 2.0 / img_height
            width = (right - left) / img_width
            height = (bottom - top) / img_height
            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        with open(os.path.join(yolo_dir, fname), "w") as out_f:
            out_f.write("\n".join(yolo_lines))

# === YOLOv8 資料集整理 ===
def prepare_yolov8_dataset(img_dir, label_dir, output_base, class_names, split_ratio=(0.8, 0.1, 0.1)):
    filenames = [f for f in os.listdir(label_dir) if f.endswith(".txt") and open(os.path.join(label_dir, f)).read().strip()]
    filenames.sort()
    random.shuffle(filenames)
    n = len(filenames)
    train_end = int(n * split_ratio[0])
    val_end = train_end + int(n * split_ratio[1])
    splits = {
        "train": filenames[:train_end],
        "val": filenames[train_end:val_end],
        "test": filenames[val_end:]
    }
    for split, files in splits.items():
        img_out = os.path.join(output_base, "images", split)
        label_out = os.path.join(output_base, "labels", split)
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(label_out, exist_ok=True)
        for fname in files:
            img_name = fname.replace(".txt", ".png")
            shutil.copy(os.path.join(DATA_DIR, "Camera/rgb", img_name), os.path.join(img_out, img_name))
            shutil.copy(os.path.join(label_dir, fname), os.path.join(label_out, fname))
    with open(os.path.join(output_base, "data.yaml"), "w") as f:
        yaml.dump({
            "train": os.path.abspath(os.path.join(output_base, "images/train")),
            "val": os.path.abspath(os.path.join(output_base, "images/val")),
            "test": os.path.abspath(os.path.join(output_base, "images/test")),
            "nc": len(class_names),
            "names": class_names
        }, f)

# === 主流程 ===
def main():
    print("[🟢] main() 開始執行...")
    open_stage(ENV_URL)
    stage = get_current_stage()
    for _ in range(30):
        simulation_app.update()
    print("[📦] 環境載入完成")

    update_semantics(stage, keep_classes=CLASS_NAMES)
    print("[🧩] 語意標註過濾完成")

    add_table()
    drinks = add_drinks()
    print("[🍹] 飲料與桌子加入完成")

    cam = rep.create.camera()

    with rep.trigger.on_frame(num_frames=NUM_FRAMES):
        with cam:
            rep.modify.pose(
                position=rep.distribution.uniform((0.3, 0.3, 0.5), (2.5, 2.5, 4.5)),
                look_at=rep.distribution.uniform((-0.3, -0.3, 0.84), (0.3, 0.3, 1.0))
            )
        with drinks:
            rep.modify.pose(
                position=rep.distribution.uniform((-0.6, -0.3, TABLE_TOP_Z), (0.6, 0.3, TABLE_TOP_Z)),
                rotation=rep.distribution.uniform((0, 0, 0), (0, 0, 360))
            )

    render_product = rep.create.render_product(cam, (WIDTH, HEIGHT))
    writer = rep.WriterRegistry.get("KittiWriter")
    writer.initialize(output_dir=DATA_DIR, omit_semantic_type=True)
    writer.attach(render_product)

    print("[📸] 開始拍攝...")
    run_orchestrator()
    print("[✅] 拍攝完成")

    print("[🔁] 開始轉換 KITTI ➜ YOLO")
    convert_kitti_to_yolo(os.path.join(DATA_DIR, "Camera/object_detection"), YOLO_DIR, WIDTH, HEIGHT)
    print("[✅] 轉換完成")

    print("[📁] 建立 YOLOv8 資料集結構")
    prepare_yolov8_dataset(os.path.join(DATA_DIR, "Camera/rgb"), YOLO_DIR, YOLOV8_DIR, CLASS_NAMES)
    print("[🏁] 全部完成")

if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()


#執行方式：cd ~/AI_Nvidia/Isaac_Sim
# ./python.sh ~/Downloads/VScode/isaacsim_fourth_stage/isaacsim_usddrink_sticker_table.py
