import os
import shutil
import random
import zipfile
import yaml
import torch
import cv2
import glob
from pathlib import Path
from ultralytics import YOLO

# 1. CONFIGURATION

CONFIG = {
    "dataset_zip":   "ELEC3.yolov8.zip",
    "dataset_dir":   "ELEC3_Dataset_Fixed",
    "bg_images_dir": "backgrounds_for_yolo", 
    "model_size":    "yolov8n.pt",           
    "epochs":        100,
    "imgsz":         640,
    "batch":         16,
    "run_name":      "balanced_amr_sponge",
    "device":        "0" if torch.cuda.is_available() else "cpu",
    "export_name":   "best.pt",
}

# 2. DATASET PREP (Extraction & Validation Check)

def prepare_robust_dataset(cfg):
    print("\n" + "="*50)
    print("[1/4] PREPARING DATASET")
    print("="*50)
    
    base_dir = Path.cwd()
    extract_dir = base_dir / cfg["dataset_dir"]
    zip_path = base_dir / cfg["dataset_zip"]
    bg_source = base_dir / cfg["bg_images_dir"]

    if not zip_path.exists():
        print(f" ERROR: {zip_path.name} not found!")
        return None
    
    if extract_dir.exists(): shutil.rmtree(extract_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    yaml_path = next(extract_dir.rglob("data.yaml"), None)
    dataset_root = yaml_path.parent
    train_img = dataset_root / "train" / "images"
    train_lbl = dataset_root / "train" / "labels"
    valid_img = dataset_root / "valid" / "images"
    valid_lbl = dataset_root / "valid" / "labels"

    # Fix Empty Validation
    if not valid_img.exists() or not any(valid_img.iterdir()):
        valid_img.mkdir(parents=True, exist_ok=True)
        valid_lbl.mkdir(parents=True, exist_ok=True)
        imgs = list(train_img.glob("*.*"))
        random.shuffle(imgs)
        for img_p in imgs[:int(len(imgs)*0.2)]:
            shutil.move(str(img_p), str(valid_img / img_p.name))
            lbl_p = train_lbl / (img_p.stem + ".txt")
            if lbl_p.exists(): shutil.move(str(lbl_p), str(valid_lbl / lbl_p.name))

    # Inject Background Noise 
    if bg_source.exists():
        bg_files = list(bg_source.glob("*.*"))
        for bg_file in bg_files:
            img = cv2.imread(str(bg_file))
            if img is None: continue
            img = cv2.resize(img, (640, 640))
            save_path = train_img / f"noise_{bg_file.name}"
            cv2.imwrite(str(save_path), img)
            with open(train_lbl / f"noise_{bg_file.stem}.txt", 'w') as f:
                pass 
        print(f" Injected {len(bg_files)} Hard Negative images.")

    # Fix YAML
    with open(yaml_path, 'r') as f:
        y_data = yaml.safe_load(f)
    y_data['path'] = str(dataset_root.absolute())
    y_data['train'], y_data['val'] = "train/images", "valid/images"
    with open(yaml_path, 'w') as f:
        yaml.dump(y_data, f, default_flow_style=False)
        
    return str(yaml_path)

# 3. BALANCED TRAINING 

def start_robust_training(yaml_file):
    print("\n" + "="*50)
    print(f"[2/4] TRAINING BALANCED MODEL")
    print("="*50)
    
    model = YOLO(CONFIG["model_size"])
    model.train(
        data=yaml_file,
        epochs=CONFIG["epochs"],
        imgsz=CONFIG["imgsz"],
        batch=CONFIG["batch"],
        name=CONFIG["run_name"],
        device=CONFIG["device"],
        
        # BALANCED PARAMETERS
        cls=3.0,             
        label_smoothing=0.05, 
        box=7.5,             
        close_mosaic=15,     
        
        # DISTANCE & MOTION AUGMENTATIONS
        scale=0.9,           
        flipud=0.0,          
        hsv_v=0.6,           
        hsv_s=0.7,           
        degrees=10.0,        
        mosaic=1.0           
    )
    return model

# 4. SMART EXPORT AND SENSITIVE CAMERA TEST

def export_and_test():
    print("\n" + "="*50)
    print("[3/4] EXPORTING MODEL")
    print("="*50)
    
    search_path = os.path.join("runs", "detect", f"{CONFIG['run_name']}*")
    folders = glob.glob(search_path)
    if not folders: return
    
    latest_folder = max(folders, key=os.path.getctime)
    best_pt = Path(latest_folder) / "weights" / "best.pt"
    local_pt = Path.cwd() / CONFIG["export_name"]

    if best_pt.exists():
        shutil.copy(best_pt, local_pt)
        print(f"SUCCESS: {CONFIG['export_name']} saved locally.")
    
    print("\n" + "="*50)
    print("[4/4] STARTING CAMERA TRACKING (HIGH SENSITIVITY)")
    print("="*50)

    model = YOLO(str(local_pt))
    
    # TEST SETTINGS FOR DISTANCE
    results = model.track(
        source=0, 
        show=True, 
        stream=True, 
        conf=0.50,   
        iou=0.45, 
        persist=True
    )

    for r in results:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Setup background folder
    Path(CONFIG["bg_images_dir"]).mkdir(exist_ok=True)
    
    yaml_file = prepare_robust_dataset(CONFIG)
    if yaml_file:
        start_robust_training(yaml_file)
        export_and_test()