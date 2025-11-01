import os
import shutil
import random
from pathlib import Path

# === CONFIGURACIÓN ===
DATASET_DIR = Path(".")  # Ejecutar dentro de yolo/
IMAGES_DIR = DATASET_DIR / "images"
LABELS_DIR = DATASET_DIR / "labels"

# Carpetas de salida
OUTPUT_DIR = DATASET_DIR / "dataset_split"
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.7, 0.2, 0.1

# Crear carpetas de salida
for split in ["train", "val", "test"]:
    (OUTPUT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

# === COMPROBAR QUÉ IMÁGENES NO TIENEN LABEL ===
images = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
labels = [f for f in os.listdir(LABELS_DIR) if f.lower().endswith(".txt")]

label_names = {os.path.splitext(f)[0] for f in labels}

missing_labels = [img for img in images if os.path.splitext(img)[0] not in label_names]

if missing_labels:
    print("Imágenes sin label correspondiente:")
    for img in missing_labels:
        print("  -", img)
else:
    print("Todas las imágenes tienen su label correspondiente.")

# === DIVIDIR EL DATASET ===
# Filtramos solo las imágenes que sí tienen su label
valid_images = [img for img in images if os.path.splitext(img)[0] in label_names]

# Barajar aleatoriamente
random.shuffle(valid_images)

# Calcular tamaños de splits
n = len(valid_images)
n_train = int(n * TRAIN_RATIO)
n_val = int(n * VAL_RATIO)

train_imgs = valid_images[:n_train]
val_imgs = valid_images[n_train:n_train + n_val]
test_imgs = valid_images[n_train + n_val:]

splits = {
    "train": train_imgs,
    "val": val_imgs,
    "test": test_imgs
}

# === COPIAR LOS ARCHIVOS ===
for split, imgs in splits.items():
    for img_name in imgs:
        base = os.path.splitext(img_name)[0]
        txt_name = base + ".txt"

        shutil.copy(IMAGES_DIR / img_name, OUTPUT_DIR / split / "images" / img_name)
        shutil.copy(LABELS_DIR / txt_name, OUTPUT_DIR / split / "labels" / txt_name)

print("División completada:")
print(f"  Train: {len(train_imgs)} imágenes")
print(f"  Val:   {len(val_imgs)} imágenes")
print(f"  Test:  {len(test_imgs)} imágenes")
