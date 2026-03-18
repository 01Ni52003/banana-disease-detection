import os, json, pathlib
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications.efficientnet import EfficientNetB0

# Basic settings
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
DATA_ROOT = pathlib.Path(__file__).resolve().parents[1] / "data" / "BananaLSD_Split"
MODEL_DIR = pathlib.Path(__file__).resolve().parents[0]

# Paths for saving
MODEL_EXPORT_PATH = MODEL_DIR / "banana_effnet.keras"   # ✅ .keras format
CLASS_MAP_PATH = MODEL_DIR / "class_map.json"

# ✅ Data Augmentation block
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

def get_datasets():
    data_root = DATA_ROOT
    train_dir = data_root / "train"
    val_dir = data_root / "val"

    if train_dir.exists() and any(train_dir.iterdir()) and val_dir.exists() and any(val_dir.iterdir()):
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            train_dir, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, shuffle=True
        )
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            val_dir, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, shuffle=True
        )
    else:
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_root, validation_split=0.2, subset="training", seed=42,
            image_size=IMAGE_SIZE, batch_size=BATCH_SIZE
        )
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_root, validation_split=0.2, subset="validation", seed=42,
            image_size=IMAGE_SIZE, batch_size=BATCH_SIZE
        )

    class_names = train_ds.class_names

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    return train_ds, val_ds, class_names

def build_model(num_classes: int):
    inputs = layers.Input(shape=(*IMAGE_SIZE, 3))
    x = data_augmentation(inputs)
    x = tf.keras.applications.efficientnet.preprocess_input(x)

    base = EfficientNetB0(include_top=False, input_tensor=x, weights="imagenet")
    base.trainable = True

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

def main():
    train_ds, val_ds, class_names = get_datasets()
    print("Classes:", class_names)
    num_classes = len(class_names)

    model = build_model(num_classes)

    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

    # 🔽 Save model (.keras format)
    model.save(MODEL_EXPORT_PATH)

    # 🔽 Save class map
    with open(CLASS_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump({i: name for i, name in enumerate(class_names)}, f, ensure_ascii=False, indent=2)

    print("✅ Model saved to:", MODEL_EXPORT_PATH)
    print("✅ Class map saved to:", CLASS_MAP_PATH)

if __name__ == "__main__":
    main()
