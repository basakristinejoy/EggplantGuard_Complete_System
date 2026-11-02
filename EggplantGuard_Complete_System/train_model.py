"""Train a four-class MobileNetV2 classifier from your own labeled images."""
import os, json
import tensorflow as tf

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "dataset")
MODEL_DIR = os.path.join(ROOT, "model")
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 12
EXPECTED = [
    "Healthy Leaf",
    "Leaf Miner Damage",
    "Fresh-Market-Ready Fruit",
    "Damaged-Borer-Infested Fruit",
]

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.isdir(DATA_DIR):
        raise SystemExit("Dataset folder not found.")
    train = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, validation_split=0.2, subset="training", seed=42,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="categorical")
    val = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, validation_split=0.2, subset="validation", seed=42,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="categorical")
    names = train.class_names
    if len(names) != 4:
        raise SystemExit(f"Expected exactly 4 class folders; found {len(names)}: {names}")
    print("Detected class folders:", names)
    with open(os.path.join(MODEL_DIR, "class_names.json"), "w", encoding="utf-8") as f:
        json.dump(names, f, indent=2)

    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.10),
    ])
    base = tf.keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet")
    base.trainable = False
    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(4, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(os.path.join(MODEL_DIR, "eggplantguard.keras"),
                                           save_best_only=True)
    ]
    model.fit(train, validation_data=val, epochs=EPOCHS, callbacks=callbacks)
    model.save(os.path.join(MODEL_DIR, "eggplantguard.keras"))
    print("Training complete. Model saved to model/eggplantguard.keras")
    print("Evaluate with a separate held-out test set before reporting accuracy.")

if __name__ == "__main__":
    main()
