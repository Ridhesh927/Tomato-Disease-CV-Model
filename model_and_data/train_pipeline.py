import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight

# ==========================================
# 1. CONFIGURATION (ADVANCED PIPELINE)
# ==========================================
# You can override this at runtime with env var TOMATO_DATASET_PATH.
DATASET_PATH = r"d:\dataset cv\archive (1)\PlantVillage"
MODEL_SAVE_PATH = "tomato_disease_model_efficientnetb3.h5"
CLASS_MAP_PATH = "class_mapping.json"

# Optional: set this to a real-world tomato leaf dataset path for true field evaluation.
REALWORLD_TEST_PATH = None

# Revert to 224x224 for MobileNetV2 speed
IMG_SIZE = (224, 224) 
BATCH_SIZE = 32 # Can fit 32 again
INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 15
GLOBAL_SEED = 1337

TOMATO_CLASSES = [
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]


def count_images_per_class(data_dir, classes):
    counts = {}
    for class_name in classes:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            counts[class_name] = 0
            continue
        valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        counts[class_name] = sum(
            1 for f in os.listdir(class_dir)
            if f.lower().endswith(valid_exts)
        )
    return counts


def print_train_style_distribution(data_dir):
    """Mimics quick check style: for folder in os.listdir("dataset/train")."""
    if not os.path.isdir(data_dir):
        print(f"Dataset path not found: {data_dir}")
        return

    print("\n--- Quick Class Distribution Check ---")
    for folder in sorted(os.listdir(data_dir)):
        class_path = os.path.join(data_dir, folder)
        if not os.path.isdir(class_path):
            continue
        valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        img_count = sum(1 for f in os.listdir(class_path) if f.lower().endswith(valid_exts))
        print(folder, img_count)


def resolve_dataset_path(config_path):
    """Resolve dataset path from env/config and common local folders."""
    env_path = os.environ.get("TOMATO_DATASET_PATH", "").strip()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    candidates = []
    if env_path:
        candidates.append(env_path)
    if config_path:
        candidates.append(config_path)

    # Helpful defaults if user keeps dataset inside project.
    candidates.extend([
        os.path.join(project_root, "dataset", "train"),
        os.path.join(project_root, "dataset", "PlantVillage"),
        os.path.join(project_root, "PlantVillage"),
    ])

    for candidate in candidates:
        if candidate and os.path.isdir(candidate):
            return candidate

    raise FileNotFoundError(
        "Dataset directory not found. Set TOMATO_DATASET_PATH or update DATASET_PATH.\n"
        "Expected folder structure like:\n"
        "<dataset_root>/Tomato_Bacterial_spot, <dataset_root>/Tomato_Early_blight, ...\n"
        "Example (Windows cmd): set TOMATO_DATASET_PATH=E:\\datasets\\PlantVillage"
    )


def compute_train_class_weights(train_dataset, num_classes):
    labels = []
    for _, y_batch in train_dataset:
        labels.extend(y_batch.numpy().tolist())

    labels = np.array(labels, dtype=np.int32)
    classes = np.arange(num_classes)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    return {int(i): float(w) for i, w in enumerate(weights)}


def save_class_mapping(class_names, file_path):
    mapping = {int(i): name for i, name in enumerate(class_names)}
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)
    print(f"Saved class mapping to {file_path}")


def load_realworld_test_dataset(data_dir, img_size, batch_size, classes):
    if not data_dir or not os.path.exists(data_dir):
        return None

    print(f"Loading optional real-world test dataset from: {data_dir}")
    dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=img_size,
        batch_size=batch_size,
        class_names=classes,
        shuffle=False,
    )
    return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

def load_and_split_dataset(data_dir, img_size, batch_size, classes):
    print(f"Loading datasets at resolution {img_size[0]}x{img_size[1]}...")
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.3,
        subset="training",
        seed=1337,
        image_size=img_size,
        batch_size=batch_size,
        class_names=classes
    )
    
    val_test_dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.3,
        subset="validation",
        seed=1337,
        image_size=img_size,
        batch_size=batch_size,
        class_names=classes
    )
    
    val_batches = tf.data.experimental.cardinality(val_test_dataset)
    val_dataset = val_test_dataset.take(val_batches // 2)
    test_dataset = val_test_dataset.skip(val_batches // 2)
    
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    
    return train_dataset, val_dataset, test_dataset

def build_advanced_model(num_classes, img_size):
    inputs = tf.keras.Input(shape=img_size + (3,))

    # Step 3 (mandatory): data augmentation block.
    if hasattr(tf.keras.layers, "RandomBrightness"):
        random_brightness = tf.keras.layers.RandomBrightness(0.2)
    else:
        random_brightness = tf.keras.layers.Lambda(lambda x: tf.image.random_brightness(x, max_delta=0.2))

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        random_brightness,
    ], name="data_augmentation")

    x = data_augmentation(inputs)
    
    # MobileNetV2 Requires Manual Rescaling! (unlike EfficientNet)
    x = tf.keras.layers.Rescaling(1./127.5, offset=-1)(x)
    
    # MobileNetV2 Base Model 
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=img_size + (3,),
        include_top=False,
        weights='imagenet'
    )
    
    # Initially freeze all layers of the base model
    base_model.trainable = False 
    
    x = base_model(x, training=False) # Keep BatchNorm layers in inference mode
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x) # Increased dropout for the larger model
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(label_smoothing=0.05),
        metrics=['accuracy']
    )
    return model, base_model

def get_callbacks():
    """Callbacks for Early Stopping, Learning Rate Reduction, and Model Checkpointing"""
    callbacks = [
        # Reduce LR when validation loss stops dropping
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.2, 
            patience=3, 
            min_lr=1e-6, 
            verbose=1
        ),
        # Stop training if validation loss hasn't improved in 5 epochs
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=5, 
            restore_best_weights=True,
            verbose=1
        ),
        # ALWAYS save the best version of the model dynamically during training
        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_SAVE_PATH,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    return callbacks

def plot_combined_history(history_initial, history_fine, initial_epochs):
    acc = history_initial.history['accuracy'] + history_fine.history['accuracy']
    val_acc = history_initial.history['val_accuracy'] + history_fine.history['val_accuracy']
    loss = history_initial.history['loss'] + history_fine.history['loss']
    val_loss = history_initial.history['val_loss'] + history_fine.history['val_loss']
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.plot([initial_epochs-1, initial_epochs-1], plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.plot([initial_epochs-1, initial_epochs-1], plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig('advanced_training_curves.png')
    plt.show()

def evaluate_model(model, test_dataset, class_names):
    print("\n--- Evaluating Advanced Model on Test Dataset ---")
    loss, accuracy = model.evaluate(test_dataset)
    print(f"\nFinal TEST Accuracy: {accuracy*100:.3f}%\n")
    
    y_true = []
    y_pred = []
    for images, labels in test_dataset:
        y_true.extend(labels.numpy())
        predictions = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(predictions, axis=-1))
        
    display_classes = [c.replace("Tomato_", "").replace("_", " ") for c in class_names]
    print(classification_report(y_true, y_pred, target_names=display_classes, digits=4))
    
    cm = confusion_matrix(y_true, y_pred)

    try:
        late_idx = class_names.index("Tomato_Late_blight")
        early_idx = class_names.index("Tomato_Early_blight")
        late_as_early = int(cm[late_idx, early_idx])
        late_total = int(np.sum(cm[late_idx]))
        print("\n--- Bias Check: Late Blight -> Early Blight ---")
        print(f"Late blight predicted as early blight: {late_as_early}/{late_total}")
    except ValueError:
        pass

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_classes)
    fig, ax = plt.subplots(figsize=(12, 12))
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=90)
    plt.title("Advanced Confusion Matrix")
    plt.tight_layout()
    plt.savefig('advanced_confusion_matrix.png')
    plt.show()

def main():
    tf.keras.utils.set_random_seed(GLOBAL_SEED)

    resolved_dataset_path = resolve_dataset_path(DATASET_PATH)
    print(f"Using dataset path: {resolved_dataset_path}")

    # Step 1 style output.
    print_train_style_distribution(resolved_dataset_path)

    counts = count_images_per_class(resolved_dataset_path, TOMATO_CLASSES)
    print("\n--- Class Distribution (Source Dataset) ---")
    for cname, count in counts.items():
        print(f"{cname}: {count}")

    train_ds, val_ds, test_ds = load_and_split_dataset(
        resolved_dataset_path, IMG_SIZE, BATCH_SIZE, TOMATO_CLASSES
    )

    class_weights = compute_train_class_weights(train_ds, len(TOMATO_CLASSES))
    print("\n--- Class Weights (Training Split) ---")
    for idx, weight in class_weights.items():
        print(f"Class {idx} ({TOMATO_CLASSES[idx]}): {weight:.4f}")

    save_class_mapping(TOMATO_CLASSES, CLASS_MAP_PATH)
    
    model, base_model = build_advanced_model(len(TOMATO_CLASSES), IMG_SIZE)
    callbacks = get_callbacks()
    
    # -------------------------------------------------------------
    # PHASE 1: Train just the new Classification Head (Base frozen)
    # -------------------------------------------------------------
    print("\n--- PHASE 1: Training Top Classification Layers ---")
    history_initial = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=INITIAL_EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks
    )
    
    # -------------------------------------------------------------
    # PHASE 2: Fine-Tuning the Deep Base Model
    # -------------------------------------------------------------
    print("\n--- PHASE 2: Unfreezing Base Model for Fine-Tuning ---")
    # Unfreeze the base model
    base_model.trainable = True

    # Step 4: unfreeze only last 30 layers.
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    for layer in base_model.layers[-30:]:
        layer.trainable = True
        
    # Recompile with a MUCH LOWER learning rate (1e-5 instead of 1e-3)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(label_smoothing=0.05),
        metrics=['accuracy']
    )
    
    model.summary()
    
    # Continue training from the end of phase 1
    total_epochs = INITIAL_EPOCHS + FINE_TUNE_EPOCHS
    history_fine = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=total_epochs,
        initial_epoch=history_initial.epoch[-1] + 1,
        class_weight=class_weights,
        callbacks=callbacks
    )
    
    plot_combined_history(history_initial, history_fine, INITIAL_EPOCHS)
    
    # Evaluate final state
    evaluate_model(model, test_ds, TOMATO_CLASSES)

    realworld_test_ds = load_realworld_test_dataset(
        REALWORLD_TEST_PATH, IMG_SIZE, BATCH_SIZE, TOMATO_CLASSES
    )
    if realworld_test_ds is not None:
        print("\n=== Real-World Dataset Evaluation ===")
        evaluate_model(model, realworld_test_ds, TOMATO_CLASSES)
    else:
        print("\nSkipping real-world evaluation (REALWORLD_TEST_PATH is not set).")

if __name__ == "__main__":
    main()
