"""
Model training with MLflow experiment tracking.
Baseline: Simple CNN for Cats vs Dogs binary classification.
"""

import logging
from pathlib import Path

try:
    import mlflow
    import mlflow.keras
    MLFLOW_AVAILABLE = True
except ImportError as e:
    logging.warning(f"MLflow not available: {e}. Training without experiment tracking.")
    MLFLOW_AVAILABLE = False

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build_cnn(input_shape=(224, 224, 3), num_classes=2):
    """Build a simple CNN baseline."""
    import tensorflow as tf
    from tensorflow.keras import layers

    model = tf.keras.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_and_track(
    data_path: str = "data/processed/dataset.npz",
    epochs: int = 3,
    batch_size: int = 32,
    experiment_name: str = "cats-vs-dogs",
):
    """Train model and log to MLflow."""
    Path("models").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    data = np.load(data_path)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    if MLFLOW_AVAILABLE:
        mlflow.set_experiment(experiment_name)
        mlflow.start_run(run_name="cnn-baseline")

    model = build_cnn()
    if MLFLOW_AVAILABLE:
        mlflow.log_params({
            "model": "simple_cnn",
            "epochs": epochs,
            "batch_size": batch_size,
            "input_shape": list(X_train.shape[1:]),
        })

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )

    # Log metrics
    train_acc = history.history["accuracy"][-1]
    val_acc = history.history["val_accuracy"][-1]
    if MLFLOW_AVAILABLE:
        mlflow.log_metrics({
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
        })

    # Eval on test
    y_pred = np.argmax(model.predict(X_test), axis=1)
    test_acc = accuracy_score(y_test, y_pred)
    if MLFLOW_AVAILABLE:
        mlflow.log_metrics({
            "test_accuracy": test_acc,
            "test_precision": precision_score(y_test, y_pred, zero_division=0),
            "test_recall": recall_score(y_test, y_pred, zero_division=0),
            "test_f1": f1_score(y_test, y_pred, zero_division=0),
        })

    # Confusion matrix artifact
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["cat", "dog"], yticklabels=["cat", "dog"])
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    cm_path = "logs/confusion_matrix.png"
    plt.savefig(cm_path, dpi=100)
    plt.close()
    if MLFLOW_AVAILABLE:
        mlflow.log_artifact(cm_path)

    # Loss curve
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history.history["loss"], label="train")
    ax.plot(history.history["val_loss"], label="val")
    ax.set_title("Loss Curve")
    ax.legend()
    plt.tight_layout()
    loss_path = "logs/loss_curve.png"
    plt.savefig(loss_path, dpi=100)
    plt.close()
    if MLFLOW_AVAILABLE:
        mlflow.log_artifact(loss_path)
        mlflow.keras.log_model(model, "model")

    # Save for inference service (.h5 for reproducibility)
    model.save("models/model.h5")
    logger.info(f"Model saved to models/model.h5, test_acc={test_acc:.4f}")
    return model
