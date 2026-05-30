"""
evaluate.py
-----------
Evaluacion completa de los dos modelos (CNN propia y Transfer Learning)
sobre el conjunto de test.

Genera para cada modelo:
  - Curvas de aprendizaje (loss/accuracy)
  - Matriz de confusion (heatmap)
  - Classification report (CSV + barras)
  - Curva ROC y Precision-Recall del detector binario
  - Analisis de umbral FPR/FNR -> theta optimo

Al final guarda models/threshold.json para usarlo en la app.

Uso:
    python src/evaluate.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve, average_precision_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import BASE_DIR
from train import AudioFeatureDataset
from train_transfer import EmbeddingDataset
from model_cnn import AudioCNN
from model_transfer import TransferHead

# Clase "background" = label_id 10
BG_ID = 10


# ─── Funcion auxiliar: inferencia con softmax ────────────────────────────

def predict_with_probs(model, loader, device):
    """Devuelve (targets, predicciones, probabilidades_softmax)."""
    model.eval()
    targets, probs = [], []
    with torch.no_grad():
        for X, y in loader:
            p = F.softmax(model(X.to(device)), dim=1).cpu().numpy()
            probs.append(p)
            targets.extend(y.numpy())
    probs = np.concatenate(probs)
    targets = np.array(targets)
    return targets, probs.argmax(1), probs


# ─── Funcion auxiliar: curvas de aprendizaje ─────────────────────────────

def plot_learning_curves(history_paths, labels, colors, save_path, split_type="both"):
    """Dibuja loss y accuracy por epoch para uno o mas modelos."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for path, label, (c1, c2) in zip(history_paths, labels, colors):
        with open(path) as f:
            h = json.load(f)
        ep = range(1, len(h["train_loss"]) + 1)
        
        if split_type in ["both", "train"]:
            ls_train = "--" if split_type == "both" else "-"
            lbl_train = f"Train ({label})" if split_type == "both" else label
            ax1.plot(ep, h["train_loss"], c=c1, ls=ls_train, lw=2, label=lbl_train)
            ax2.plot(ep, h["train_acc"],  c=c1, ls=ls_train, lw=2, label=lbl_train)
            
        if split_type in ["both", "val"]:
            lbl_val = f"Val ({label})" if split_type == "both" else label
            ax1.plot(ep, h["val_loss"],   c=c2, lw=2, label=lbl_val)
            ax2.plot(ep, h["val_acc"],    c=c2, lw=2, label=lbl_val)

    title_suffix = ""
    if split_type == "train": title_suffix = " (Train)"
    elif split_type == "val": title_suffix = " (Validación)"

    ax1.set(title=f"Loss{title_suffix}", xlabel="Epoch", ylabel="CrossEntropy")
    ax2.set(title=f"Accuracy{title_suffix}", xlabel="Epoch", ylabel="Accuracy")
    for ax in (ax1, ax2):
        ax.grid(True, ls=":", alpha=0.6)
        ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> {save_path.name}")


# ─── Evaluacion completa de un modelo ────────────────────────────────────

def evaluate_model(model, loader, class_names, tag, results_dir, device, cmap="Blues"):
    """
    Ejecuta los 5 pasos de evaluacion para un modelo y genera todos
    los archivos de resultados con el sufijo 'tag' en el nombre.
    Devuelve un diccionario con las metricas resumen.
    """
    print(f"\n{'='*60}")
    print(f"  EVALUACION: {tag}")
    print(f"{'='*60}")

    # ── PASO 1: Predicciones + softmax ───────────────────────────────
    targets, preds, probs = predict_with_probs(model, loader, device)
    acc = (targets == preds).mean() * 100
    print(f"\n  [1/5] Accuracy: {acc:.2f}%")

    # ── PASO 2: Matriz de confusion (heatmap) ────────────────────────
    cm = confusion_matrix(targets, preds)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_pct, annot=cm, fmt="d", cmap=cmap,
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 10})
    plt.title(f"Matriz de Confusion - {tag}", fontsize=16, fontweight="bold")
    plt.ylabel("Clase Real")
    plt.xlabel("Clase Predicha")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(results_dir / f"confusion_matrix_{tag}.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [2/5] Matriz de confusion -> confusion_matrix_{tag}.png")

    # ── PASO 3: Classification report (CSV + barras) ─────────────────
    report = classification_report(
        targets, preds, target_names=class_names, output_dict=True, zero_division=0,
    )
    # Quedarnos solo con las filas de las clases (no macro/weighted/accuracy)
    df = pd.DataFrame({k: report[k] for k in class_names}).T
    df = df[["precision", "recall", "f1-score", "support"]]
    df.to_csv(results_dir / f"classification_report_{tag}.csv", float_format="%.4f")

    # Grafica de barras
    x = np.arange(len(class_names))
    w = 0.25
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (col, color) in enumerate(zip(
        ["precision", "recall", "f1-score"],
        ["#3b82f6", "#10b981", "#f59e0b"],
    )):
        ax.bar(x + i * w, df[col].values, w, label=col.capitalize(), color=color)
    ax.set_xticks(x + w)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Metricas por clase - {tag}", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", ls=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(results_dir / f"classification_bars_{tag}.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [3/5] Report CSV + barras -> classification_report_{tag}.csv / classification_bars_{tag}.png")

    # ── PASO 4: Detector binario (peligro vs fondo) ──────────────────
    #   score = 1 - P(background)   -> alto = mas probable que sea peligro
    #   label = 1 si NO es background, 0 si es background
    scores = 1.0 - probs[:, BG_ID]
    labels_bin = (targets != BG_ID).astype(int)

    # Curva ROC
    fpr, tpr, _ = roc_curve(labels_bin, scores)
    roc_auc = auc(fpr, tpr)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.plot(fpr, tpr, "#3b82f6", lw=2, label=f"AUC = {roc_auc:.4f}")
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax1.set(title="Curva ROC", xlabel="FPR", ylabel="TPR")
    ax1.legend(fontsize=12)
    ax1.grid(True, ls=":", alpha=0.5)

    # Curva Precision-Recall
    prec, rec, _ = precision_recall_curve(labels_bin, scores)
    ap = average_precision_score(labels_bin, scores)

    ax2.plot(rec, prec, "#10b981", lw=2, label=f"AP = {ap:.4f}")
    ax2.set(title="Curva Precision-Recall", xlabel="Recall", ylabel="Precision")
    ax2.legend(fontsize=12)
    ax2.grid(True, ls=":", alpha=0.5)

    fig.suptitle(f"Detector binario - {tag}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(results_dir / f"detector_curves_{tag}.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [4/5] ROC (AUC={roc_auc:.4f}) + PR (AP={ap:.4f}) -> detector_curves_{tag}.png")

    # ── PASO 5: Barrido de umbral ────────────────────────────────────
    thetas = np.linspace(0.10, 0.95, 200)
    fpr_arr, fnr_arr = [], []
    for th in thetas:
        pos = (scores >= th)
        fp = (pos & (labels_bin == 0)).sum()
        tn = (~pos & (labels_bin == 0)).sum()
        fn = (~pos & (labels_bin == 1)).sum()
        tp = (pos & (labels_bin == 1)).sum()
        fpr_arr.append(fp / max(fp + tn, 1))
        fnr_arr.append(fn / max(fn + tp, 1))
    fpr_arr, fnr_arr = np.array(fpr_arr), np.array(fnr_arr)

    # Theta optimo: donde FPR y FNR se cruzan
    idx = np.argmin(np.abs(fpr_arr - fnr_arr))
    theta_opt = float(thetas[idx])

    plt.figure(figsize=(9, 6))
    plt.plot(thetas, fpr_arr, "#ef4444", lw=2, label="FPR")
    plt.plot(thetas, fnr_arr, "#3b82f6", lw=2, label="FNR")
    plt.axvline(theta_opt, color="#f59e0b", lw=2, ls="--", label=f"θ = {theta_opt:.3f}")
    plt.scatter([theta_opt], [fpr_arr[idx]], color="#f59e0b", s=100, zorder=5)
    plt.title(f"Analisis de umbral - {tag}", fontsize=14, fontweight="bold")
    plt.xlabel("Umbral θ")
    plt.ylabel("Tasa de error")
    plt.legend(fontsize=11)
    plt.grid(True, ls=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(results_dir / f"threshold_analysis_{tag}.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [5/5] Umbral optimo = {theta_opt:.3f} -> threshold_analysis_{tag}.png")

    return {
        "accuracy": round(acc, 4),
        "roc_auc": round(roc_auc, 4),
        "average_precision": round(ap, 4),
        "optimal_threshold": round(theta_opt, 4),
    }


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  EVALUACION COMPLETA")
    print("=" * 60)

    results_dir = BASE_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    models_dir = BASE_DIR / "models"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}\n")

    # ── Curvas de aprendizaje ────────────────────────────────────────
    print("Curvas de aprendizaje:")
    plot_learning_curves(
        [models_dir / "audio_cnn_history.json",
         models_dir / "audio_cnn_noise_history.json"],
        labels=["Base", "Ruido"],
        colors=[("#3b82f6", "#1d4ed8"), ("#f59e0b", "#b45309")],
        save_path=results_dir / "curvas_aprendizaje_cnn_train.png",
        split_type="train"
    )
    plot_learning_curves(
        [models_dir / "audio_cnn_history.json",
         models_dir / "audio_cnn_noise_history.json"],
        labels=["Base", "Ruido"],
        colors=[("#3b82f6", "#1d4ed8"), ("#f59e0b", "#b45309")],
        save_path=results_dir / "curvas_aprendizaje_cnn_val.png",
        split_type="val"
    )
    plot_learning_curves(
        [models_dir / "transfer_head_history.json"],
        labels=["Transfer"],
        colors=[("#10b981", "#047857")],
        save_path=results_dir / "curvas_aprendizaje_transfer_train.png",
        split_type="train"
    )
    plot_learning_curves(
        [models_dir / "transfer_head_history.json"],
        labels=["Transfer"],
        colors=[("#10b981", "#047857")],
        save_path=results_dir / "curvas_aprendizaje_transfer_val.png",
        split_type="val"
    )

    # ── A) CNN propia (modelo con ruido) ─────────────────────────────
    df = pd.read_csv(BASE_DIR / "dataset_index_features.csv")
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    class_names = (test_df.sort_values("label_id")[["label_id", "label_name"]]
                   .drop_duplicates()["label_name"].tolist())

    model_cnn = AudioCNN(num_classes=len(class_names))
    model_cnn.load_state_dict(torch.load(
        models_dir / "audio_cnn_noise_best.pt", map_location=device, weights_only=True))
    model_cnn.to(device)

    loader_cnn = DataLoader(
        AudioFeatureDataset(test_df, augment_noise=False),
        batch_size=32, shuffle=False, num_workers=0)

    res_cnn = evaluate_model(model_cnn, loader_cnn, class_names,
                             "cnn", results_dir, device, cmap="Blues")

    # ── B) Transfer Learning ─────────────────────────────────────────
    df_e = pd.read_csv(BASE_DIR / "dataset_index_emb.csv")
    df_e = df_e[df_e["embedding_path"].notna() & (df_e["embedding_path"] != "")]
    test_e = df_e[df_e["split"] == "test"].reset_index(drop=True)
    class_names_t = (test_e.sort_values("label_id")[["label_id", "label_name"]]
                     .drop_duplicates()["label_name"].tolist())

    model_tf = TransferHead(num_classes=len(class_names_t))
    model_tf.load_state_dict(torch.load(
        models_dir / "transfer_head_best.pt", map_location=device, weights_only=True))
    model_tf.to(device)

    loader_tf = DataLoader(
        EmbeddingDataset(test_e),
        batch_size=32, shuffle=False, num_workers=0)

    res_tf = evaluate_model(model_tf, loader_tf, class_names_t,
                            "transfer", results_dir, device, cmap="Greens")

    # ── Guardar threshold.json ───────────────────────────────────────
    threshold = {
        "bg_class_id": BG_ID,
        "bg_class_name": "background",
        "cnn": res_cnn,
        "transfer": res_tf,
        "selected_model": "transfer",
        "selected_threshold": res_tf["optimal_threshold"],
    }
    with open(models_dir / "threshold.json", "w") as f:
        json.dump(threshold, f, indent=2)
    print(f"\n>>> threshold.json guardado en models/")

    # ── Resumen ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  {'':25} {'CNN':>10} {'Transfer':>10}")
    print(f"  {'-'*45}")
    print(f"  {'Accuracy (%)':25} {res_cnn['accuracy']:>9.2f}% {res_tf['accuracy']:>9.2f}%")
    print(f"  {'ROC AUC':25} {res_cnn['roc_auc']:>10.4f} {res_tf['roc_auc']:>10.4f}")
    print(f"  {'Avg Precision':25} {res_cnn['average_precision']:>10.4f} {res_tf['average_precision']:>10.4f}")
    print(f"  {'Umbral optimo':25} {res_cnn['optimal_threshold']:>10.4f} {res_tf['optimal_threshold']:>10.4f}")
    print(f"{'='*60}")
    print(f"  Umbral para la app: {res_tf['optimal_threshold']:.4f} (transfer)")


if __name__ == "__main__":
    main()
