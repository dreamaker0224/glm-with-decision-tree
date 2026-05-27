"""
模型評估模組 (Model Evaluation Module)
計算 ROC、AUC、F1、F2 score，找出最佳 threshold，生成 confusion matrix
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, f1_score, fbeta_score,
    confusion_matrix, precision_recall_curve
)
from sklearn.linear_model import LogisticRegression
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # 使用非互動式後端
import warnings
warnings.filterwarnings('ignore')

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def train_and_evaluate_model(X_train: pd.DataFrame,
                             y_train: pd.Series,
                             group_name: str) -> tuple:
    """
    訓練邏輯回歸模型並返回預測機率

    參數:
        X_train: 訓練特徵
        y_train: 訓練標籤
        group_name: 群組名稱

    回傳:
        (model, y_pred_proba)
    """

    print(f"\n--- {group_name} 模型訓練與評估 ---")

    # 訓練模型
    model = LogisticRegression(
        penalty='l2',
        C=1.0,
        solver='lbfgs',
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train, y_train)

    # 預測機率
    y_pred_proba = model.predict_proba(X_train)[:, 1]

    return model, y_pred_proba


def find_best_threshold_f1(y_true: np.ndarray, y_pred_proba: np.ndarray) -> tuple:
    """
    找出使 F1 score 最大的 threshold

    參數:
        y_true: 真實標籤
        y_pred_proba: 預測機率

    回傳:
        (best_threshold, best_f1)
    """

    thresholds = np.arange(0.1, 0.9, 0.01)
    f1_scores = []

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        f1_scores.append(f1)

    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    return best_threshold, best_f1


def find_best_threshold_f2(y_true: np.ndarray, y_pred_proba: np.ndarray) -> tuple:
    """
    找出使 F2 score 最大的 threshold
    F2 score 更重視 Recall（召回率）

    參數:
        y_true: 真實標籤
        y_pred_proba: 預測機率

    回傳:
        (best_threshold, best_f2)
    """

    thresholds = np.arange(0.1, 0.9, 0.01)
    f2_scores = []

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
        f2_scores.append(f2)

    best_idx = np.argmax(f2_scores)
    best_threshold = thresholds[best_idx]
    best_f2 = f2_scores[best_idx]

    return best_threshold, best_f2


def calculate_metrics_at_threshold(y_true: np.ndarray,
                                   y_pred_proba: np.ndarray,
                                   threshold: float) -> dict:
    """
    在給定 threshold 下計算所有評估指標

    參數:
        y_true: 真實標籤
        y_pred_proba: 預測機率
        threshold: 分類閾值

    回傳:
        包含各種指標的字典
    """

    y_pred = (y_pred_proba >= threshold).astype(int)

    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # 計算指標
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)

    return {
        'threshold': threshold,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'f2_score': f2
    }


def evaluate_group(input_path: str, target_col: str, group_name: str) -> dict:
    """
    評估單一群組的模型表現

    參數:
        input_path: OHE 後的資料路徑
        target_col: 目標變數欄位名稱
        group_name: 群組名稱

    回傳:
        評估結果字典
    """

    print(f"\n{'='*80}")
    print(f"{group_name} 模型評估")
    print(f"{'='*80}")

    # 讀取資料
    df = pd.read_excel(input_path)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    print(f"\n資料形狀: {df.shape}")
    print(f"Complete (1) 比例: {y.mean():.2%}")

    # 訓練模型並取得預測機率
    model, y_pred_proba = train_and_evaluate_model(X, y, group_name)

    # 計算 ROC 和 AUC
    fpr, tpr, roc_thresholds = roc_curve(y, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    print(f"\n[1/4] ROC 與 AUC")
    print(f"   - AUC: {roc_auc:.4f}")

    # 找出最佳 F1 threshold
    best_f1_threshold, best_f1 = find_best_threshold_f1(y.values, y_pred_proba)

    print(f"\n[2/4] 最佳 F1 Score Threshold")
    print(f"   - Threshold: {best_f1_threshold:.3f}")
    print(f"   - F1 Score: {best_f1:.4f}")

    # 找出最佳 F2 threshold
    best_f2_threshold, best_f2 = find_best_threshold_f2(y.values, y_pred_proba)

    print(f"\n[3/4] 最佳 F2 Score Threshold")
    print(f"   - Threshold: {best_f2_threshold:.3f}")
    print(f"   - F2 Score: {best_f2:.4f}")

    # 使用預設 threshold 0.5
    default_threshold = 0.5

    # 計算各 threshold 下的指標
    print(f"\n[4/4] 計算各 Threshold 下的指標")

    metrics_default = calculate_metrics_at_threshold(y.values, y_pred_proba, default_threshold)
    metrics_f1 = calculate_metrics_at_threshold(y.values, y_pred_proba, best_f1_threshold)
    metrics_f2 = calculate_metrics_at_threshold(y.values, y_pred_proba, best_f2_threshold)

    print(f"   - 預設 (0.5): F1={metrics_default['f1_score']:.4f}, F2={metrics_default['f2_score']:.4f}")
    print(f"   - 最佳 F1: F1={metrics_f1['f1_score']:.4f}, F2={metrics_f1['f2_score']:.4f}")
    print(f"   - 最佳 F2: F1={metrics_f2['f1_score']:.4f}, F2={metrics_f2['f2_score']:.4f}")

    # 整理結果
    result = {
        'group_name': group_name,
        'model': model,
        'y_true': y.values,
        'y_pred_proba': y_pred_proba,
        'roc_curve': {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': roc_thresholds,
            'auc': roc_auc
        },
        'best_f1_threshold': best_f1_threshold,
        'best_f1_score': best_f1,
        'best_f2_threshold': best_f2_threshold,
        'best_f2_score': best_f2,
        'metrics': {
            'default': metrics_default,
            'best_f1': metrics_f1,
            'best_f2': metrics_f2
        }
    }

    return result


def plot_roc_curves(high_result: dict, low_result: dict, output_path: str):
    """
    繪製 High Group 和 Low Group 的 ROC curves

    參數:
        high_result: High Group 評估結果
        low_result: Low Group 評估結果
        output_path: 輸出圖片路徑
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # High Group ROC
    ax1.plot(high_result['roc_curve']['fpr'],
             high_result['roc_curve']['tpr'],
             color='darkorange', lw=2,
             label=f'ROC curve (AUC = {high_result["roc_curve"]["auc"]:.4f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title('High Group ROC Curve', fontsize=14, fontweight='bold')
    ax1.legend(loc="lower right")
    ax1.grid(alpha=0.3)

    # Low Group ROC
    ax2.plot(low_result['roc_curve']['fpr'],
             low_result['roc_curve']['tpr'],
             color='darkgreen', lw=2,
             label=f'ROC curve (AUC = {low_result["roc_curve"]["auc"]:.4f})')
    ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate', fontsize=12)
    ax2.set_ylabel('True Positive Rate', fontsize=12)
    ax2.set_title('Low Group ROC Curve', fontsize=14, fontweight='bold')
    ax2.legend(loc="lower right")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"   - 已儲存 ROC curves: {output_path}")


def plot_confusion_matrices(result: dict, output_path: str):
    """
    繪製單一群組在不同 threshold 下的 confusion matrices

    參數:
        result: 評估結果
        output_path: 輸出圖片路徑
    """

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    thresholds = ['default', 'best_f1', 'best_f2']
    titles = [
        f"Default (0.5)",
        f"Best F1 ({result['best_f1_threshold']:.3f})",
        f"Best F2 ({result['best_f2_threshold']:.3f})"
    ]

    for idx, (thresh_key, title) in enumerate(zip(thresholds, titles)):
        metrics = result['metrics'][thresh_key]

        cm = np.array([[metrics['tn'], metrics['fp']],
                       [metrics['fn'], metrics['tp']]])

        im = axes[idx].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        axes[idx].set_title(f"{result['group_name']}\n{title}",
                           fontsize=12, fontweight='bold')

        # 添加數值
        for i in range(2):
            for j in range(2):
                text = axes[idx].text(j, i, f"{cm[i, j]:,}",
                                     ha="center", va="center",
                                     color="white" if cm[i, j] > cm.max() / 2 else "black",
                                     fontsize=14, fontweight='bold')

        axes[idx].set_ylabel('Actual', fontsize=11)
        axes[idx].set_xlabel('Predicted', fontsize=11)
        axes[idx].set_xticks([0, 1])
        axes[idx].set_yticks([0, 1])
        axes[idx].set_xticklabels(['Quit (0)', 'Complete (1)'])
        axes[idx].set_yticklabels(['Quit (0)', 'Complete (1)'])

        # 添加指標文字
        info_text = (f"Accuracy: {metrics['accuracy']:.3f}\n"
                    f"Precision: {metrics['precision']:.3f}\n"
                    f"Recall: {metrics['recall']:.3f}\n"
                    f"F1: {metrics['f1_score']:.3f}\n"
                    f"F2: {metrics['f2_score']:.3f}")
        axes[idx].text(1.5, 0.5, info_text, transform=axes[idx].transData,
                      fontsize=9, verticalalignment='center',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"   - 已儲存 Confusion Matrices: {output_path}")


def evaluate_models(high_input_path: str,
                   low_input_path: str,
                   output_dir: str) -> tuple:
    """
    評估 High Group 和 Low Group 的模型

    參數:
        high_input_path: High Group OHE 資料路徑
        low_input_path: Low Group OHE 資料路徑
        output_dir: 輸出目錄

    回傳:
        (high_result, low_result)
    """

    print("=" * 80)
    print("步驟 4.5: 模型評估 (Model Evaluation)")
    print("計算 ROC, AUC, F1, F2，找出最佳 thresholds，生成 Confusion Matrices")
    print("=" * 80)

    target_col = 'ServiceStatus'

    # 確保輸出目錄存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 評估 High Group
    print("\n[1/2] 評估 High Group")
    high_result = evaluate_group(high_input_path, target_col, "High Group")

    # 評估 Low Group
    print("\n[2/2] 評估 Low Group")
    low_result = evaluate_group(low_input_path, target_col, "Low Group")

    # 繪製圖表
    print(f"\n[繪圖] 生成評估圖表")

    roc_output = f"{output_dir}/roc_curves.png"
    plot_roc_curves(high_result, low_result, roc_output)

    high_cm_output = f"{output_dir}/high_group_confusion_matrices.png"
    plot_confusion_matrices(high_result, high_cm_output)

    low_cm_output = f"{output_dir}/low_group_confusion_matrices.png"
    plot_confusion_matrices(low_result, low_cm_output)

    # 儲存評估結果為 Excel
    evaluation_data = []
    for group_result in [high_result, low_result]:
        for thresh_key in ['default', 'best_f1', 'best_f2']:
            metrics = group_result['metrics'][thresh_key]
            evaluation_data.append({
                'Group': group_result['group_name'],
                'Threshold_Type': thresh_key,
                'Threshold': metrics['threshold'],
                'AUC': group_result['roc_curve']['auc'],
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1_Score': metrics['f1_score'],
                'F2_Score': metrics['f2_score'],
                'TP': metrics['tp'],
                'TN': metrics['tn'],
                'FP': metrics['fp'],
                'FN': metrics['fn']
            })

    evaluation_df = pd.DataFrame(evaluation_data)
    eval_excel_path = f"{output_dir}/../../output/4.5_model_evaluation.xlsx"
    evaluation_df.to_excel(eval_excel_path, index=False, engine='openpyxl')
    print(f"   - 已儲存評估指標: {eval_excel_path}")

    print("\n✓ 模型評估完成！")
    print("=" * 80)
    print()

    return high_result, low_result


if __name__ == "__main__":
    # 測試用
    high_input = "output/3_high_group_transformed.xlsx"
    low_input = "output/3_low_group_transformed.xlsx"
    output_dir = "reports/figures"

    high_result, low_result = evaluate_models(high_input, low_input, output_dir)
