"""
方法比較模組 (Approach Comparison Module)
比較兩種方法的表現：
1. 兩階段方法 (Two-Stage): 決策樹分群 + 分群邏輯回歸
2. 單一模型方法 (Single Model): 直接邏輯回歸
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def compare_approaches(high_eval_result: dict,
                      low_eval_result: dict,
                      single_eval_result: dict,
                      output_dir: str) -> pd.DataFrame:
    """
    比較兩種方法的表現

    參數:
        high_eval_result: High Group 評估結果
        low_eval_result: Low Group 評估結果
        single_eval_result: 單一模型評估結果
        output_dir: 輸出目錄

    回傳:
        比較結果 DataFrame
    """

    print("=" * 80)
    print("方法比較 (Approach Comparison)")
    print("=" * 80)

    # 準備比較數據
    comparison_data = []

    # 兩階段方法 - High Group
    for thresh_key in ['default', 'best_f1', 'best_f2']:
        metrics = high_eval_result['metrics'][thresh_key]
        comparison_data.append({
            'Approach': 'Two-Stage',
            'Group': 'High Group',
            'Threshold_Type': thresh_key,
            'Threshold': metrics['threshold'],
            'AUC': high_eval_result['roc_curve']['auc'],
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1_Score': metrics['f1_score'],
            'F2_Score': metrics['f2_score'],
            'Sample_Size': len(high_eval_result['y_true']),
            'Positive_Rate': high_eval_result['y_true'].mean()
        })

    # 兩階段方法 - Low Group
    for thresh_key in ['default', 'best_f1', 'best_f2']:
        metrics = low_eval_result['metrics'][thresh_key]
        comparison_data.append({
            'Approach': 'Two-Stage',
            'Group': 'Low Group',
            'Threshold_Type': thresh_key,
            'Threshold': metrics['threshold'],
            'AUC': low_eval_result['roc_curve']['auc'],
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1_Score': metrics['f1_score'],
            'F2_Score': metrics['f2_score'],
            'Sample_Size': len(low_eval_result['y_true']),
            'Positive_Rate': low_eval_result['y_true'].mean()
        })

    # 單一模型方法
    for thresh_key in ['default', 'best_f1', 'best_f2']:
        if thresh_key in single_eval_result['metrics']:
            metrics = single_eval_result['metrics'][thresh_key]
            comparison_data.append({
                'Approach': 'Single Model',
                'Group': 'All Data',
                'Threshold_Type': thresh_key,
                'Threshold': metrics['threshold'],
                'AUC': single_eval_result['roc_curve']['auc'],
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1_Score': metrics['f1_score'],
                'F2_Score': metrics['f2_score'],
                'Sample_Size': len(single_eval_result['y_true']),
                'Positive_Rate': single_eval_result['y_true'].mean()
            })

    comparison_df = pd.DataFrame(comparison_data)

    # 儲存比較結果
    output_path = f"{output_dir}/../6_approach_comparison.xlsx"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_excel(output_path, index=False, engine='openpyxl')

    print(f"\n[1/3] 已儲存比較結果: {output_path}")

    # 列印摘要比較
    print(f"\n[2/3] 方法比較摘要 (Best F1 Threshold):")
    print("=" * 80)

    # Two-Stage Method
    two_stage_high_f1 = comparison_df[
        (comparison_df['Approach'] == 'Two-Stage') &
        (comparison_df['Group'] == 'High Group') &
        (comparison_df['Threshold_Type'] == 'best_f1')
    ].iloc[0]

    two_stage_low_f1 = comparison_df[
        (comparison_df['Approach'] == 'Two-Stage') &
        (comparison_df['Group'] == 'Low Group') &
        (comparison_df['Threshold_Type'] == 'best_f1')
    ].iloc[0]

    # Single Model
    single_f1 = comparison_df[
        (comparison_df['Approach'] == 'Single Model') &
        (comparison_df['Threshold_Type'] == 'best_f1')
    ].iloc[0]

    print("\n兩階段方法 (Two-Stage Approach):")
    print(f"  High Group (n={two_stage_high_f1['Sample_Size']:.0f}, Complete率={two_stage_high_f1['Positive_Rate']:.2%}):")
    print(f"    AUC: {two_stage_high_f1['AUC']:.4f} | F1: {two_stage_high_f1['F1_Score']:.4f} | Recall: {two_stage_high_f1['Recall']:.4f}")
    print(f"  Low Group (n={two_stage_low_f1['Sample_Size']:.0f}, Complete率={two_stage_low_f1['Positive_Rate']:.2%}):")
    print(f"    AUC: {two_stage_low_f1['AUC']:.4f} | F1: {two_stage_low_f1['F1_Score']:.4f} | Recall: {two_stage_low_f1['Recall']:.4f}")

    print(f"\n單一模型方法 (Single Model Approach):")
    print(f"  All Data (n={single_f1['Sample_Size']:.0f}, Complete率={single_f1['Positive_Rate']:.2%}):")
    print(f"    AUC: {single_f1['AUC']:.4f} | F1: {single_f1['F1_Score']:.4f} | Recall: {single_f1['Recall']:.4f}")

    # 繪製比較圖
    print(f"\n[3/3] 繪製比較圖表")
    plot_comparison_charts(comparison_df, output_dir)

    print("\n✓ 方法比較完成！")
    print("=" * 80)
    print()

    return comparison_df


def plot_comparison_charts(comparison_df: pd.DataFrame, output_dir: str):
    """
    繪製方法比較圖表

    參數:
        comparison_df: 比較結果 DataFrame
        output_dir: 輸出目錄
    """

    # 篩選 Best F1 資料
    best_f1_df = comparison_df[comparison_df['Threshold_Type'] == 'best_f1'].copy()

    # 圖表 1: AUC 比較
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    metrics = ['AUC', 'F1_Score', 'Recall']
    titles = ['AUC Comparison', 'F1 Score Comparison', 'Recall Comparison']
    ylabels = ['AUC', 'F1 Score', 'Recall']

    for idx, (metric, title, ylabel) in enumerate(zip(metrics, titles, ylabels)):
        ax = axes[idx]

        # 兩階段方法
        two_stage_data = best_f1_df[best_f1_df['Approach'] == 'Two-Stage']
        x_two_stage = np.arange(len(two_stage_data))
        y_two_stage = two_stage_data[metric].values
        labels_two_stage = two_stage_data['Group'].values

        # 單一模型
        single_data = best_f1_df[best_f1_df['Approach'] == 'Single Model']
        y_single = single_data[metric].values[0] if len(single_data) > 0 else 0

        # 繪製
        bars1 = ax.bar(x_two_stage, y_two_stage, width=0.35, label='Two-Stage', color='steelblue')
        bars2 = ax.axhline(y=y_single, color='darkorange', linestyle='--', linewidth=2, label='Single Model')

        ax.set_xlabel('Group', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xticks(x_two_stage)
        ax.set_xticklabels(labels_two_stage, rotation=15)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.0])

        # 添加數值標籤
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)

        # 單一模型數值標籤
        ax.text(len(x_two_stage)-0.5, y_single + 0.02,
               f'{y_single:.3f}',
               ha='center', va='bottom', fontsize=9, color='darkorange', fontweight='bold')

    plt.tight_layout()
    output_path = f"{output_dir}/approach_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"   - 已儲存比較圖表: {output_path}")


def evaluate_single_model_with_thresholds(single_result: dict) -> dict:
    """
    對單一模型進行閾值搜尋，找出最佳 F1 和 F2 threshold

    參數:
        single_result: train_single_logistic_regression 的返回結果

    回傳:
        更新後的結果（包含不同閾值下的指標）
    """

    from sklearn.metrics import f1_score, fbeta_score, confusion_matrix

    y_true = single_result['y_true']
    y_pred_proba = single_result['y_pred_proba']

    # 找出最佳 F1 threshold
    thresholds = np.arange(0.1, 0.9, 0.01)
    f1_scores = []
    f2_scores = []

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
        f1_scores.append(f1)
        f2_scores.append(f2)

    best_f1_idx = np.argmax(f1_scores)
    best_f1_threshold = thresholds[best_f1_idx]
    best_f1 = f1_scores[best_f1_idx]

    best_f2_idx = np.argmax(f2_scores)
    best_f2_threshold = thresholds[best_f2_idx]
    best_f2 = f2_scores[best_f2_idx]

    # 計算各閾值下的完整指標
    def calc_metrics(threshold):
        y_pred = (y_pred_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        return {
            'threshold': threshold,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'accuracy': (tp + tn) / (tp + tn + fp + fn),
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'f2_score': fbeta_score(y_true, y_pred, beta=2, zero_division=0)
        }

    single_result['best_f1_threshold'] = best_f1_threshold
    single_result['best_f1_score'] = best_f1
    single_result['best_f2_threshold'] = best_f2_threshold
    single_result['best_f2_score'] = best_f2

    single_result['metrics']['default'] = calc_metrics(0.5)
    single_result['metrics']['best_f1'] = calc_metrics(best_f1_threshold)
    single_result['metrics']['best_f2'] = calc_metrics(best_f2_threshold)

    return single_result


if __name__ == "__main__":
    # 測試用
    print("This module is designed to be imported and used with evaluation results.")
