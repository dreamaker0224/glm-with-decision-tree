"""
單一邏輯回歸模組 (Single Logistic Regression Module)
不經過決策樹分群，直接對所有資料訓練單一邏輯回歸模型
用於與兩階段方法（決策樹分群 + 分群邏輯回歸）進行比較
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    fbeta_score, roc_auc_score, roc_curve, confusion_matrix
)
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def train_single_logistic_regression(input_path: str,
                                     target_col: str,
                                     output_path: str) -> dict:
    """
    對所有資料訓練單一邏輯回歸模型

    參數:
        input_path: 轉換後資料路徑（OHE 後，全為 0/1 二元變數）
        target_col: 目標變數欄位名稱
        output_path: 係數輸出路徑

    回傳:
        包含模型、預測結果和評估指標的字典
    """

    print("=" * 80)
    print("單一邏輯回歸方法 (Single Logistic Regression Approach)")
    print("不經過決策樹分群，直接對所有資料建模")
    print("=" * 80)

    # 讀取資料
    print(f"\n[1/4] 讀取資料: {input_path}")
    df = pd.read_excel(input_path)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    print(f"   - 資料形狀: {df.shape}")
    print(f"   - 特徵數量: {X.shape[1]}")
    print(f"   - 樣本數量: {X.shape[0]}")
    print(f"   - Complete (1) 比例: {y.mean():.2%}")

    # 訓練模型
    print(f"\n[2/4] 訓練單一邏輯回歸模型")

    model = LogisticRegression(
        penalty='l2',
        C=1.0,
        solver='lbfgs',
        max_iter=1000,
        random_state=42
    )

    model.fit(X, y)
    print(f"   - 模型訓練完成")

    # 預測
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]

    # 評估指標
    print(f"\n[3/4] 計算評估指標")

    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    f2 = fbeta_score(y, y_pred, beta=2, zero_division=0)

    try:
        auc = roc_auc_score(y, y_pred_proba)
    except:
        auc = 0.0

    print(f"   - AUC: {auc:.4f}")
    print(f"   - Accuracy: {accuracy:.4f}")
    print(f"   - Precision: {precision:.4f}")
    print(f"   - Recall: {recall:.4f}")
    print(f"   - F1 Score: {f1:.4f}")
    print(f"   - F2 Score: {f2:.4f}")

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y, y_pred_proba)

    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

    print(f"\n   Confusion Matrix (at threshold 0.5):")
    print(f"   - True Negatives: {tn:,}")
    print(f"   - False Positives: {fp:,}")
    print(f"   - False Negatives: {fn:,}")
    print(f"   - True Positives: {tp:,}")

    # 儲存係數
    print(f"\n[4/4] 儲存模型係數")

    coefficients = model.coef_[0]
    intercept = model.intercept_[0]

    coef_data = []
    for feature, coef in zip(X.columns, coefficients):
        coef_data.append({
            'Approach': 'Single LR',
            'Feature': feature,
            'Coefficient': coef,
            'Abs_Coefficient': abs(coef),
            'Direction': 'Positive' if coef > 0 else 'Negative'
        })

    coef_data.append({
        'Approach': 'Single LR',
        'Feature': 'Intercept',
        'Coefficient': intercept,
        'Abs_Coefficient': abs(intercept),
        'Direction': 'Positive' if intercept > 0 else 'Negative'
    })

    coefficients_df = pd.DataFrame(coef_data)
    coefficients_df = coefficients_df.sort_values('Abs_Coefficient', ascending=False)

    # 確保輸出目錄存在
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    coefficients_df.to_excel(output_path, index=False, engine='openpyxl')
    print(f"   - 已儲存係數: {output_path}")

    print(f"\n   前 10 大影響特徵:")
    for idx, row in coefficients_df.head(10).iterrows():
        if row['Feature'] != 'Intercept':
            print(f"      {row['Feature']:40s}: {row['Coefficient']:7.4f} ({row['Direction']})")

    print("\n✓ 單一邏輯回歸訓練完成！")
    print("=" * 80)
    print()

    # 返回結果
    result = {
        'model': model,
        'X': X,
        'y_true': y.values,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'metrics': {
            'auc': auc,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'f2_score': f2,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn
        },
        'roc_curve': {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'auc': auc
        },
        'coefficients_df': coefficients_df
    }

    return result


def create_single_approach_transformed_data(preprocessed_path: str,
                                           output_path: str,
                                           binning_rules_path: str) -> pd.DataFrame:
    """
    為單一邏輯回歸方法準備轉換後的資料
    使用與兩階段方法相同的特徵轉換邏輯

    參數:
        preprocessed_path: 預處理後資料路徑
        output_path: 轉換後資料輸出路徑
        binning_rules_path: 分段規則輸出路徑

    回傳:
        轉換後的 DataFrame
    """

    print("=" * 80)
    print("單一方法特徵轉換 (Single Approach Feature Transformation)")
    print("=" * 80)

    # 這裡可以重用 feature_transform.py 的邏輯
    # 為了簡化，我們直接讀取已經由兩階段方法生成的轉換資料
    # 將 High Group 和 Low Group 的轉換資料合併

    print("\n[1/2] 合併 High Group 和 Low Group 的轉換資料")

    try:
        df_high = pd.read_excel('output/3_high_group_transformed.xlsx')
        df_low = pd.read_excel('output/3_low_group_transformed.xlsx')

        print(f"   - High Group: {len(df_high)} 筆, {df_high.shape[1]} 欄")
        print(f"   - Low Group: {len(df_low)} 筆, {df_low.shape[1]} 欄")

        # 找出所有欄位的聯集
        all_columns = list(set(df_high.columns) | set(df_low.columns))

        # 確保 ServiceStatus 在最後
        if 'ServiceStatus' in all_columns:
            all_columns.remove('ServiceStatus')
            all_columns.sort()
            all_columns.append('ServiceStatus')
        else:
            all_columns.sort()

        # 對齊欄位，缺少的欄位填 0（OHE 二元變數，缺少表示該類別不存在）
        for col in all_columns:
            if col not in df_high.columns:
                df_high[col] = 0
            if col not in df_low.columns:
                df_low[col] = 0

        # 重新排序欄位
        df_high = df_high[all_columns]
        df_low = df_low[all_columns]

        # 合併
        df_combined = pd.concat([df_high, df_low], ignore_index=True)

        print(f"   - 合併後: {len(df_combined)} 筆, {df_combined.shape[1]} 欄")
        print(f"   - 檢查缺失值: {df_combined.isna().sum().sum()} 個")

        # 儲存
        print(f"\n[2/2] 儲存合併後的資料")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df_combined.to_excel(output_path, index=False, engine='openpyxl')
        print(f"   - 已儲存: {output_path}")

        print("\n✓ 單一方法特徵轉換完成！")
        print("=" * 80)
        print()

        return df_combined

    except FileNotFoundError:
        print("   ⚠️  需要先執行兩階段方法以生成轉換後的資料")
        return None


if __name__ == "__main__":
    # 測試用
    # 先準備資料
    combined_path = "output/3_single_approach_transformed.xlsx"
    create_single_approach_transformed_data(
        "output/1_preprocessed_data.xlsx",
        combined_path,
        "output/3_continuous_bins_rules.xlsx"
    )

    # 訓練單一模型
    output_coef = "output/4_single_lr_coefficients.xlsx"
    result = train_single_logistic_regression(
        combined_path,
        'ServiceStatus',
        output_coef
    )

    print(f"AUC: {result['metrics']['auc']:.4f}")
    print(f"F1: {result['metrics']['f1_score']:.4f}")
