"""
邏輯回歸模組 (Logistic Regression Module)
分別對 High Group 和 Low Group 訓練獨立的邏輯回歸模型
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def train_logistic_regression(X: pd.DataFrame, y: pd.Series, group_name: str) -> tuple:
    """
    訓練邏輯回歸模型

    參數:
        X: 特徵矩陣 (全為 0/1 二元變數)
        y: 目標變數
        group_name: 群組名稱

    回傳:
        (model, coefficients_df)
    """

    print(f"\n--- {group_name} 邏輯回歸訓練 ---")

    # 訓練邏輯回歸模型
    print(f"[1/3] 訓練邏輯回歸模型")
    print(f"   - 特徵數量: {X.shape[1]}")
    print(f"   - 樣本數量: {X.shape[0]}")
    print(f"   - Complete (1) 比例: {y.mean():.2%}")

    # 使用 L2 正則化的邏輯回歸
    model = LogisticRegression(
        penalty='l2',
        C=1.0,               # 正則化強度 (C 越小，正則化越強)
        solver='lbfgs',      # 優化算法
        max_iter=1000,       # 最大迭代次數
        random_state=42
    )

    model.fit(X, y)
    print(f"   - 模型訓練完成")

    # 評估模型
    print(f"\n[2/3] 模型評估")
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]

    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y, y_pred_proba)
        print(f"   - AUC: {auc:.4f}")
    except:
        print(f"   - AUC: N/A (可能類別分布問題)")

    print(f"   - 準確率 (Accuracy): {accuracy:.4f}")
    print(f"   - 精確率 (Precision): {precision:.4f}")
    print(f"   - 召回率 (Recall): {recall:.4f}")
    print(f"   - F1-Score: {f1:.4f}")

    # 提取係數
    print(f"\n[3/3] 提取模型係數")

    coefficients = model.coef_[0]
    intercept = model.intercept_[0]

    # 建立係數 DataFrame
    coef_data = []
    for feature, coef in zip(X.columns, coefficients):
        coef_data.append({
            'Group': group_name,
            'Feature': feature,
            'Coefficient': coef,
            'Abs_Coefficient': abs(coef),
            'Direction': 'Positive' if coef > 0 else 'Negative'
        })

    # 加入截距
    coef_data.append({
        'Group': group_name,
        'Feature': 'Intercept',
        'Coefficient': intercept,
        'Abs_Coefficient': abs(intercept),
        'Direction': 'Positive' if intercept > 0 else 'Negative'
    })

    coefficients_df = pd.DataFrame(coef_data)

    # 按絕對值排序
    coefficients_df = coefficients_df.sort_values('Abs_Coefficient', ascending=False)

    print(f"   - 截距 (Intercept): {intercept:.4f}")
    print(f"   - 前 5 大影響特徵:")
    for idx, row in coefficients_df.head(5).iterrows():
        if row['Feature'] != 'Intercept':
            print(f"      * {row['Feature']}: {row['Coefficient']:.4f} ({row['Direction']})")

    return model, coefficients_df


def train_group_logistic_regressions(high_input_path: str,
                                      low_input_path: str,
                                      output_path: str) -> tuple:
    """
    對 High Group 和 Low Group 分別訓練邏輯回歸模型

    參數:
        high_input_path: High Group 轉換後資料路徑
        low_input_path: Low Group 轉換後資料路徑
        output_path: 係數輸出路徑

    回傳:
        (model_high, model_low, coefficients_df)
    """

    print("=" * 80)
    print("步驟 4: 分群邏輯回歸 (Group-wise Logistic Regression)")
    print("=" * 80)

    target_col = 'ServiceStatus'

    # 1. 讀取 High Group 資料
    print(f"\n[1/4] 讀取 High Group 轉換後資料: {high_input_path}")
    df_high = pd.read_excel(high_input_path)
    print(f"   - 資料形狀: {df_high.shape}")

    X_high = df_high.drop(columns=[target_col])
    y_high = df_high[target_col]

    # 2. 讀取 Low Group 資料
    print(f"\n[2/4] 讀取 Low Group 轉換後資料: {low_input_path}")
    df_low = pd.read_excel(low_input_path)
    print(f"   - 資料形狀: {df_low.shape}")

    X_low = df_low.drop(columns=[target_col])
    y_low = df_low[target_col]

    # 3. 訓練 High Group 邏輯回歸
    print(f"\n[3/4] 訓練 High Group 邏輯回歸")
    model_high, coef_high = train_logistic_regression(X_high, y_high, "High Group")

    # 4. 訓練 Low Group 邏輯回歸
    print(f"\n[4/4] 訓練 Low Group 邏輯回歸")
    model_low, coef_low = train_logistic_regression(X_low, y_low, "Low Group")

    # 合併係數
    coefficients_df = pd.concat([coef_high, coef_low], ignore_index=True)

    # 輸出係數
    print(f"\n[輸出] 儲存模型係數")

    # 確保輸出目錄存在
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    coefficients_df.to_excel(output_path, index=False, engine='openpyxl')
    print(f"   - 已儲存係數: {output_path}")

    print("\n✓ 邏輯回歸訓練完成！")
    print("=" * 80)
    print()

    return model_high, model_low, coefficients_df


if __name__ == "__main__":
    # 測試用
    high_input = "output/3_high_group_transformed.xlsx"
    low_input = "output/3_low_group_transformed.xlsx"
    output = "output/4_lr_coefficients.xlsx"

    model_high, model_low, coef_df = train_group_logistic_regressions(
        high_input, low_input, output
    )

    print(f"Coefficients: {len(coef_df)} entries")
