"""
決策樹分流模組 (Decision Tree Segmentation Module)
訓練決策樹模型，計算預測機率，並以平均機率作為閾值切分高低風險群
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from pathlib import Path


def identify_feature_types(df: pd.DataFrame, target_col: str) -> tuple:
    """
    識別連續變數與類別變數

    參數:
        df: 預處理後的 DataFrame
        target_col: 目標變數欄位名稱

    回傳:
        (continuous_features, categorical_features)
    """
    continuous_features = []
    categorical_features = []

    for col in df.columns:
        if col == target_col:
            continue

        if df[col].dtype in ['float64', 'int64']:
            # 數值型欄位，但需要判斷是否為類別變數
            unique_count = df[col].nunique()
            # 如果唯一值數量小於 10，視為類別變數
            if unique_count < 10:
                categorical_features.append(col)
            else:
                continuous_features.append(col)
        else:
            # 字串型欄位為類別變數
            categorical_features.append(col)

    return continuous_features, categorical_features


def train_decision_tree_and_split(input_path: str,
                                   high_output_path: str,
                                   low_output_path: str) -> tuple:
    """
    訓練決策樹模型，計算預測機率，並以平均機率切分群組

    參數:
        input_path: 預處理後的資料路徑
        high_output_path: 高完成機率群輸出路徑
        low_output_path: 低完成機率群輸出路徑

    回傳:
        (df_high, df_low, threshold, model)
    """

    print("=" * 80)
    print("步驟 2: 決策樹分流 (Decision Tree Segmentation)")
    print("=" * 80)

    # 1. 讀取預處理後的資料
    print(f"\n[1/6] 讀取預處理後的資料: {input_path}")
    df = pd.read_excel(input_path)
    print(f"   - 資料形狀: {df.shape}")

    # 2. 識別特徵類型
    print("\n[2/6] 識別特徵類型")
    target_col = 'ServiceStatus'
    continuous_features, categorical_features = identify_feature_types(df, target_col)

    print(f"   - 連續變數 ({len(continuous_features)}): {continuous_features}")
    print(f"   - 類別變數 ({len(categorical_features)}): {categorical_features}")

    # 3. 準備特徵矩陣
    print("\n[3/6] 準備特徵矩陣")

    # 對類別變數進行 One-Hot Encoding
    df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=False)
    print(f"   - One-Hot Encoding 後的欄位數量: {len(df_encoded.columns)}")

    # 分離特徵與目標變數
    X = df_encoded.drop(columns=[target_col])
    y = df_encoded[target_col]

    print(f"   - 特徵矩陣形狀: {X.shape}")
    print(f"   - 目標變數形狀: {y.shape}")
    print(f"   - Complete (1) 比例: {y.mean():.2%}")

    # 4. 訓練決策樹模型
    print("\n[4/6] 訓練決策樹模型")

    # 設定隨機種子以確保可重現性
    random_state = 42

    # 訓練決策樹（使用適當的參數避免過擬合）
    model = DecisionTreeClassifier(
        max_depth=10,           # 限制深度避免過擬合
        min_samples_split=100,  # 最小分裂樣本數
        min_samples_leaf=50,    # 最小葉節點樣本數
        random_state=random_state
    )

    model.fit(X, y)
    print(f"   - 模型訓練完成")
    print(f"   - 決策樹深度: {model.get_depth()}")
    print(f"   - 葉節點數量: {model.get_n_leaves()}")

    # 計算訓練集準確率
    train_score = model.score(X, y)
    print(f"   - 訓練集準確率: {train_score:.4f}")

    # 5. 計算預測機率並切分群組
    print("\n[5/6] 計算預測機率並切分群組")

    # 預測 Complete (類別 1) 的機率
    predict_proba = model.predict_proba(X)[:, 1]

    # 計算平均機率作為閾值
    threshold = predict_proba.mean()
    print(f"   - Complete 的真實平均機率 (實際比例): {y.mean():.4f}")
    print(f"   - Complete 的預測平均機率 (閾值): {threshold:.4f}")

    # 將預測機率加入原始資料
    df_with_proba = df.copy()
    df_with_proba['PredictedProba_Complete'] = predict_proba

    # 根據閾值切分群組
    df_high = df_with_proba[df_with_proba['PredictedProba_Complete'] >= threshold].copy()
    df_low = df_with_proba[df_with_proba['PredictedProba_Complete'] < threshold].copy()

    print(f"\n   - 切分結果:")
    print(f"     * High Group (預測機率 >= {threshold:.4f}): {len(df_high)} 筆 ({len(df_high)/len(df)*100:.1f}%)")
    print(f"       - 實際 Complete 比例: {df_high['ServiceStatus'].mean():.2%}")
    print(f"       - 平均預測機率: {df_high['PredictedProba_Complete'].mean():.4f}")

    print(f"     * Low Group (預測機率 < {threshold:.4f}): {len(df_low)} 筆 ({len(df_low)/len(df)*100:.1f}%)")
    print(f"       - 實際 Complete 比例: {df_low['ServiceStatus'].mean():.2%}")
    print(f"       - 平均預測機率: {df_low['PredictedProba_Complete'].mean():.4f}")

    # 6. 輸出切分後的資料
    print(f"\n[6/6] 輸出切分後的資料")

    # 確保輸出目錄存在
    Path(high_output_path).parent.mkdir(parents=True, exist_ok=True)

    # 儲存 High Group
    df_high.to_excel(high_output_path, index=False, engine='openpyxl')
    print(f"   - 已儲存 High Group: {high_output_path}")

    # 儲存 Low Group
    df_low.to_excel(low_output_path, index=False, engine='openpyxl')
    print(f"   - 已儲存 Low Group: {low_output_path}")

    print("\n✓ 決策樹分流完成！")
    print("=" * 80)
    print()

    return df_high, df_low, threshold, model


if __name__ == "__main__":
    # 測試用
    input_file = "output/1_preprocessed_data.xlsx"
    high_output_file = "output/2_high_group_raw.xlsx"
    low_output_file = "output/2_low_group_raw.xlsx"

    df_high, df_low, threshold, model = train_decision_tree_and_split(
        input_file, high_output_file, low_output_file
    )

    print(f"High Group: {df_high.shape}")
    print(f"Low Group: {df_low.shape}")
    print(f"Threshold: {threshold:.4f}")
