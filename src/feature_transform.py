"""
特徵轉換模組 (Feature Transformation Module)
將所有特徵轉換為 0/1 二元變數，以符合羅吉斯迴歸的要求
- 連續變數: 使用決策樹監督式分段 (Supervised Binning)
- 名目變數: 使用 n-1 的 One-Hot Encoding
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def identify_feature_types(df: pd.DataFrame, target_col: str) -> tuple:
    """
    識別連續變數與類別變數

    參數:
        df: DataFrame
        target_col: 目標變數欄位名稱

    回傳:
        (continuous_features, categorical_features)
    """
    continuous_features = []
    categorical_features = []

    # 排除目標變數和預測機率欄位
    exclude_cols = [target_col, 'PredictedProba_Complete']

    for col in df.columns:
        if col in exclude_cols:
            continue

        if df[col].dtype in ['float64', 'int64']:
            # 數值型欄位，判斷是否為類別變數
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


def bin_continuous_variable_with_dt(df: pd.DataFrame, feature: str, target: str, max_bins: int = 5) -> tuple:
    """
    使用決策樹對單一連續變數進行監督式分段

    參數:
        df: DataFrame
        feature: 連續變數欄位名稱
        target: 目標變數欄位名稱
        max_bins: 最大分段數量

    回傳:
        (binned_series, split_points, bin_labels)
    """

    # 移除缺失值
    df_clean = df[[feature, target]].dropna()

    if len(df_clean) == 0:
        raise ValueError(f"特徵 {feature} 沒有有效數據")

    X = df_clean[[feature]]
    y = df_clean[target]

    # 訓練簡單的決策樹來找切點
    dt = DecisionTreeClassifier(
        max_leaf_nodes=max_bins,  # 限制最大分段數
        min_samples_leaf=50,       # 每個葉節點至少 50 個樣本
        random_state=42
    )

    dt.fit(X, y)

    # 提取決策樹的分裂閾值
    tree = dt.tree_
    split_points = []

    def extract_thresholds(node_id=0):
        """遞迴提取所有分裂閾值"""
        if tree.feature[node_id] != -2:  # 不是葉節點
            threshold = tree.threshold[node_id]
            split_points.append(threshold)
            extract_thresholds(tree.children_left[node_id])
            extract_thresholds(tree.children_right[node_id])

    extract_thresholds()

    # 排序切點
    split_points = sorted(set(split_points))

    # 如果沒有找到切點，使用四分位數
    if len(split_points) == 0:
        split_points = [df[feature].quantile(0.5)]

    # 根據切點創建分段
    bins = [-np.inf] + split_points + [np.inf]
    bin_labels = [f"Bin_{i}" for i in range(len(bins) - 1)]

    # 對原始資料進行分段
    binned = pd.cut(df[feature], bins=bins, labels=bin_labels, include_lowest=True)

    return binned, split_points, bin_labels


def transform_features_to_binary(df: pd.DataFrame,
                                  target_col: str,
                                  group_name: str) -> tuple:
    """
    將所有特徵轉換為 0/1 二元變數

    參數:
        df: 原始 DataFrame
        target_col: 目標變數欄位名稱
        group_name: 群組名稱 (用於記錄)

    回傳:
        (df_transformed, binning_rules)
    """

    print(f"\n--- {group_name} 特徵轉換 ---")

    # 1. 識別特徵類型
    continuous_features, categorical_features = identify_feature_types(df, target_col)

    print(f"[1/3] 識別特徵類型")
    print(f"   - 連續變數 ({len(continuous_features)}): {continuous_features}")
    print(f"   - 類別變數 ({len(categorical_features)}): {categorical_features}")

    # 2. 處理連續變數 (使用決策樹分段)
    print(f"\n[2/3] 處理連續變數 (決策樹監督式分段)")

    binning_rules = []
    # 如果有目標變數，保留它；否則創建空 DataFrame
    if target_col is not None:
        df_transformed = df[[target_col]].copy()
    else:
        df_transformed = pd.DataFrame(index=df.index)

    for feature in continuous_features:
        print(f"   - 處理 {feature}...")

        try:
            # 使用決策樹分段
            binned, split_points, bin_labels = bin_continuous_variable_with_dt(
                df, feature, target_col, max_bins=5
            )

            # 記錄分段規則
            for i, (bin_label, threshold) in enumerate(zip(bin_labels, [-np.inf] + split_points)):
                if i < len(split_points):
                    upper_threshold = split_points[i]
                    rule = f"{threshold:.2f} <= {feature} < {upper_threshold:.2f}"
                else:
                    rule = f"{feature} >= {split_points[-1]:.2f}"

                binning_rules.append({
                    'Group': group_name,
                    'Feature': feature,
                    'FeatureType': 'Continuous',
                    'Bin': bin_label,
                    'SplitPoints': ', '.join([f"{sp:.2f}" for sp in split_points]),
                    'Rule': rule
                })

            # 將分段結果轉換為 One-Hot Encoding (n-1)
            binned_df = pd.get_dummies(binned, prefix=feature, drop_first=True, dtype=int)
            df_transformed = pd.concat([df_transformed, binned_df], axis=1)

            print(f"      完成！切點: {[f'{sp:.2f}' for sp in split_points]}, 產生 {len(binned_df.columns)} 個二元變數")

        except Exception as e:
            print(f"      警告: 處理 {feature} 時發生錯誤: {e}")
            # 如果失敗，使用簡單的中位數切分
            median_val = df[feature].median()
            df_transformed[f"{feature}_AboveMedian"] = (df[feature] >= median_val).astype(int)
            print(f"      使用中位數切分: {median_val:.2f}")

            binning_rules.append({
                'Group': group_name,
                'Feature': feature,
                'FeatureType': 'Continuous',
                'Bin': 'AboveMedian',
                'SplitPoints': f"{median_val:.2f}",
                'Rule': f"{feature} >= {median_val:.2f}"
            })

    # 3. 處理類別變數 (One-Hot Encoding with n-1)
    print(f"\n[3/3] 處理類別變數 (One-Hot Encoding, n-1 方式)")

    for feature in categorical_features:
        print(f"   - 處理 {feature}...")

        # 取得類別列表
        categories = df[feature].unique()
        n_categories = len(categories)

        # One-Hot Encoding with drop_first=True (n-1 方式)
        encoded_df = pd.get_dummies(df[feature], prefix=feature, drop_first=True, dtype=int)
        df_transformed = pd.concat([df_transformed, encoded_df], axis=1)

        # 記錄編碼規則
        dropped_category = sorted(categories)[0]  # 預設 drop_first 會移除第一個（排序後）
        for col in encoded_df.columns:
            category = col.replace(f"{feature}_", "")
            binning_rules.append({
                'Group': group_name,
                'Feature': feature,
                'FeatureType': 'Categorical',
                'Bin': category,
                'SplitPoints': 'N/A',
                'Rule': f"{feature} == {category}"
            })

        print(f"      完成！類別數: {n_categories}, 產生 {len(encoded_df.columns)} 個二元變數 (dropped: {dropped_category})")

    print(f"\n   轉換完成：")
    print(f"   - 原始特徵數: {len(continuous_features) + len(categorical_features)}")
    print(f"   - 轉換後二元變數數: {len(df_transformed.columns) - 1}")  # 減去目標變數

    return df_transformed, binning_rules


def transform_groups(high_input_path: str,
                     low_input_path: str,
                     high_output_path: str,
                     low_output_path: str,
                     rules_output_path: str) -> tuple:
    """
    對 High Group 和 Low Group 分別進行特徵轉換

    參數:
        high_input_path: High Group 原始資料路徑
        low_input_path: Low Group 原始資料路徑
        high_output_path: High Group 轉換後資料輸出路徑
        low_output_path: Low Group 轉換後資料輸出路徑
        rules_output_path: 分段規則輸出路徑

    回傳:
        (df_high_transformed, df_low_transformed, binning_rules_df)
    """

    print("=" * 80)
    print("步驟 3: 特徵轉換 (Feature Transformation)")
    print("=" * 80)

    target_col = 'ServiceStatus'

    # 讀取 High Group 資料
    print(f"\n[1/4] 讀取 High Group 資料: {high_input_path}")
    df_high = pd.read_excel(high_input_path)
    print(f"   - 資料形狀: {df_high.shape}")

    # 讀取 Low Group 資料
    print(f"\n[2/4] 讀取 Low Group 資料: {low_input_path}")
    df_low = pd.read_excel(low_input_path)
    print(f"   - 資料形狀: {df_low.shape}")

    # 轉換 High Group
    print(f"\n[3/4] 轉換 High Group 特徵")
    df_high_transformed, high_rules = transform_features_to_binary(
        df_high, target_col, "High Group"
    )

    # 轉換 Low Group
    print(f"\n[4/4] 轉換 Low Group 特徵")
    df_low_transformed, low_rules = transform_features_to_binary(
        df_low, target_col, "Low Group"
    )

    # 合併分段規則
    all_rules = high_rules + low_rules
    binning_rules_df = pd.DataFrame(all_rules)

    # 輸出結果
    print(f"\n[輸出] 儲存轉換結果")

    # 確保輸出目錄存在
    Path(high_output_path).parent.mkdir(parents=True, exist_ok=True)

    # 儲存 High Group 轉換後資料
    df_high_transformed.to_excel(high_output_path, index=False, engine='openpyxl')
    print(f"   - 已儲存 High Group 轉換後資料: {high_output_path}")

    # 儲存 Low Group 轉換後資料
    df_low_transformed.to_excel(low_output_path, index=False, engine='openpyxl')
    print(f"   - 已儲存 Low Group 轉換後資料: {low_output_path}")

    # 儲存分段規則
    binning_rules_df.to_excel(rules_output_path, index=False, engine='openpyxl')
    print(f"   - 已儲存分段規則: {rules_output_path}")

    print("\n✓ 特徵轉換完成！所有特徵已轉換為 0/1 二元變數")
    print("=" * 80)
    print()

    return df_high_transformed, df_low_transformed, binning_rules_df


if __name__ == "__main__":
    # 測試用
    high_input = "output/2_high_group_raw.xlsx"
    low_input = "output/2_low_group_raw.xlsx"
    high_output = "output/3_high_group_transformed.xlsx"
    low_output = "output/3_low_group_transformed.xlsx"
    rules_output = "output/3_continuous_bins_rules.xlsx"

    df_high, df_low, rules = transform_groups(
        high_input, low_input, high_output, low_output, rules_output
    )

    print(f"High Group Transformed: {df_high.shape}")
    print(f"Low Group Transformed: {df_low.shape}")
    print(f"Binning Rules: {len(rules)} rules")
