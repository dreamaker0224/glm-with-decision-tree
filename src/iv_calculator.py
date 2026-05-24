"""
IV 計算模組 (Information Value Calculator)
計算 Weight of Evidence (WoE) 與 Information Value (IV)
用於衡量特徵對目標變數的預測能力
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def calculate_woe_iv(df: pd.DataFrame, feature: str, target: str) -> dict:
    """
    計算單一特徵的 WoE 和 IV

    參數:
        df: DataFrame
        feature: 特徵欄位名稱
        target: 目標變數欄位名稱

    回傳:
        包含 WoE 和 IV 資訊的字典
    """

    # 建立交叉表
    crosstab = pd.crosstab(df[feature], df[target], margins=False)

    # 確保有兩個類別 (0 和 1)
    if 0 not in crosstab.columns:
        crosstab[0] = 0
    if 1 not in crosstab.columns:
        crosstab[1] = 0

    # 計算各類別的總數
    total_bad = crosstab[0].sum()   # Quit (0)
    total_good = crosstab[1].sum()  # Complete (1)

    # 避免除以零
    if total_bad == 0 or total_good == 0:
        return {
            'feature': feature,
            'category': None,
            'n_bad': 0,
            'n_good': 0,
            'pct_bad': 0,
            'pct_good': 0,
            'woe': 0,
            'iv': 0
        }

    woe_iv_data = []

    for category in crosstab.index:
        n_bad = crosstab.loc[category, 0]
        n_good = crosstab.loc[category, 1]

        # 計算比例
        pct_bad = n_bad / total_bad if total_bad > 0 else 0
        pct_good = n_good / total_good if total_good > 0 else 0

        # 避免 log(0)
        if pct_bad == 0:
            pct_bad = 0.0001
        if pct_good == 0:
            pct_good = 0.0001

        # 計算 WoE
        woe = np.log(pct_good / pct_bad)

        # 計算 IV
        iv = (pct_good - pct_bad) * woe

        woe_iv_data.append({
            'feature': feature,
            'category': category,
            'n_bad': n_bad,
            'n_good': n_good,
            'pct_bad': pct_bad,
            'pct_good': pct_good,
            'woe': woe,
            'iv': iv
        })

    return woe_iv_data


def calculate_iv_for_group(df: pd.DataFrame, target_col: str, group_name: str) -> pd.DataFrame:
    """
    計算群組中所有特徵的 IV 值

    參數:
        df: DataFrame (轉換後的資料，全為 0/1 二元變數)
        target_col: 目標變數欄位名稱
        group_name: 群組名稱

    回傳:
        包含所有特徵 IV 值的 DataFrame
    """

    print(f"\n--- {group_name} IV 計算 ---")

    features = [col for col in df.columns if col != target_col]

    print(f"[1/2] 計算 {len(features)} 個特徵的 WoE 和 IV")

    all_woe_iv = []

    for feature in features:
        woe_iv_data = calculate_woe_iv(df, feature, target_col)
        all_woe_iv.extend(woe_iv_data)

    # 建立 DataFrame
    woe_iv_df = pd.DataFrame(all_woe_iv)
    woe_iv_df['Group'] = group_name

    # 計算每個特徵的總 IV
    print(f"\n[2/2] 彙總特徵 IV 值")

    feature_iv_summary = woe_iv_df.groupby('feature')['iv'].sum().reset_index()
    feature_iv_summary.columns = ['Feature', 'IV']
    feature_iv_summary = feature_iv_summary.sort_values('IV', ascending=False)

    # IV 值解釋
    def iv_interpretation(iv):
        if iv < 0.02:
            return 'Useless'
        elif iv < 0.1:
            return 'Weak'
        elif iv < 0.3:
            return 'Medium'
        elif iv < 0.5:
            return 'Strong'
        else:
            return 'Very Strong'

    feature_iv_summary['Interpretation'] = feature_iv_summary['IV'].apply(iv_interpretation)
    feature_iv_summary['Group'] = group_name

    print(f"\n   前 10 高 IV 值特徵:")
    for idx, row in feature_iv_summary.head(10).iterrows():
        print(f"      {idx+1:2d}. {row['Feature']:40s} | IV: {row['IV']:.4f} | {row['Interpretation']}")

    return woe_iv_df, feature_iv_summary


def calculate_iv_for_groups(high_input_path: str,
                            low_input_path: str,
                            output_path: str) -> tuple:
    """
    計算 High Group 和 Low Group 的 IV 值

    參數:
        high_input_path: High Group 轉換後資料路徑
        low_input_path: Low Group 轉換後資料路徑
        output_path: IV 值輸出路徑

    回傳:
        (iv_results_df, iv_summary_df)
    """

    print("=" * 80)
    print("步驟 5: 特徵價值計算 (Information Value Calculation)")
    print("=" * 80)

    target_col = 'ServiceStatus'

    # 1. 讀取 High Group 資料
    print(f"\n[1/4] 讀取 High Group 轉換後資料: {high_input_path}")
    df_high = pd.read_excel(high_input_path)
    print(f"   - 資料形狀: {df_high.shape}")

    # 2. 讀取 Low Group 資料
    print(f"\n[2/4] 讀取 Low Group 轉換後資料: {low_input_path}")
    df_low = pd.read_excel(low_input_path)
    print(f"   - 資料形狀: {df_low.shape}")

    # 3. 計算 High Group IV
    print(f"\n[3/4] 計算 High Group IV")
    woe_iv_high, iv_summary_high = calculate_iv_for_group(df_high, target_col, "High Group")

    # 4. 計算 Low Group IV
    print(f"\n[4/4] 計算 Low Group IV")
    woe_iv_low, iv_summary_low = calculate_iv_for_group(df_low, target_col, "Low Group")

    # 合併結果
    woe_iv_all = pd.concat([woe_iv_high, woe_iv_low], ignore_index=True)
    iv_summary_all = pd.concat([iv_summary_high, iv_summary_low], ignore_index=True)

    # 輸出結果
    print(f"\n[輸出] 儲存 IV 計算結果")

    # 確保輸出目錄存在
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # 使用 ExcelWriter 創建多個工作表
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # 工作表 1: 詳細的 WoE 和 IV
        woe_iv_all.to_excel(writer, sheet_name='WoE_IV_Details', index=False)

        # 工作表 2: IV 摘要
        iv_summary_all.to_excel(writer, sheet_name='IV_Summary', index=False)

        # 工作表 3: High Group Top Features
        iv_summary_high.head(20).to_excel(writer, sheet_name='High_Group_Top20', index=False)

        # 工作表 4: Low Group Top Features
        iv_summary_low.head(20).to_excel(writer, sheet_name='Low_Group_Top20', index=False)

    print(f"   - 已儲存 IV 結果: {output_path}")
    print(f"   - 包含 4 個工作表:")
    print(f"      1. WoE_IV_Details: 詳細的 WoE 和 IV 計算")
    print(f"      2. IV_Summary: 特徵 IV 摘要")
    print(f"      3. High_Group_Top20: High Group 前 20 高 IV 特徵")
    print(f"      4. Low_Group_Top20: Low Group 前 20 高 IV 特徵")

    print("\n✓ IV 計算完成！")
    print("=" * 80)
    print()

    return woe_iv_all, iv_summary_all


if __name__ == "__main__":
    # 測試用
    high_input = "output/3_high_group_transformed.xlsx"
    low_input = "output/3_low_group_transformed.xlsx"
    output = "output/5_iv_results.xlsx"

    woe_iv_df, iv_summary_df = calculate_iv_for_groups(
        high_input, low_input, output
    )

    print(f"WoE/IV Details: {len(woe_iv_df)} entries")
    print(f"IV Summary: {len(iv_summary_df)} features")
