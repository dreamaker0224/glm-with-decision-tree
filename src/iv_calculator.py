"""
IV 計算模組 (Information Value Calculator)
計算 Weight of Evidence (WoE) 與 Information Value (IV)
用於衡量特徵對目標變數的預測能力

修正 Bug 1:
- 在 OHE 後對每個 dummy variable 計算 WoE
- 將屬於同一原始特徵的所有 variables 的 IV 加總
- 每個原始特徵只有一個 IV 值
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def extract_original_feature_name(variable_name: str) -> str:
    """
    從 OHE 後的 variable 名稱提取原始 feature 名稱

    例如：
    - AnnualIncome_Bin_1 → AnnualIncome
    - ServiceType_E → ServiceType
    - Gender_M → Gender

    參數:
        variable_name: OHE 後的變數名稱

    回傳:
        原始特徵名稱
    """
    # 處理 Binned 連續變數 (例如 AnnualIncome_Bin_1)
    if '_Bin_' in variable_name:
        return variable_name.split('_Bin_')[0]

    # 處理類別變數 (例如 ServiceType_E, Gender_M)
    # 取最後一個 _ 之前的部分
    parts = variable_name.split('_')
    if len(parts) >= 2:
        # 保留除了最後一個部分之外的所有內容
        return '_'.join(parts[:-1])

    # 如果沒有 _，就是原始變數名稱
    return variable_name


def calculate_woe_iv_for_binary_variable(df: pd.DataFrame, variable: str, target: str) -> dict:
    """
    修正：對單一 binary variable (0/1) 計算 WoE 和 IV
    只計算 variable=1 時的 WoE 和 IV contribution

    公式：
    p(x=1|Y=Yes) = P(variable=1 | target=1)
    p(x=1|Y=No) = P(variable=1 | target=0)
    WoE = ln(p(x=1|Y=Yes) / p(x=1|Y=No))
    IV = (p(x=1|Y=Yes) - p(x=1|Y=No)) × WoE

    參數:
        df: DataFrame
        variable: 二元變數欄位名稱 (值為 0 或 1)
        target: 目標變數欄位名稱

    回傳:
        包含 WoE 和 IV 資訊的字典
    """

    # 計算各類別的總數
    total_yes = (df[target] == 1).sum()  # Complete (1)
    total_no = (df[target] == 0).sum()   # Quit (0)

    # 避免除以零
    if total_yes == 0 or total_no == 0:
        return {
            'variable': variable,
            'n_yes_when_var_1': 0,
            'n_no_when_var_1': 0,
            'p_var_1_given_yes': 0,
            'p_var_1_given_no': 0,
            'woe': 0,
            'iv_contribution': 0
        }

    # 計算當 variable=1 時，Yes 和 No 的數量
    n_yes_when_var_1 = ((df[variable] == 1) & (df[target] == 1)).sum()
    n_no_when_var_1 = ((df[variable] == 1) & (df[target] == 0)).sum()

    # 計算條件機率
    # p(x=1|Y=Yes) = P(variable=1 | target=1)
    p_var_1_given_yes = n_yes_when_var_1 / total_yes if total_yes > 0 else 0

    # p(x=1|Y=No) = P(variable=1 | target=0)
    p_var_1_given_no = n_no_when_var_1 / total_no if total_no > 0 else 0

    # 避免 log(0)
    if p_var_1_given_yes == 0:
        p_var_1_given_yes = 0.0001
    if p_var_1_given_no == 0:
        p_var_1_given_no = 0.0001

    # 計算 WoE
    woe = np.log(p_var_1_given_yes / p_var_1_given_no)

    # 計算 IV contribution
    iv_contribution = (p_var_1_given_yes - p_var_1_given_no) * woe

    return {
        'variable': variable,
        'n_yes_when_var_1': n_yes_when_var_1,
        'n_no_when_var_1': n_no_when_var_1,
        'p_var_1_given_yes': p_var_1_given_yes,
        'p_var_1_given_no': p_var_1_given_no,
        'woe': woe,
        'iv_contribution': iv_contribution
    }


def calculate_iv_for_group(df: pd.DataFrame, target_col: str, group_name: str) -> tuple:
    """
    修正：計算群組中所有特徵的 IV 值

    1. 對每個 OHE 後的 variable 計算 WoE 和 IV contribution
    2. 將屬於同一原始 feature 的所有 variables 的 IV 加總
    3. 每個原始 feature 只有一個 IV 值

    參數:
        df: DataFrame (OHE 後的資料，全為 0/1 二元變數)
        target_col: 目標變數欄位名稱
        group_name: 群組名稱

    回傳:
        (woe_iv_details_df, feature_iv_summary_df)
    """

    print(f"\n--- {group_name} IV 計算 (修正版) ---")

    variables = [col for col in df.columns if col != target_col]

    print(f"[1/3] 對 {len(variables)} 個 OHE variables 計算 WoE 和 IV")

    # 步驟 1: 對每個 variable 計算 WoE 和 IV contribution
    all_woe_iv = []

    for variable in variables:
        woe_iv_data = calculate_woe_iv_for_binary_variable(df, variable, target_col)

        # 提取原始特徵名稱
        original_feature = extract_original_feature_name(variable)
        woe_iv_data['original_feature'] = original_feature
        woe_iv_data['group'] = group_name

        all_woe_iv.append(woe_iv_data)

    # 建立 DataFrame
    woe_iv_df = pd.DataFrame(all_woe_iv)

    print(f"\n[2/3] 將同一原始特徵的 IV contributions 加總")

    # 步驟 2: 按原始特徵分組，加總 IV
    feature_iv_summary = woe_iv_df.groupby('original_feature').agg({
        'iv_contribution': 'sum'  # 修正：將同一特徵的所有 IV contributions 加總
    }).reset_index()

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

    print(f"\n[3/3] 計算完成，前 10 高 IV 值特徵:")
    for idx, row in feature_iv_summary.head(10).iterrows():
        print(f"      {idx+1:2d}. {row['Feature']:40s} | IV: {row['IV']:.4f} | {row['Interpretation']}")

    return woe_iv_df, feature_iv_summary


def calculate_iv_for_groups(high_input_path: str,
                            low_input_path: str,
                            output_path: str) -> tuple:
    """
    計算 High Group 和 Low Group 的 IV 值

    參數:
        high_input_path: High Group OHE 後資料路徑
        low_input_path: Low Group OHE 後資料路徑
        output_path: IV 值輸出路徑

    回傳:
        (woe_iv_details_df, feature_iv_summary_df)
    """

    print("=" * 80)
    print("步驟 5: 特徵價值計算 (Information Value Calculation)")
    print("修正: 對 OHE variables 計算 WoE，按原始 feature 加總 IV")
    print("=" * 80)

    target_col = 'ServiceStatus'

    # 1. 讀取 High Group 資料
    print(f"\n[1/4] 讀取 High Group OHE 資料: {high_input_path}")
    df_high = pd.read_excel(high_input_path)
    print(f"   - 資料形狀: {df_high.shape}")

    # 2. 讀取 Low Group 資料
    print(f"\n[2/4] 讀取 Low Group OHE 資料: {low_input_path}")
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
        # 工作表 1: 詳細的 WoE 和 IV (每個 variable)
        woe_iv_all.to_excel(writer, sheet_name='WoE_IV_Details', index=False)

        # 工作表 2: IV 摘要 (每個原始 feature)
        iv_summary_all.to_excel(writer, sheet_name='IV_Summary', index=False)

        # 工作表 3: High Group Top Features
        iv_summary_high.head(20).to_excel(writer, sheet_name='High_Group_Top20', index=False)

        # 工作表 4: Low Group Top Features
        iv_summary_low.head(20).to_excel(writer, sheet_name='Low_Group_Top20', index=False)

    print(f"   - 已儲存 IV 結果: {output_path}")
    print(f"   - 包含 4 個工作表:")
    print(f"      1. WoE_IV_Details: 每個 OHE variable 的 WoE 和 IV contribution")
    print(f"      2. IV_Summary: 每個原始 feature 的總 IV (已加總)")
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
