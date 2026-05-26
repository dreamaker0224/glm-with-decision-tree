"""
資料預處理模組 (Data Preprocessing Module)
負責讀取原始資料、轉換目標變數、移除無用欄位、處理缺失值
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_and_preprocess_data(input_path: str, output_path: str) -> pd.DataFrame:
    """
    讀取原始資料並進行預處理

    參數:
        input_path: 原始資料檔案路徑 (dataset.xlsx)
        output_path: 預處理後資料的輸出路徑

    回傳:
        預處理後的 DataFrame
    """

    print("=" * 80)
    print("步驟 1: 資料預處理 (Data Preprocessing)")
    print("=" * 80)

    # 1. 讀取原始資料
    print(f"\n[1/5] 讀取原始資料: {input_path}")
    df = pd.read_excel(input_path)
    print(f"   - 原始資料形狀: {df.shape}")
    print(f"   - 欄位數量: {len(df.columns)}")

    # 2. 轉換目標變數 ServiceStatus
    print("\n[2/5] 轉換目標變數 ServiceStatus")
    print(f"   - 轉換前的值分布:")
    print(df['ServiceStatus'].value_counts())

    # Complete 編碼為 1 (正類別), Quit 編碼為 0 (負類別)
    df['ServiceStatus'] = df['ServiceStatus'].map({'Complete': 1, 'Quit': 0})

    print(f"\n   - 轉換後的值分布:")
    print(df['ServiceStatus'].value_counts())
    print(f"   - Complete 比例: {df['ServiceStatus'].mean():.2%}")

    # 3. 移除無用欄位
    print("\n[3/5] 移除無用欄位")
    # 修正：加上 Year 和 Month，避免時間變數洩漏 (Data Leakage)
    drop_columns = ['CustID', 'ServiceStartDate', 'WeeksWithService', 'Year', 'Month']
    print(f"   - 移除欄位: {drop_columns}")
    print(f"   - 理由:")
    print(f"     * CustID: 學生 ID，無預測價值")
    print(f"     * ServiceStartDate: 避免時間資訊外洩")
    print(f"     * WeeksWithService: 避免時間資訊外洩導致模型作弊")
    print(f"     * Year, Month: 時間變數會導致模型死記歷史，失去泛化能力 (修正 Bug 2)")

    df = df.drop(columns=drop_columns)
    print(f"   - 移除後的欄位數量: {len(df.columns)}")

    # 4. 檢查缺失值
    print("\n[4/5] 檢查缺失值")
    missing_counts = df.isnull().sum()
    missing_cols = missing_counts[missing_counts > 0]

    if len(missing_cols) > 0:
        print("   - 發現缺失值:")
        for col, count in missing_cols.items():
            pct = count / len(df) * 100
            print(f"     * {col}: {count} ({pct:.2f}%)")

        # 處理缺失值策略
        print("\n   - 處理策略:")
        for col in missing_cols.index:
            if df[col].dtype in ['float64', 'int64']:
                # 數值型：使用中位數填補
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"     * {col}: 使用中位數 ({median_val}) 填補")
            else:
                # 類別型：使用眾數填補
                mode_val = df[col].mode()[0]
                df[col].fillna(mode_val, inplace=True)
                print(f"     * {col}: 使用眾數 ({mode_val}) 填補")
    else:
        print("   - 無缺失值")

    # 5. 輸出預處理後的資料
    print(f"\n[5/5] 輸出預處理後的資料: {output_path}")

    # 確保輸出目錄存在
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # 儲存為 Excel
    df.to_excel(output_path, index=False, engine='openpyxl')
    print(f"   - 已儲存: {output_path}")
    print(f"   - 最終資料形狀: {df.shape}")

    # 輸出欄位摘要
    print("\n[摘要] 預處理後的欄位列表:")
    for idx, col in enumerate(df.columns, 1):
        dtype = df[col].dtype
        unique_count = df[col].nunique()
        print(f"   {idx:2d}. {col:25s} | 型態: {str(dtype):15s} | 唯一值數量: {unique_count}")

    print("\n✓ 資料預處理完成！")
    print("=" * 80)
    print()

    return df


if __name__ == "__main__":
    # 測試用
    input_file = "data/dataset.xlsx"
    output_file = "output/1_preprocessed_data.xlsx"

    df = load_and_preprocess_data(input_file, output_file)
    print(f"預處理後資料: {df.shape}")
