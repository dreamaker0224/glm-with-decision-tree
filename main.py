"""
主程式 (Main Pipeline)
串接所有模組，執行完整的學生課程完成率預測分析流程
"""

import sys
from pathlib import Path
from datetime import datetime

# 加入 src 目錄到 Python 路徑
sys.path.append(str(Path(__file__).parent / 'src'))

from src.preprocess import load_and_preprocess_data
from src.decision_tree import train_decision_tree_and_split
from src.feature_transform import transform_groups
from src.logistic_regression import train_group_logistic_regressions
from src.model_evaluation import evaluate_models
from src.iv_calculator import calculate_iv_for_groups
from src.report_generator import generate_business_insight_report


def main():
    """
    執行完整的分析流程
    """

    print("=" * 80)
    print("學生課程完成率預測分析專案")
    print("Student Course Completion Rate Prediction and Analysis Project")
    print("=" * 80)
    print(f"開始時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()

    # 定義檔案路徑
    input_data = "data/dataset.xlsx"

    preprocessed_data = "output/1_preprocessed_data.xlsx"

    high_group_raw = "output/2_high_group_raw.xlsx"
    low_group_raw = "output/2_low_group_raw.xlsx"

    high_group_transformed = "output/3_high_group_transformed.xlsx"
    low_group_transformed = "output/3_low_group_transformed.xlsx"
    binning_rules = "output/3_continuous_bins_rules.xlsx"

    lr_coefficients = "output/4_lr_coefficients.xlsx"

    evaluation_results = "output/4.5_model_evaluation.xlsx"
    figures_dir = "reports/figures"

    iv_results = "output/5_iv_results.xlsx"

    report_output = "reports/business_insight_report.md"

    try:
        # Step 1: 資料預處理
        print("▶ 步驟 1/6: 資料預處理")
        df_preprocessed = load_and_preprocess_data(input_data, preprocessed_data)

        # Step 2: 決策樹分流
        print("\n▶ 步驟 2/6: 決策樹分流")
        df_high, df_low, threshold, dt_model = train_decision_tree_and_split(
            preprocessed_data, high_group_raw, low_group_raw
        )

        # Step 3: 特徵轉換
        print("\n▶ 步驟 3/6: 特徵轉換")
        df_high_transformed, df_low_transformed, binning_rules_df = transform_groups(
            high_group_raw, low_group_raw,
            high_group_transformed, low_group_transformed, binning_rules
        )

        # Step 4: 邏輯回歸訓練
        print("\n▶ 步驟 4/7: 分群邏輯回歸")
        model_high, model_low, coefficients_df = train_group_logistic_regressions(
            high_group_transformed, low_group_transformed, lr_coefficients
        )

        # Step 5: 模型評估
        print("\n▶ 步驟 5/7: 模型評估 (ROC, AUC, F1, F2, Confusion Matrix)")
        high_eval_result, low_eval_result = evaluate_models(
            high_group_transformed, low_group_transformed, figures_dir
        )

        # Step 6: IV 計算
        print("\n▶ 步驟 6/7: 特徵價值計算 (IV)")
        woe_iv_df, iv_summary_df = calculate_iv_for_groups(
            high_group_transformed, low_group_transformed, iv_results
        )

        # Step 7: 生成商業報告
        print("\n▶ 步驟 7/7: 生成商業洞察報告")
        generate_business_insight_report(
            high_group_raw, low_group_raw, binning_rules, lr_coefficients,
            iv_results, evaluation_results, report_output, threshold,
            high_eval_result, low_eval_result
        )

        # 完成摘要
        print("=" * 80)
        print("✓ 所有步驟完成！")
        print("=" * 80)
        print("\n[輸出檔案清單]")
        print(f"  1. 預處理資料:        {preprocessed_data}")
        print(f"  2. High Group 原始:   {high_group_raw}")
        print(f"  3. Low Group 原始:    {low_group_raw}")
        print(f"  4. High Group 轉換:   {high_group_transformed}")
        print(f"  5. Low Group 轉換:    {low_group_transformed}")
        print(f"  6. 分段規則:          {binning_rules}")
        print(f"  7. 邏輯回歸係數:      {lr_coefficients}")
        print(f"  8. 模型評估結果:      {evaluation_results}")
        print(f"  9. IV 計算結果:       {iv_results}")
        print(f" 10. 商業洞察報告:      {report_output}")
        print(f" 11. 評估圖表:          {figures_dir}/")
        print(f"     - ROC curves")
        print(f"     - Confusion matrices")

        print(f"\n結束時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        print("\n[下一步行動]")
        print("  1. 查閱商業洞察報告: reports/business_insight_report.md")
        print("  2. 檢視各階段輸出的 Excel 檔案")
        print("  3. 根據報告建議制定學生輔導策略")
        print()

        return True

    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ 執行過程發生錯誤！")
        print("=" * 80)
        print(f"錯誤訊息: {str(e)}")
        print()
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
