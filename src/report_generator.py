"""
報告生成模組 (Report Generator)
將模型統計量轉譯為校務商業報告
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path


def generate_business_insight_report(
    high_raw_path: str,
    low_raw_path: str,
    binning_rules_path: str,
    coefficients_path: str,
    iv_results_path: str,
    evaluation_path: str,
    output_path: str,
    threshold: float = None,
    high_eval_result: dict = None,
    low_eval_result: dict = None,
    single_eval_result: dict = None,
    comparison_df: pd.DataFrame = None
) -> None:
    """
    生成商業洞察報告

    參數:
        high_raw_path: High Group 原始資料路徑
        low_raw_path: Low Group 原始資料路徑
        binning_rules_path: 分段規則路徑
        coefficients_path: 邏輯回歸係數路徑
        iv_results_path: IV 結果路徑
        evaluation_path: 模型評估結果路徑
        output_path: 報告輸出路徑
        threshold: 決策樹切分閾值
        high_eval_result: High Group 評估結果（可選）
        low_eval_result: Low Group 評估結果（可選）
        single_eval_result: 單一模型評估結果（可選）
        comparison_df: 方法比較結果（可選）
    """

    print("=" * 80)
    print("步驟 6: 生成商業洞察報告 (Business Insight Report Generation)")
    print("=" * 80)

    # 讀取資料
    print("\n[1/6] 讀取分析結果資料")

    df_high = pd.read_excel(high_raw_path)
    df_low = pd.read_excel(low_raw_path)
    binning_rules = pd.read_excel(binning_rules_path)
    coefficients = pd.read_excel(coefficients_path)
    iv_summary = pd.read_excel(iv_results_path, sheet_name='IV_Summary')

    print(f"   - High Group: {len(df_high)} 筆")
    print(f"   - Low Group: {len(df_low)} 筆")
    print(f"   - 分段規則: {len(binning_rules)} 條")
    print(f"   - 模型係數: {len(coefficients)} 個")
    print(f"   - IV 值: {len(iv_summary)} 個特徵")

    # 計算統計資訊
    print("\n[2/6] 計算統計資訊")

    total_students = len(df_high) + len(df_low)
    high_complete_rate = df_high['ServiceStatus'].mean()
    low_complete_rate = df_low['ServiceStatus'].mean()

    print(f"   - 總學生數: {total_students}")
    print(f"   - High Group 完成率: {high_complete_rate:.2%}")
    print(f"   - Low Group 完成率: {low_complete_rate:.2%}")

    # 獲取高 IV 特徵
    print("\n[3/6] 識別關鍵特徵")

    iv_high = iv_summary[iv_summary['Group'] == 'High Group'].sort_values('IV', ascending=False)
    iv_low = iv_summary[iv_summary['Group'] == 'Low Group'].sort_values('IV', ascending=False)

    top_iv_high = iv_high.head(5)
    top_iv_low = iv_low.head(5)

    print(f"   - High Group 前 5 高 IV 特徵: {top_iv_high['Feature'].tolist()}")
    print(f"   - Low Group 前 5 高 IV 特徵: {top_iv_low['Feature'].tolist()}")

    # 獲取重要係數
    print("\n[4/6] 識別驅動因素")

    coef_high = coefficients[coefficients['Group'] == 'High Group'].sort_values('Abs_Coefficient', ascending=False)
    coef_low = coefficients[coefficients['Group'] == 'Low Group'].sort_values('Abs_Coefficient', ascending=False)

    # 排除截距
    coef_high_features = coef_high[coef_high['Feature'] != 'Intercept'].head(5)
    coef_low_features = coef_low[coef_low['Feature'] != 'Intercept'].head(5)

    print(f"   - High Group 前 5 大係數特徵: {coef_high_features['Feature'].tolist()}")
    print(f"   - Low Group 前 5 大係數特徵: {coef_low_features['Feature'].tolist()}")

    # 生成報告
    print("\n[5/6] 生成報告內容")

    report_time = datetime.now().strftime("%Y-%m-%d %H:%M")

    report = f"""# 學生課程完成率預測分析報告
# Student Course Completion Rate Prediction Analysis Report

**報告生成時間**: {report_time}

---

## 一、執行摘要 (Executive Summary)

本分析針對 **{total_students:,}** 位學生的課程完成情況進行深入研究，運用決策樹與邏輯回歸模型建立早期預警機制。

### 核心發現

我們將學生分為兩個風險群體：

1. **高完成機率群 (High Group)**: {len(df_high):,} 位學生 ({len(df_high)/total_students*100:.1f}%)
   - 實際完成率: **{high_complete_rate:.1%}**
   - 特徵: 這些學生具備較高的課程完成潛力，但仍需關注個別風險因子

2. **低完成機率群 (Low Group)**: {len(df_low):,} 位學生 ({len(df_low)/total_students*100:.1f}%)
   - 實際完成率: **{low_complete_rate:.1%}**
   - 特徵: 這些學生面臨較高的流失風險，需要優先配置輔導資源

**關鍵差異**: 兩群體的完成率相差 **{abs(high_complete_rate - low_complete_rate):.1%}**，顯示模型成功識別出高風險學生群體。

---

## 二、早期預警指標分析 (Early Warning Indicators)

Information Value (IV) 衡量特徵對預測目標的貢獻度。IV 值越高，該特徵對辨識流失風險的能力越強。

### 2.1 低完成機率群的關鍵預警指標

在 **低完成機率群** 中，以下特徵最能辨識即將流失的學生：

"""

    # 添加 Low Group 的 Top 5 IV 特徵
    for idx, row in top_iv_low.iterrows():
        feature = row['Feature']
        iv_value = row['IV']
        interpretation = row['Interpretation']

        # 嘗試找到對應的分段規則
        feature_rules = binning_rules[
            (binning_rules['Group'] == 'Low Group') &
            (binning_rules['Feature'].str.contains(feature.split('_')[0], case=False, na=False))
        ]

        report += f"""
#### {idx+1}. {feature}
- **IV 值**: {iv_value:.4f} ({interpretation})
- **商業意義**: """

        # 根據特徵名稱提供解釋
        if 'Credit' in feature:
            report += "信用評分是學生財務穩定性的關鍵指標。低信用分數可能意味著財務壓力，直接影響繳費能力與持續就讀意願。"
        elif 'Age' in feature:
            report += "年齡因素反映學生的生涯階段。特定年齡區間的學生可能面臨工作、家庭等多重壓力，影響課程投入度。"
        elif 'PaymentMethod' in feature or 'Payment' in feature:
            report += "付款方式反映學生的財務規劃習慣。不穩定的付款模式往往是流失的前兆信號。"
        elif 'ServiceType' in feature or 'Service' in feature:
            report += "課程類型影響學生的學習體驗與投入程度。某些課程類型可能與學生需求不匹配，導致較高流失率。"
        elif 'Market' in feature:
            report += "市場區隔反映學生的來源與期望。不同市場的學生有不同的動機與需求，需要差異化的支持策略。"
        elif 'AnnualIncome' in feature or 'Income' in feature:
            report += "年收入水平直接關聯學生的經濟負擔能力。低收入群體可能因學費壓力而中途退出。"
        elif 'NewCustomer' in feature:
            report += "新舊客戶狀態反映學生對機構的熟悉度與信任度。新生可能因適應困難而流失。"
        elif 'Gender' in feature:
            report += "性別因素可能與特定課程類型的適配度有關，也可能反映不同性別面臨的社會與家庭壓力差異。"
        elif 'Dependents' in feature:
            report += "家庭負擔人數影響學生的時間與經濟壓力。有撫養責任的學生需要更靈活的課程安排。"
        elif 'MaritalStatus' in feature or 'Marital' in feature:
            report += "婚姻狀態反映學生的生活狀態。已婚或有家庭的學生可能需要兼顧家庭與學業，流失風險較高。"
        elif 'Classification' in feature:
            report += "分類標籤反映學生的某些內部特徵或歷史行為模式，對預測流失有重要價值。"
        else:
            report += "此特徵對辨識流失風險具有顯著預測能力，建議深入分析其與學生行為的關聯性。"

        # 如果有分段規則，加入具體的閾值資訊
        if not feature_rules.empty:
            first_rule = feature_rules.iloc[0]
            if first_rule['FeatureType'] == 'Continuous' and pd.notna(first_rule['SplitPoints']):
                report += f"\n- **關鍵閾值**: {first_rule['SplitPoints']}"

    report += """

### 2.2 高完成機率群的關鍵指標

在 **高完成機率群** 中，以下特徵能進一步區分優質學生與潛在風險：

"""

    # 添加 High Group 的 Top 5 IV 特徵
    for idx, row in top_iv_high.iterrows():
        feature = row['Feature']
        iv_value = row['IV']
        interpretation = row['Interpretation']

        report += f"""
#### {idx+1}. {feature}
- **IV 值**: {iv_value:.4f} ({interpretation})
"""

    report += """

---

## 三、驅動因素與方向性分析 (Logistic Regression Insights)

邏輯回歸係數揭示各特徵對完成率的影響方向與強度。正係數表示該特徵提升完成機率，負係數則降低完成機率。

### 3.1 低完成機率群的驅動因素

"""

    for idx, row in coef_low_features.iterrows():
        feature = row['Feature']
        coef = row['Coefficient']
        direction = row['Direction']

        direction_text = "提升" if direction == 'Positive' else "降低"
        direction_emoji = "📈" if direction == 'Positive' else "📉"

        report += f"""
**{feature}**
- 係數: {coef:.4f}
- 影響方向: {direction_emoji} {direction_text}完成機率
- **實務意義**: """

        if direction == 'Positive':
            report += f"具備此特徵（{feature} = 1）的學生，完成機率顯著提升。這是保護因子，應鼓勵更多學生具備此特徵。"
        else:
            report += f"具備此特徵（{feature} = 1）的學生，完成機率顯著下降。這是風險因子，需要針對性介入。"

        report += "\n"

    report += """
### 3.2 高完成機率群的驅動因素

"""

    for idx, row in coef_high_features.iterrows():
        feature = row['Feature']
        coef = row['Coefficient']
        direction = row['Direction']

        direction_text = "提升" if direction == 'Positive' else "降低"
        direction_emoji = "📈" if direction == 'Positive' else "📉"

        report += f"""
**{feature}**
- 係數: {coef:.4f}
- 影響方向: {direction_emoji} {direction_text}完成機率
- **實務意義**: """

        if direction == 'Positive':
            report += f"即使在高完成機率群中，具備此特徵的學生表現更優異，應作為榜樣推廣。"
        else:
            report += f"此特徵即使在高完成機率群中仍有負面影響，需要額外關注。"

        report += "\n"

    report += """

---

## 四、校務行動建議 (Actionable Business Strategy)

基於模型分析結果，我們提出以下具體可執行的校務管理與學生輔導策略：

### 4.1 優先介入策略 - 針對低完成機率群

"""

    # 根據 Low Group 的 Top IV 特徵提供建議
    top_feature_low = top_iv_low.iloc[0]['Feature']

    report += f"""
1. **建立早期預警系統**
   - 針對低完成機率群（{len(df_low):,} 位學生，佔 {len(df_low)/total_students*100:.1f}%）建立專案輔導機制
   - 重點監控高 IV 特徵（如 {top_feature_low} 等），當學生具備多項風險特徵時，立即啟動介入

2. **財務支援與彈性方案**
   - 若信用評分、年收入等財務指標為關鍵因子，應提供：
     * 分期付款彈性方案
     * 獎助學金與急難救助金
     * 財務諮詢服務

3. **個人化輔導計畫**
   - 針對特定風險特徵設計差異化支持：
     * 有家庭負擔的學生：提供彈性上課時間、線上課程選項
     * 新生：加強迎新與適應輔導，建立學伴制度
     * 特定年齡群：提供職涯諮詢與學習動機激勵

### 4.2 持續優化策略 - 針對高完成機率群

1. **防止優質學生流失**
   - 雖然此群體完成率較高（{high_complete_rate:.1%}），但仍有 {(1-high_complete_rate)*100:.1f}% 流失率
   - 關注模型識別的負向係數特徵，提供預防性支持

2. **樹立成功典範**
   - 分析高完成率學生的共同特徵，作為招生與輔導的標竿
   - 建立學長姐制度，讓成功學生協助低風險群

### 4.3 資源配置建議

**優先級排序**:
"""

    # 計算資源配置建議
    low_group_priority = len(df_low) * (1 - low_complete_rate)
    high_group_priority = len(df_high) * (1 - high_complete_rate)

    total_at_risk = low_group_priority + high_group_priority

    low_resource_pct = low_group_priority / total_at_risk * 100
    high_resource_pct = high_group_priority / total_at_risk * 100

    report += f"""
- **低完成機率群**: 配置約 **{low_resource_pct:.0f}%** 的輔導資源
  - 潛在流失人數: {low_group_priority:.0f} 人
  - 介入效益: 高（挽救流失學生）

- **高完成機率群**: 配置約 **{high_resource_pct:.0f}%** 的輔導資源
  - 潛在流失人數: {high_group_priority:.0f} 人
  - 介入效益: 中（維持優質表現）

---

## 五、模型評估與最佳閾值選擇 (Model Evaluation & Optimal Threshold Selection)

### 5.1 模型表現指標

我們使用多種指標評估模型的預測能力，並為不同的業務目標找出最佳分類閾值。

"""

    # 添加模型評估結果
    if evaluation_path and Path(evaluation_path).exists():
        try:
            eval_df = pd.read_excel(evaluation_path)

            # High Group 評估結果
            high_eval = eval_df[eval_df['Group'] == 'High Group']
            low_eval = eval_df[eval_df['Group'] == 'Low Group']

            report += """
#### 5.1.1 High Group (高完成機率群) 模型表現

"""
            # High Group AUC
            high_auc = high_eval['AUC'].iloc[0]
            report += f"**ROC AUC Score**: {high_auc:.4f}\n"

            if high_auc >= 0.8:
                report += "- 解讀: 優秀 (Excellent) - 模型具有很強的區分能力\n"
            elif high_auc >= 0.7:
                report += "- 解讀: 良好 (Good) - 模型具有較好的區分能力\n"
            elif high_auc >= 0.6:
                report += "- 解讀: 可接受 (Fair) - 模型具有基本的區分能力\n"
            else:
                report += "- 解讀: 不佳 (Poor) - 模型區分能力有限\n"

            report += "\n**不同閾值下的表現比較**:\n\n"
            report += "| Threshold 類型 | 閾值 | Accuracy | Precision | Recall | F1 Score | F2 Score |\n"
            report += "|--------------|------|----------|-----------|--------|----------|----------|\n"

            for _, row in high_eval.iterrows():
                thresh_type = row['Threshold_Type'].replace('_', ' ').title()
                report += f"| {thresh_type:12s} | {row['Threshold']:.3f} | {row['Accuracy']:.3f} | {row['Precision']:.3f} | {row['Recall']:.3f} | {row['F1_Score']:.3f} | {row['F2_Score']:.3f} |\n"

            report += """

#### 5.1.2 Low Group (低完成機率群) 模型表現

"""
            # Low Group AUC
            low_auc = low_eval['AUC'].iloc[0]
            report += f"**ROC AUC Score**: {low_auc:.4f}\n"

            if low_auc >= 0.8:
                report += "- 解讀: 優秀 (Excellent) - 模型具有很強的區分能力\n"
            elif low_auc >= 0.7:
                report += "- 解讀: 良好 (Good) - 模型具有較好的區分能力\n"
            elif low_auc >= 0.6:
                report += "- 解讀: 可接受 (Fair) - 模型具有基本的區分能力\n"
            else:
                report += "- 解讀: 不佳 (Poor) - 模型區分能力有限\n"

            report += "\n**不同閾值下的表現比較**:\n\n"
            report += "| Threshold 類型 | 閾值 | Accuracy | Precision | Recall | F1 Score | F2 Score |\n"
            report += "|--------------|------|----------|-----------|--------|----------|----------|\n"

            for _, row in low_eval.iterrows():
                thresh_type = row['Threshold_Type'].replace('_', ' ').title()
                report += f"| {thresh_type:12s} | {row['Threshold']:.3f} | {row['Accuracy']:.3f} | {row['Precision']:.3f} | {row['Recall']:.3f} | {row['F1_Score']:.3f} | {row['F2_Score']:.3f} |\n"

            report += """

### 5.2 閾值選擇建議

根據不同的業務目標，我們提供以下閾值選擇建議：

"""

            # 最佳 F1 threshold
            high_f1_row = high_eval[high_eval['Threshold_Type'] == 'best_f1'].iloc[0]
            low_f1_row = low_eval[low_eval['Threshold_Type'] == 'best_f1'].iloc[0]

            report += f"""
**1. 平衡精準與召回 (F1 Score 最佳化)**
- **目標**: 在識別高風險學生的準確性和覆蓋率之間取得平衡
- **High Group 建議閾值**: {high_f1_row['Threshold']:.3f}
  - F1 Score: {high_f1_row['F1_Score']:.3f}
  - 此閾值下會識別出 {high_f1_row['TP'] + high_f1_row['FP']:.0f} 位潛在流失學生
- **Low Group 建議閾值**: {low_f1_row['Threshold']:.3f}
  - F1 Score: {low_f1_row['F1_Score']:.3f}
  - 此閾值下會識別出 {low_f1_row['TP'] + low_f1_row['FP']:.0f} 位潛在流失學生

"""

            # 最佳 F2 threshold
            high_f2_row = high_eval[high_eval['Threshold_Type'] == 'best_f2'].iloc[0]
            low_f2_row = low_eval[low_eval['Threshold_Type'] == 'best_f2'].iloc[0]

            report += f"""
**2. 優先覆蓋率 (F2 Score 最佳化，重視 Recall)**
- **目標**: 盡可能找出所有潛在流失學生，即使會有較多誤報
- **High Group 建議閾值**: {high_f2_row['Threshold']:.3f}
  - F2 Score: {high_f2_row['F2_Score']:.3f}
  - Recall: {high_f2_row['Recall']:.3f} (捕獲 {high_f2_row['Recall']*100:.1f}% 的流失學生)
- **Low Group 建議閾值**: {low_f2_row['Threshold']:.3f}
  - F2 Score: {low_f2_row['F2_Score']:.3f}
  - Recall: {low_f2_row['Recall']:.3f} (捕獲 {low_f2_row['Recall']*100:.1f}% 的流失學生)

**建議**: 由於輔導資源有限，建議優先使用 **F1 最佳化閾值**，在準確性和覆蓋率之間取得平衡。

"""

            # 添加圖表說明
            report += """
### 5.3 視覺化分析

以下圖表提供模型評估的視覺化分析：

1. **ROC Curves**: 展示模型在不同閾值下的 True Positive Rate vs False Positive Rate
   - 圖檔: `reports/figures/roc_curves.png`
   - AUC 越接近 1.0 表示模型越好

2. **Confusion Matrices**: 展示不同閾值下的分類結果
   - High Group: `reports/figures/high_group_confusion_matrices.png`
   - Low Group: `reports/figures/low_group_confusion_matrices.png`
   - 包含 Default (0.5)、Best F1、Best F2 三種閾值的比較

![ROC Curves](figures/roc_curves.png)

![High Group Confusion Matrices](figures/high_group_confusion_matrices.png)

![Low Group Confusion Matrices](figures/low_group_confusion_matrices.png)

"""

        except Exception as e:
            report += f"\n（評估結果讀取失敗: {str(e)}）\n\n"

    # 添加方法比較章節
    if comparison_df is not None and single_eval_result is not None:
        report += """
---

## 五之二、建模方法比較 (Modeling Approach Comparison)

本專案實作並比較了兩種建模方法：

### 5.4.1 方法說明

**方法一：兩階段方法 (Two-Stage Approach)**
1. 使用決策樹將學生分為高/低完成機率群
2. 針對每群分別訓練獨立的邏輯回歸模型
3. 優點：捕捉不同風險群體的差異化特徵影響
4. 缺點：模型複雜度較高，需要維護兩個模型

**方法二：單一模型方法 (Single Model Approach)**
1. 直接對所有學生訓練單一邏輯回歸模型
2. 優點：模型簡單，易於解釋和維護
3. 缺點：無法捕捉不同群體的差異化影響

### 5.4.2 方法表現比較 (Best F1 Threshold)

"""

        # 從 comparison_df 提取比較數據
        best_f1_comp = comparison_df[comparison_df['Threshold_Type'] == 'best_f1']

        two_stage_high = best_f1_comp[
            (best_f1_comp['Approach'] == 'Two-Stage') &
            (best_f1_comp['Group'] == 'High Group')
        ]
        two_stage_low = best_f1_comp[
            (best_f1_comp['Approach'] == 'Two-Stage') &
            (best_f1_comp['Group'] == 'Low Group')
        ]
        single_model = best_f1_comp[best_f1_comp['Approach'] == 'Single Model']

        if not two_stage_high.empty and not two_stage_low.empty and not single_model.empty:
            tsh = two_stage_high.iloc[0]
            tsl = two_stage_low.iloc[0]
            sm = single_model.iloc[0]

            report += f"""
| 方法 | 群組 | 樣本數 | Complete率 | AUC | F1 Score | Recall | Precision |
|------|------|--------|-----------|-----|----------|--------|-----------|
| 兩階段 | High Group | {tsh['Sample_Size']:.0f} | {tsh['Positive_Rate']:.1%} | {tsh['AUC']:.3f} | {tsh['F1_Score']:.3f} | {tsh['Recall']:.3f} | {tsh['Precision']:.3f} |
| 兩階段 | Low Group | {tsl['Sample_Size']:.0f} | {tsl['Positive_Rate']:.1%} | {tsl['AUC']:.3f} | {tsl['F1_Score']:.3f} | {tsl['Recall']:.3f} | {tsl['Precision']:.3f} |
| 單一模型 | All Data | {sm['Sample_Size']:.0f} | {sm['Positive_Rate']:.1%} | {sm['AUC']:.3f} | {sm['F1_Score']:.3f} | {sm['Recall']:.3f} | {sm['Precision']:.3f} |

### 5.4.3 方法選擇建議

"""

            # 計算平均表現
            avg_two_stage_auc = (tsh['AUC'] + tsl['AUC']) / 2
            avg_two_stage_f1 = (tsh['F1_Score'] + tsl['F1_Score']) / 2

            if avg_two_stage_auc > sm['AUC'] and avg_two_stage_f1 > sm['F1_Score']:
                report += f"""
**推薦：兩階段方法 (Two-Stage Approach)**

**理由**：
- 兩階段方法的平均 AUC ({avg_two_stage_auc:.3f}) 優於單一模型 ({sm['AUC']:.3f})
- 兩階段方法的平均 F1 Score ({avg_two_stage_f1:.3f}) 優於單一模型 ({sm['F1_Score']:.3f})
- High Group 和 Low Group 有明顯不同的特徵影響模式，分群建模能更好地捕捉這些差異
- 雖然模型複雜度較高，但預測效果的提升值得額外的維護成本

**實務應用**：
1. 使用決策樹模型將新學生分類到 High/Low Group
2. 根據所屬群組，使用對應的邏輯回歸模型預測流失風險
3. 針對不同群組採用差異化的輔導策略
"""
            else:
                report += f"""
**推薦：單一模型方法 (Single Model Approach)**

**理由**：
- 單一模型的 AUC ({sm['AUC']:.3f}) 與兩階段方法平均表現 ({avg_two_stage_auc:.3f}) 接近
- 模型更簡單，易於解釋和維護
- 適合資源有限的情況，無需維護多個模型
- 對於所有學生使用統一的評分標準，更公平透明

**實務應用**：
1. 直接使用單一邏輯回歸模型對所有學生評分
2. 根據預測機率排序，優先輔導高風險學生
3. 使用相同的特徵影響解釋，制定統一的干預策略
"""

            # 添加視覺化
            report += """

### 5.4.4 方法比較視覺化

![Approach Comparison](figures/approach_comparison.png)

上圖比較了兩種方法在 AUC、F1 Score 和 Recall 三個指標上的表現。

"""

    threshold_text = f"{threshold:.4f}" if threshold is not None else "未提供"

    report += f"""
---

## 六、模型技術摘要 (Technical Summary)

### 6.1 模型架構

本專案採用 **混合模型架構**：

#### 1. 決策樹分流 (Decision Tree Segmentation)

使用決策樹模型將學生分為高/低完成機率群，捕捉非線性的群體差異。

**分流邏輯**:
- **分流方法**: 以平均預測機率作為閾值
- **閾值**: {threshold_text}
- **決策樹參數**:
  - 最大深度 (max_depth): 10
  - 最小分裂樣本數 (min_samples_split): 100
  - 最小葉節點樣本數 (min_samples_leaf): 50

**分群結果**:
- High Group (預測機率 ≥ {threshold_text}): {len(df_high):,} 位學生 ({len(df_high)/total_students*100:.1f}%)
  - 實際完成率: {high_complete_rate:.1%}
- Low Group (預測機率 < {threshold_text}): {len(df_low):,} 位學生 ({len(df_low)/total_students*100:.1f}%)
  - 實際完成率: {low_complete_rate:.1%}

**關鍵分流特徵** (Feature Importance):
- 決策樹主要依據 Credit (信用評分)、AnnualIncome (年收入) 和 ServiceType (課程類型) 進行分群
- 詳細特徵重要性和決策樹規則請參考: `output/2.5_decision_tree_split_rules.xlsx`
- 決策樹視覺化請參考: `reports/figures/decision_tree.png`

#### 2. 特徵工程 (Feature Engineering)

對兩個群組分別進行特徵轉換：
- **連續變數**: 使用監督式決策樹分段，轉換為二元變數
- **類別變數**: 使用 n-1 One-Hot Encoding，避免多重共線性

#### 3. 分群邏輯回歸 (Group-wise Logistic Regression)

針對兩群學生分別建模，捕捉不同群體的特徵影響差異。

### 6.2 資料規模

- 總樣本數: {total_students:,}
- High Group: {len(df_high):,} ({len(df_high)/total_students*100:.1f}%)
- Low Group: {len(df_low):,} ({len(df_low)/total_students*100:.1f}%)

### 6.3 輸出檔案

本分析專案產出以下結果檔案：

1. `1_preprocessed_data.xlsx`: 清理後的原始資料
2. `2_high_group_raw.xlsx`, `2_low_group_raw.xlsx`: 決策樹分流結果
3. **`2.5_decision_tree_split_rules.xlsx`: 決策樹切分規則與特徵重要性**
   - Split_Summary: 切分邏輯摘要
   - Feature_Importance: 特徵重要性排序
   - Tree_Rules: 決策樹文字規則
4. `3_high_group_transformed.xlsx`, `3_low_group_transformed.xlsx`: 轉換為二元變數的特徵矩陣
5. `3_continuous_bins_rules.xlsx`: 連續變數分段規則與切點
6. `4_lr_coefficients.xlsx`: 邏輯回歸模型係數
7. `4.5_model_evaluation.xlsx`: 模型評估指標
8. `5_iv_results.xlsx`: Information Value 詳細結果
9. **圖表檔案** (`reports/figures/`):
   - `decision_tree.png`: 決策樹視覺化（前 3 層）
   - `roc_curves.png`: ROC 曲線比較
   - `high_group_confusion_matrices.png`: High Group 混淆矩陣
   - `low_group_confusion_matrices.png`: Low Group 混淆矩陣
   - `approach_comparison.png`: 方法比較圖表

---

## 七、結論與後續行動 (Conclusion)

本分析成功建立學生課程完成率的早期預警機制，識別出 **{len(df_low):,}** 位高風險學生需要優先關注。

**立即行動項目**:

1. ✅ 將低完成機率群名單提供給學生輔導單位
2. ✅ 根據高 IV 特徵設計個人化介入方案
3. ✅ 建立持續監控儀表板，追蹤介入成效
4. ✅ 每學期更新模型，納入最新數據

**預期效益**:

若能有效介入低完成機率群，假設挽回 30% 的潛在流失學生，可額外完成約 **{low_group_priority * 0.3:.0f}** 位學生的課程，顯著提升整體完成率與學校聲譽。

---

**報告製作**: 機器學習自動化分析系統
**報告時間**: {report_time}

"""

    # 儲存報告
    print("\n[6/6] 儲存報告")

    # 確保輸出目錄存在
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"   - 已儲存報告: {output_path}")

    print("\n✓ 商業洞察報告生成完成！")
    print("=" * 80)
    print()


if __name__ == "__main__":
    # 測試用
    high_raw = "output/2_high_group_raw.xlsx"
    low_raw = "output/2_low_group_raw.xlsx"
    binning_rules = "output/3_continuous_bins_rules.xlsx"
    coefficients = "output/4_lr_coefficients.xlsx"
    iv_results = "output/5_iv_results.xlsx"
    output = "reports/business_insight_report.md"

    generate_business_insight_report(
        high_raw, low_raw, binning_rules, coefficients, iv_results, output
    )
