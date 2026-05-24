# Role
你是一位兼具資深資料科學家與校務決策顧問（Business Analytics Consultant）雙重身份的專家。請幫我建構一個用於分析「學生課程完成率」的機器學習模組化專案，並基於模型結果產出具備真實商務/校務管理價值的分析報告。

# Project Background & Goal
本專案的數據來自一間學校。核心目標是預測並分析學生是否能夠順利完成課程（Complete），或是中途退出（Quit）。學校管理階層希望透過此模型建立「早期預警機制」，找出影響學生流失的核心關鍵特徵，以便精準配置輔導資源。

# Data Requirements
1. 資料集名稱：`dataset`（`dataset.xlsx`）。
2. 目標變數 (Target/Class)：`ServiceStatus`。包含兩個類別：
   - `Complete`：代表順利完成課程（正類別，預處理時請編碼為 1）。
   - `Quit`：代表中途退出/休學（負類別，預處理時請編碼為 0）。
3. 必須排除的欄位 (Drop Columns)：`CustID`（學生 ID）、`ServiceStartDate`（服務/開課日期）、`WeeksWithService`（已上課週數，避免時間資訊外洩導致模型作弊）。請在特徵工程前將這三個欄位移除。

# Pipeline & Modular Requirements
專案必須高度模組化，請嚴格遵守以下指定的資料夾結構。**【重要要求】：每個步驟執行完畢後，都必須利用 pandas 將該階段的處理結果、運算數據或規則輸出為 Excel 檔案 (.xlsx) 保存。**

1. **資料預處理 (src/preprocess.py)**：
   - 讀取資料、將 `ServiceStatus` 轉換為數值（1/0）、移除無用欄位、處理缺失值。
   - **Excel 輸出**：將初步清理完成的資料集存為 `output/1_preprocessed_data.xlsx`。

2. **決策樹分流 (src/decision_tree.py)**：
   - 訓練 Decision Tree 模型，計算每位學生的 `Complete` 預測機率。
   - 計算資料集中 `Complete` 的真實比例（即平均機率），**並嚴格以此「平均機率」作為 Threshold 切點**。
   - 將學生切分為兩群：「大於等於平均機率群 (High Group)」與「小於平均機率群 (Low Group)」。
   - **Excel 輸出**：將切分好的兩群資料分別存為 `output/2_high_group_raw.xlsx` 與 `output/2_low_group_raw.xlsx`。

3. **羅吉斯迴歸專屬特徵轉換 (src/feature_transform.py)**：
   - 為了符合 Logistic Regression 的要求，**所有進入模型的特徵都必須是 0 與 1 的二元變數**。
   - **連續變數 (Continuous Variables)**：必須先使用單一特徵的 Decision Tree 對該變數進行監督式分段 (Binning) 找出最佳切點，將其轉換為類別區間後，再轉為 0/1 的 Dummy variables。**【重要】：必須透過程式提取 Decision Tree 找出的切點（Split Points/Thresholds），將每個連續變數的「分段規則與切點數值」記錄下來。**
   - **名目變數 (Nominal Variables)**：必須使用 One-Hot Encoding，且 n 個類別只能產生 n-1 個變數（即 `drop='first'`），以避免多重共線性 (Multicollinearity)。
   - 針對 High Group 與 Low Group 分別執行此轉換邏輯。
   - **Excel 輸出**：
     - 將轉換後（全為 0/1 變數）的特徵矩陣存為 `output/3_high_group_transformed.xlsx` 與 `output/3_low_group_transformed.xlsx`。
     - **將所有連續變數的「分段規則與切點紀錄」存為 `output/3_continuous_bins_rules.xlsx`。**

4. **分群邏輯回歸 (src/logistic_regression.py)**：
   - 使用轉換後（全為 0/1 特徵）的 High Group 與 Low Group 資料，分別訓練獨立的 Logistic Regression 模型。
   - **Excel 輸出**：將兩群的特徵與對應的模型係數 (Coefficients)、截距 (Intercept) 整理成 DataFrame，存為 `output/4_lr_coefficients.xlsx`。

5. **特徵價值計算 (src/iv_calculator.py)**：
   - 實作 Weight of Evidence (WoE) 與 Information Value (IV) 的計算邏輯。
   - 分別計算兩群學生中，各個特徵（轉換為 0/1 後的版本）的 IV 值。
   - **Excel 輸出**：將 High Group 與 Low Group 的特徵 IV 值明細存為 `output/5_iv_results.xlsx`。

6. **主程式 (main.py)**：
   - 負責串接上述模組，執行完整的 Pipeline，確保所有 Excel 檔正確產出，並觸發報告生成邏輯。

# Business Insight Report Requirement
請實作一個報告生成模組 `src/report_generator.py`（最終輸出至 `reports/business_insight_report.md`）。該報告必須脫離純技術術語，**以學校營運與學生輔導的「真實商業視角」進行撰寫**，包含：
1. **執行摘要 (Executive Summary)**：用一句話說明高/低風險群體的學生核心差異。
2. **早期預警指標分析 (IV 值解讀)**：
   - 說明在「低完成機率群」中，哪些特徵的 IV 值最高？這代表什麼樣的學生行為或背景是瀕臨退出的關鍵。請結合**連續變數的分段切點**進行說明（例如：「當出勤率低於 75% 時，流失風險顯著提升」）。
3. **驅動因素與方向性 (邏輯回歸係數解讀)**：
   - 由於所有特徵已轉為 0/1，請直觀解釋「具備某特徵（=1）」會如何拉高或降低完成率（參考對數勝算比 Log-Odds 的概念）。對比兩群的差異。
4. **校務行動建議 (Actionable Business Strategy)**：
   - 結合模型找出的閾值與變數，提供 2-3 個具體的營運與輔導介入建議。

# Folder Structure
請嚴格按照以下架構生成檔案，並為每個檔案提供完整的、可執行的 Python 程式碼：

project_root/
├── data/
│   └── dataset.xlsx                 # 原始數據
├── output/                         # 放置所有過程中產出的 Excel 檔案
├── src/
│   ├── __init__.py
│   ├── preprocess.py               # 資料清洗
│   ├── decision_tree.py            # DT 預測與平均機率切分
│   ├── feature_transform.py        # DT 連續變數分段(含切點紀錄)與 n-1 One-Hot Encoding
│   ├── logistic_regression.py      # 分群 LR 模型訓練
│   ├── iv_calculator.py            # 計算 WoE 與 IV 值
│   └── report_generator.py         # 將模型統計量轉譯為校務商業報告
├── reports/
│   └── business_insight_report.md  # 最終產出的繁體中文商業分析報告
├── requirements.txt                # 需包含 openpyxl, pandas, scikit-learn 等套件
└── main.py                         # 執行進入點

# Output Format
請依序輸出上述每個檔案的完整程式碼。程式碼內請加上詳細的繁體中文註解。確保有使用 `openpyxl` 或 `xlsxwriter` 來支援 DataFrame 的 `.to_excel()` 功能。