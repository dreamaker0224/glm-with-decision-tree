# Role & Context
你之前為我產出的「學生課程完成率預測專案」架構非常完整，Excel 輸出與模組化也做得很好。但我在檢視分析報告與跑出的數據時，發現了兩個在資料科學與商業邏輯上的嚴重錯誤（Bugs）。
請根據以下的錯誤說明，修改對應的 Python 模組並重新提供給我正確的程式碼。

# Bug Reports & Fix Instructions

## Bug 1：Information Value (IV) 計算的時機點錯誤
- **錯誤狀況**：目前的程式碼是在進行 One-Hot Encoding (產生 n-1 個 0/1 Dummy variables) 之後，才去對這些 0/1 變數（如 `AnnualIncome_Bin_4`）單獨計算 IV 值。
- **為什麼這是錯的**：在數學與實務定義上，IV 值是用來衡量「單一完整特徵」的總體資訊量。把一個特徵拆碎成多個 Dummy variables 分別算 IV，會嚴重低估該特徵的預測力，且在商業報告上難以解釋。
- **修正要求**：
  1. 請修改資料流，**IV 值的計算必須發生在「連續變數完成分箱 (Binning) 之後」，但在「One-Hot Encoding 之前」**。
  2. 請針對「原始名目變數」與「分段後的連續變數」計算整體的 IV 值。例如 `AnnualIncome` 這個特徵只能算出「一個」總體 IV 值，而不是 Bin_1, Bin_2 各有一個。
  3. 修改 `src/iv_calculator.py`，確保它吃到的資料是還沒被 OHE 展開的。

## Bug 2：包含不應作為特徵的時間變數 (Data Leakage)
- **錯誤狀況**：在報告中，`Year_2011` 被當作預測指標，且 IV 值很高。
- **為什麼這是錯的**：年份 (`Year`) 這類時間變數不能放入模型中，這會導致模型死記歷史年份，完全失去預測未來學生的泛化能力。
- **修正要求**：
  1. 在 `src/preprocess.py` 的 Drop Columns 清單中，必須加上 `Year`。
  2. 確保 `Year` 不會進入後續的分流、分箱、IV 計算與邏輯回歸模型中。

# Files to Update
請根據上述修正指示，重新產出並完整提供以下幾個檔案的最新程式碼（請確保 Excel 輸出的流程依然保留）：
1. `src/preprocess.py` (新增移除 Year)
2. `src/feature_transform.py` (確保明確切分 Binning 邏輯與 OHE 邏輯)
3. `src/iv_calculator.py` (修正 IV 計算邏輯，針對整體特徵而非 Dummy variables)
4. `src/report_generator.py` (修正報告文案抓取的變數名稱與解讀邏輯)
5. `main.py` (如果資料流有變動，請同步更新主程式的串接順序)

請直接給出修改後的完整程式碼，並在註解中標示「# 修正：...」讓我能快速看出你改了哪裡。
