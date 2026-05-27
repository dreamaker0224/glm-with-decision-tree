# 模型預測使用說明

## 概述

本專案已保存以下模型，可對新資料進行預測：

1. **決策樹模型** (`models/decision_tree_model.pkl`): 用於將學生分為高/低完成機率群
2. **High Group 邏輯回歸模型** (`models/logistic_regression_high_group.pkl`): 預測高完成機率群學生
3. **Low Group 邏輯回歸模型** (`models/logistic_regression_low_group.pkl`): 預測低完成機率群學生

## 快速開始

### 1. Python API 方式

```python
from src.predict import StudentCompletionPredictor
import pandas as pd

# 讀取新資料
df_new = pd.read_excel('new_students_data.xlsx')

# 初始化預測器
predictor = StudentCompletionPredictor()

# 進行預測
results = predictor.predict(df_new, return_details=True)

# 查看結果
print(results)
```

### 2. 從檔案預測

```python
from src.predict import predict_from_file

# 直接從檔案讀取並預測
predictions = predict_from_file(
    input_path='new_students_data.xlsx',
    output_path='predictions/my_predictions.xlsx'
)
```

## 輸入資料格式

新資料必須包含與訓練資料相同的欄位（除了 `ServiceStatus` 目標變數）：

- `ServiceType`: 課程類型
- `Credit`: 信用評分
- `Age`: 年齡
- `Government`: 是否政府客戶
- `Market`: 市場區隔
- `NewCustomer`: 是否新客戶
- `PaymentMethod`: 付款方式
- `Gender`: 性別
- `Dependents`: 撫養人數
- `MaritalStatus`: 婚姻狀態
- `Classification1`: 分類1
- `Classification2`: 分類2
- `AnnualIncome`: 年收入

**注意**: 不要包含以下欄位（會被自動移除）：
- `CustID`: 學生 ID
- `ServiceStartDate`: 服務開始日期
- `WeeksWithService`: 服務週數
- `Year`, `Month`: 時間變數

## 輸出結果說明

預測結果包含以下欄位：

| 欄位 | 說明 |
|------|------|
| `Group` | 學生所屬群組（'High' 或 'Low'） |
| `DT_PredictedProba` | 決策樹預測的完成機率 |
| `LR_PredictedProba` | 邏輯回歸預測的完成機率 |
| `Prediction` | 最終預測結果（1=Complete, 0=Quit） |

## 範例

```python
# 範例 1: 單筆學生資料預測
import pandas as pd
from src.predict import StudentCompletionPredictor

# 創建單筆學生資料
new_student = pd.DataFrame({
    'ServiceType': ['A'],
    'Credit': [55000],
    'Age': [30],
    'Government': ['N'],
    'Market': ['Western'],
    'NewCustomer': ['N'],
    'PaymentMethod': ['Monthly'],
    'Gender': ['F'],
    'Dependents': ['No'],
    'MaritalStatus': ['Single'],
    'Classification1': [0],
    'Classification2': [1],
    'AnnualIncome': [50000]
})

# 預測
predictor = StudentCompletionPredictor()
result = predictor.predict(new_student, return_details=True)

print(f"群組: {result['Group'].iloc[0]}")
print(f"決策樹機率: {result['DT_PredictedProba'].iloc[0]:.4f}")
print(f"邏輯回歸機率: {result['LR_PredictedProba'].iloc[0]:.4f}")
print(f"預測結果: {'Complete' if result['Prediction'].iloc[0] == 1 else 'Quit'}")
```

```python
# 範例 2: 批量預測
from src.predict import predict_from_file

# 從 Excel 檔案讀取並預測
predictions = predict_from_file(
    input_path='data/new_enrollments_2026.xlsx',
    output_path='predictions/may_2026_predictions.xlsx'
)

# 查看統計
print(f"總學生數: {len(predictions)}")
print(f"預測完成: {predictions['Prediction'].sum()} 人")
print(f"預測流失: {(predictions['Prediction'] == 0).sum()} 人")
```

## 模型檔案結構

```
models/
├── decision_tree_model.pkl              # 決策樹模型
├── decision_tree_metadata.pkl           # 決策樹元資訊（閾值、特徵名稱等）
├── logistic_regression_high_group.pkl   # High Group 邏輯回歸模型
├── logistic_regression_low_group.pkl    # Low Group 邏輯回歸模型
└── logistic_regression_metadata.pkl     # 邏輯回歸元資訊（特徵列表等）

output/
└── 3_continuous_bins_rules.xlsx         # 連續變數分段規則（用於特徵轉換）
```

## 注意事項

1. **資料格式一致性**: 新資料必須與訓練資料的格式一致
2. **缺失值處理**: 系統會自動填補 `PaymentMethod` 和 `Gender` 的缺失值
3. **類別值**: 如果新資料包含訓練時沒見過的類別值，該欄位會被設為 0
4. **預測閾值**: 目前使用 0.5 作為分類閾值，可以根據業務需求調整

## 更新模型

如果要用新資料重新訓練模型：

```bash
# 1. 準備新的訓練資料，放到 data/dataset.xlsx
# 2. 執行完整訓練流程
python3 main.py

# 新的模型會自動保存到 models/ 目錄
```

## 疑難排解

### 找不到模型檔案

確保已經執行過一次完整的訓練流程：
```bash
python3 main.py
```

### 預測結果異常

檢查輸入資料是否包含所有必要欄位：
```python
required_columns = [
    'ServiceType', 'Credit', 'Age', 'Government', 'Market',
    'NewCustomer', 'PaymentMethod', 'Gender', 'Dependents',
    'MaritalStatus', 'Classification1', 'Classification2', 'AnnualIncome'
]

missing = set(required_columns) - set(df.columns)
if missing:
    print(f"缺少欄位: {missing}")
```

### 類別值不匹配

新資料中的類別值必須與訓練資料一致。如果出現新的類別值，可能需要：
1. 將新類別對應到現有類別
2. 或重新訓練模型包含新類別

## 技術細節

### 預測流程

1. **資料預處理**: 移除不需要的欄位，填補缺失值
2. **決策樹分群**: 使用決策樹模型計算完成機率，根據閾值 (0.1808) 分為 High/Low Group
3. **特徵轉換**: 使用保存的分段規則將連續變數轉為二元變數，對類別變數進行 One-Hot Encoding
4. **邏輯回歸預測**: 根據群組使用對應的邏輯回歸模型預測完成機率
5. **二元分類**: 使用 0.5 閾值將機率轉為 Complete (1) 或 Quit (0)

### 模型資訊

- **決策樹**: 
  - max_depth=10
  - min_samples_split=100
  - min_samples_leaf=50
  - 閾值: 0.1808

- **邏輯回歸**:
  - penalty='l2'
  - C=1.0
  - solver='lbfgs'
  - max_iter=1000

## 相關文件

- [專案說明](../CLAUDE.md)
- [商業洞察報告](../reports/business_insight_report.md)
- [模型評估結果](../output/4.5_model_evaluation.xlsx)
