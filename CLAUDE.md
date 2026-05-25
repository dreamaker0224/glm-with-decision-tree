# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

學生課程完成率預測分析系統 (Student Course Completion Rate Prediction System)

**Purpose**: 建立早期預警機制，識別高風險學生並配置輔導資源。使用混合模型架構（決策樹分流 + 分群邏輯回歸）預測學生是否會完成課程。

**Target Variable**: `ServiceStatus`
- `Complete` (1): 順利完成課程
- `Quit` (0): 中途退出

**Critical Data Constraints**:
- 必須移除欄位：`CustID`, `ServiceStartDate`, `WeeksWithService`（避免時間資訊外洩）
- 所有特徵最終必須轉換為 **0/1 二元變數**

## Running the Project

### Execute Full Pipeline
```bash
python3 main.py
```

Runs the complete 6-step analysis pipeline (takes ~1 minute):
1. Data preprocessing
2. Decision tree segmentation (High/Low Group split)
3. Feature transformation (supervised binning + one-hot encoding)
4. Group-wise logistic regression
5. Information Value (IV) calculation
6. Business insight report generation

### Run Individual Modules (for testing)
```bash
python3 src/preprocess.py
python3 src/decision_tree.py
python3 src/feature_transform.py
python3 src/logistic_regression.py
python3 src/iv_calculator.py
python3 src/report_generator.py
```

### Check Dependencies
```bash
python3 -c "import pandas, numpy, sklearn, openpyxl; print('All dependencies OK')"
```

## Architecture

### Pipeline Flow

```
data/dataset.xlsx
    ↓
[1. Preprocess] → output/1_preprocessed_data.xlsx
    ↓
[2. Decision Tree Split] → output/2_high_group_raw.xlsx
                         → output/2_low_group_raw.xlsx
    ↓
[3. Feature Transform] → output/3_high_group_transformed.xlsx
                       → output/3_low_group_transformed.xlsx
                       → output/3_continuous_bins_rules.xlsx
    ↓
[4. Logistic Regression] → output/4_lr_coefficients.xlsx
    ↓
[5. IV Calculation] → output/5_iv_results.xlsx
    ↓
[6. Report Generation] → reports/business_insight_report.md
```

### Key Design Decisions

**1. Two-Stage Modeling Approach**
- **Stage 1 (Decision Tree)**: Split students into risk groups using mean predicted probability as threshold
- **Stage 2 (Logistic Regression)**: Train separate models for High Group and Low Group
- **Rationale**: Different risk groups have different feature dynamics; separate models capture this better

**2. Feature Engineering for Logistic Regression**
- **Continuous Variables**: Use single-feature decision trees for supervised binning (finds optimal split points)
- **Categorical Variables**: One-Hot Encoding with `drop='first'` (n-1 encoding to avoid multicollinearity)
- **Critical**: ALL features must be 0/1 binary variables before entering logistic regression
- **Split Points Recording**: `output/3_continuous_bins_rules.xlsx` stores the exact thresholds for business interpretation

**3. Information Value (IV) Calculation**
```python
# For each feature category:
WoE = ln(pct_good / pct_bad)
IV = (pct_good - pct_bad) × WoE

# Total IV = sum of all categories
# IV > 0.1 = useful predictor for early warning
```

### Module Responsibilities

| Module | Input | Output | Key Function |
|--------|-------|--------|--------------|
| `preprocess.py` | `data/dataset.xlsx` | `output/1_preprocessed_data.xlsx` | Clean data, encode target, handle missing values |
| `decision_tree.py` | `output/1_*.xlsx` | `output/2_high/low_group_raw.xlsx` | Train DT, predict probabilities, split by mean threshold |
| `feature_transform.py` | `output/2_*.xlsx` | `output/3_*_transformed.xlsx` + rules | Convert all features to 0/1 binary variables |
| `logistic_regression.py` | `output/3_*.xlsx` | `output/4_lr_coefficients.xlsx` | Train separate LR models per group |
| `iv_calculator.py` | `output/3_*.xlsx` | `output/5_iv_results.xlsx` | Calculate WoE and IV for feature selection |
| `report_generator.py` | All `output/*.xlsx` | `reports/business_insight_report.md` | Generate business-focused analysis report |

## Modifying the Pipeline

### Adding a New Feature Engineering Step

1. Modify `src/feature_transform.py`:
   - Update `identify_feature_types()` if needed
   - Add custom transformation logic in `transform_features_to_binary()`
   - Ensure output is still 0/1 binary

2. Update binning rules recording to include new feature types

### Changing the Segmentation Logic

Edit `src/decision_tree.py`:
- Current threshold: `threshold = predict_proba.mean()` (mean probability)
- Alternative: Use median, quantile, or custom business rule
- **Important**: Update the split logic in both High/Low group creation

### Modifying Model Hyperparameters

**Decision Tree** (`src/decision_tree.py`):
```python
DecisionTreeClassifier(
    max_depth=10,           # Controls tree complexity
    min_samples_split=100,  # Minimum samples to split
    min_samples_leaf=50,    # Minimum samples per leaf
)
```

**Logistic Regression** (`src/logistic_regression.py`):
```python
LogisticRegression(
    penalty='l2',
    C=1.0,              # Regularization strength (smaller = stronger)
    solver='lbfgs',
)
```

**Supervised Binning** (`src/feature_transform.py`):
```python
DecisionTreeClassifier(
    max_leaf_nodes=max_bins,  # Controls number of bins (default: 5)
    min_samples_leaf=50,
)
```

## Output Files

All outputs are Excel files (.xlsx) for business user consumption:

- `1_preprocessed_data.xlsx`: Cleaned dataset (51,058 records)
- `2_high_group_raw.xlsx`: Students with high completion probability (~40%)
- `2_low_group_raw.xlsx`: Students with low completion probability (~60%)
- `3_*_transformed.xlsx`: Binary feature matrices (all 0/1)
- `3_continuous_bins_rules.xlsx`: **Critical for business interpretation** - contains exact split points
- `4_lr_coefficients.xlsx`: Model coefficients showing feature impact direction
- `5_iv_results.xlsx`: Feature predictive power (4 sheets: details, summary, top features)

**Business Report**: `reports/business_insight_report.md` (繁體中文)
- Targets school administrators, not data scientists
- Translates IV values and coefficients into actionable recommendations
- Includes resource allocation suggestions based on risk group sizes

## Important Constraints

1. **Binary Features Only**: Logistic regression requires all features to be 0/1. Do NOT pass continuous or categorical (n>2) variables directly.

2. **Supervised Binning**: Continuous variables must use decision tree binning (not equal-width or equal-frequency) to find optimal splits based on the target.

3. **n-1 Encoding**: Categorical variables must drop one category (`drop='first'`) to avoid the dummy variable trap.

4. **Threshold Rule**: Decision tree split uses the **mean predicted probability** (not median, not fixed value like 0.5). This ensures groups reflect the actual class distribution.

5. **Excel Output**: Every pipeline step must save its output as .xlsx. This is a hard requirement for business stakeholders.

## Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'sklearn'`
```bash
python3 -m pip install scikit-learn --break-system-packages
# or: sudo apt-get install python3-sklearn
```

**Issue**: Pipeline fails at feature transformation
- Check if any continuous variable has < 100 valid values (too few for binning)
- Verify no new missing values were introduced
- Ensure target variable is still 0/1 encoded

**Issue**: IV calculation returns all zeros
- Verify features are 0/1 binary (check column data types)
- Ensure target variable has both classes (0 and 1)
- Check if feature has zero variance (all same value)

**Issue**: Report shows incorrect statistics
- Re-run entire pipeline from `main.py` (modules depend on each other)
- Check if threshold value was passed correctly to report generator
- Verify all output files exist before report generation
