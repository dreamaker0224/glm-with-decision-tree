"""
預測模組 (Prediction Module)
載入訓練好的模型，對新資料進行預測
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class StudentCompletionPredictor:
    """
    學生課程完成率預測器

    使用訓練好的決策樹和邏輯回歸模型進行預測
    """

    def __init__(self, models_dir: str = "models", output_dir: str = "output"):
        """
        初始化預測器

        參數:
            models_dir: 模型檔案目錄
            output_dir: 輸出檔案目錄（包含特徵轉換規則）
        """
        self.models_dir = models_dir
        self.output_dir = output_dir

        # 載入模型和元資訊
        self._load_models()
        self._load_binning_rules()
        self._load_optimal_thresholds()

    def _load_models(self):
        """載入所有模型和元資訊"""

        print("載入模型...")

        # 載入決策樹模型
        dt_model_path = f"{self.models_dir}/decision_tree_model.pkl"
        dt_metadata_path = f"{self.models_dir}/decision_tree_metadata.pkl"

        if not Path(dt_model_path).exists():
            raise FileNotFoundError(f"找不到決策樹模型: {dt_model_path}")

        self.dt_model = joblib.load(dt_model_path)
        self.dt_metadata = joblib.load(dt_metadata_path)
        self.threshold = self.dt_metadata['threshold']

        print(f"   ✓ 決策樹模型（閾值: {self.threshold:.4f}）")

        # 載入邏輯回歸模型
        high_model_path = f"{self.models_dir}/logistic_regression_high_group.pkl"
        low_model_path = f"{self.models_dir}/logistic_regression_low_group.pkl"
        lr_metadata_path = f"{self.models_dir}/logistic_regression_metadata.pkl"

        if not Path(high_model_path).exists():
            raise FileNotFoundError(f"找不到 High Group 模型: {high_model_path}")

        self.lr_high_model = joblib.load(high_model_path)
        self.lr_low_model = joblib.load(low_model_path)
        self.lr_metadata = joblib.load(lr_metadata_path)

        print(f"   ✓ 邏輯回歸模型（High Group & Low Group）")

    def _load_binning_rules(self):
        """載入特徵轉換規則"""

        print("載入特徵轉換規則...")

        binning_rules_path = f"{self.output_dir}/3_continuous_bins_rules.xlsx"

        if not Path(binning_rules_path).exists():
            raise FileNotFoundError(f"找不到分段規則: {binning_rules_path}")

        self.binning_rules = pd.read_excel(binning_rules_path)

        print(f"   ✓ 連續變數分段規則（{len(self.binning_rules)} 條）")

    def _load_optimal_thresholds(self):
        """載入訓練時找到的最佳閾值"""

        print("載入最佳閾值...")

        eval_path = f"{self.output_dir}/4.5_model_evaluation.xlsx"

        if not Path(eval_path).exists():
            print(f"   ⚠ 找不到評估結果，將使用預設閾值 (0.5)")
            self.optimal_thresholds = None
            return

        eval_df = pd.read_excel(eval_path)

        # 提取 High Group 閾值
        high_group = eval_df[eval_df['Group'] == 'High Group']
        high_thresholds = {
            'default': high_group[high_group['Threshold_Type'] == 'default']['Threshold'].iloc[0],
            'best_f1': high_group[high_group['Threshold_Type'] == 'best_f1']['Threshold'].iloc[0],
            'best_f2': high_group[high_group['Threshold_Type'] == 'best_f2']['Threshold'].iloc[0]
        }

        # 提取 Low Group 閾值
        low_group = eval_df[eval_df['Group'] == 'Low Group']
        low_thresholds = {
            'default': low_group[low_group['Threshold_Type'] == 'default']['Threshold'].iloc[0],
            'best_f1': low_group[low_group['Threshold_Type'] == 'best_f1']['Threshold'].iloc[0],
            'best_f2': low_group[low_group['Threshold_Type'] == 'best_f2']['Threshold'].iloc[0]
        }

        self.optimal_thresholds = {
            'high': high_thresholds,
            'low': low_thresholds
        }

        print(f"   ✓ 最佳閾值（High: F1={high_thresholds['best_f1']:.2f}, F2={high_thresholds['best_f2']:.2f}; "
              f"Low: F1={low_thresholds['best_f1']:.2f}, F2={low_thresholds['best_f2']:.2f}）")

    def preprocess_new_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        預處理新資料（與訓練時相同的邏輯）

        參數:
            df: 新資料 DataFrame（需包含與訓練資料相同的欄位）

        回傳:
            預處理後的 DataFrame
        """

        df = df.copy()

        # 移除不需要的欄位（如果存在）
        drop_columns = ['CustID', 'ServiceStartDate', 'WeeksWithService', 'Year', 'Month']
        existing_drop_cols = [col for col in drop_columns if col in df.columns]
        if existing_drop_cols:
            df = df.drop(columns=existing_drop_cols)

        # 處理缺失值（與訓練時相同的策略）
        if 'PaymentMethod' in df.columns and df['PaymentMethod'].isna().any():
            df['PaymentMethod'] = df['PaymentMethod'].fillna('Other')

        if 'Gender' in df.columns and df['Gender'].isna().any():
            df['Gender'] = df['Gender'].fillna('F')

        return df

    def _apply_decision_tree_ohe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        對資料進行 One-Hot Encoding（與決策樹訓練時相同）

        參數:
            df: 預處理後的資料

        回傳:
            OHE 後的資料
        """

        # 識別類別變數
        categorical_features = self.dt_metadata['categorical_features']

        # One-Hot Encoding
        df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=False)

        # 對齊特徵欄位（確保與訓練時一致）
        feature_columns = self.dt_metadata['feature_columns']

        # 補充缺少的欄位（填 0）
        for col in feature_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0

        # 只保留訓練時使用的欄位，並按照相同順序
        df_encoded = df_encoded[feature_columns]

        return df_encoded

    def _split_by_decision_tree(self, df: pd.DataFrame, df_encoded: pd.DataFrame) -> tuple:
        """
        使用決策樹模型分群

        參數:
            df: 原始資料
            df_encoded: OHE 後的資料

        回傳:
            (df_high, df_low, high_indices, low_indices)
        """

        # 預測機率
        predict_proba = self.dt_model.predict_proba(df_encoded)[:, 1]

        # 加入預測機率
        df_with_proba = df.copy()
        df_with_proba['PredictedProba_Complete'] = predict_proba

        # 根據閾值分群
        high_mask = predict_proba >= self.threshold
        low_mask = predict_proba < self.threshold

        df_high = df_with_proba[high_mask].copy()
        df_low = df_with_proba[low_mask].copy()

        high_indices = df_with_proba.index[high_mask].tolist()
        low_indices = df_with_proba.index[low_mask].tolist()

        return df_high, df_low, high_indices, low_indices

    def _transform_features_to_binary(self, df: pd.DataFrame, group: str) -> pd.DataFrame:
        """
        將特徵轉換為二元變數（使用保存的分段規則）

        參數:
            df: 分群後的資料
            group: 'High Group' 或 'Low Group'

        回傳:
            轉換後的 DataFrame
        """

        # 移除 PredictedProba_Complete 欄位
        df = df.drop(columns=['PredictedProba_Complete'], errors='ignore')

        # 識別特徵類型
        from src.feature_transform import identify_feature_types

        continuous_features, categorical_features = identify_feature_types(df, None)

        # 創建轉換後的 DataFrame
        df_transformed = pd.DataFrame(index=df.index)

        # 處理連續變數 - 使用保存的分段規則
        group_rules = self.binning_rules[self.binning_rules['Group'] == group]

        for feature in continuous_features:
            # 獲取該特徵的分段規則
            feature_rules = group_rules[group_rules['Feature'] == feature]

            if not feature_rules.empty:
                # 解析切點
                split_points_str = feature_rules.iloc[0]['SplitPoints']
                if pd.notna(split_points_str):
                    split_points = [float(x) for x in split_points_str.split(', ')]

                    # 應用分段
                    bins = [-np.inf] + split_points + [np.inf]
                    bin_labels = [f'Bin_{i+1}' for i in range(len(bins) - 1)]

                    binned = pd.cut(df[feature], bins=bins, labels=bin_labels, include_lowest=True)

                    # One-Hot Encoding (drop_first=True)
                    binned_df = pd.get_dummies(binned, prefix=feature, drop_first=True, dtype=int)
                    df_transformed = pd.concat([df_transformed, binned_df], axis=1)
                else:
                    # 如果沒有切點，使用中位數切分（fallback）
                    median_val = df[feature].median()
                    df_transformed[f"{feature}_AboveMedian"] = (df[feature] >= median_val).astype(int)
            else:
                # 如果找不到規則，使用中位數切分（fallback）
                median_val = df[feature].median()
                df_transformed[f"{feature}_AboveMedian"] = (df[feature] >= median_val).astype(int)

        # 處理類別變數 - One-Hot Encoding (drop_first=True)
        if categorical_features:
            for feature in categorical_features:
                # One-Hot Encoding
                feature_dummies = pd.get_dummies(df[feature], prefix=feature, drop_first=True, dtype=int)
                df_transformed = pd.concat([df_transformed, feature_dummies], axis=1)

        return df_transformed

    def predict(self, df: pd.DataFrame,
                threshold: str | float | dict = 'best_f1',
                return_details: bool = False) -> pd.DataFrame:
        """
        對新資料進行預測

        參數:
            df: 新資料 DataFrame
            threshold: 分類閾值，支援以下格式：
                - 'default': 使用 0.5（兩個群組相同）
                - 'best_f1': 使用訓練時找到的最佳 F1 閾值（推薦）
                - 'best_f2': 使用訓練時找到的最佳 F2 閾值（優先召回率）
                - float: 統一閾值（兩個群組使用相同閾值）
                - dict: 分別指定，如 {'high': 0.26, 'low': 0.11}
            return_details: 是否返回詳細資訊（群組、預測機率等）

        回傳:
            包含預測結果的 DataFrame
        """

        print("\n" + "=" * 80)
        print("學生課程完成率預測")
        print("=" * 80)

        # 解析閾值
        if isinstance(threshold, str):
            if self.optimal_thresholds is None:
                print(f"   ⚠ 無法載入最佳閾值，使用預設值 0.5")
                high_threshold = 0.5
                low_threshold = 0.5
            elif threshold in ['default', 'best_f1', 'best_f2']:
                high_threshold = self.optimal_thresholds['high'][threshold]
                low_threshold = self.optimal_thresholds['low'][threshold]
                print(f"   ✓ 使用 {threshold} 閾值（High: {high_threshold:.2f}, Low: {low_threshold:.2f}）")
            else:
                raise ValueError(f"不支援的閾值類型: {threshold}。請使用 'default', 'best_f1', 'best_f2', 數值, 或字典")
        elif isinstance(threshold, dict):
            high_threshold = threshold.get('high', 0.5)
            low_threshold = threshold.get('low', 0.5)
            print(f"   ✓ 使用自訂閾值（High: {high_threshold:.2f}, Low: {low_threshold:.2f}）")
        else:
            # 數值型閾值，兩個群組使用相同值
            high_threshold = float(threshold)
            low_threshold = float(threshold)
            print(f"   ✓ 使用統一閾值: {threshold:.2f}")

        print(f"\n[1/5] 預處理資料")
        df_preprocessed = self.preprocess_new_data(df)
        print(f"   - 資料筆數: {len(df_preprocessed)}")

        print(f"\n[2/5] 決策樹分群")
        df_encoded = self._apply_decision_tree_ohe(df_preprocessed)
        df_high, df_low, high_indices, low_indices = self._split_by_decision_tree(
            df_preprocessed, df_encoded
        )
        print(f"   - High Group: {len(df_high)} 筆 ({len(df_high)/len(df_preprocessed)*100:.1f}%)")
        print(f"   - Low Group: {len(df_low)} 筆 ({len(df_low)/len(df_preprocessed)*100:.1f}%)")

        # 初始化結果 DataFrame
        results = pd.DataFrame(index=df.index)
        results['Group'] = None
        results['DT_PredictedProba'] = None
        results['LR_PredictedProba'] = None
        results['Prediction'] = None

        print(f"\n[3/5] 特徵轉換")

        # High Group 預測
        if len(df_high) > 0:
            print(f"   - 轉換 High Group 特徵")
            df_high_transformed = self._transform_features_to_binary(df_high, 'High Group')

            # 對齊特徵欄位
            high_features = self.lr_metadata['high_group_features']
            for col in high_features:
                if col not in df_high_transformed.columns:
                    df_high_transformed[col] = 0
            df_high_transformed = df_high_transformed[high_features]

            print(f"\n[4/5] 邏輯回歸預測 - High Group（閾值: {high_threshold:.2f}）")
            high_proba = self.lr_high_model.predict_proba(df_high_transformed)[:, 1]
            high_pred = (high_proba >= high_threshold).astype(int)

            # 填入結果
            for i, idx in enumerate(high_indices):
                results.loc[idx, 'Group'] = 'High'
                results.loc[idx, 'DT_PredictedProba'] = df_high.loc[df_high.index[i], 'PredictedProba_Complete']
                results.loc[idx, 'LR_PredictedProba'] = high_proba[i]
                results.loc[idx, 'Prediction'] = high_pred[i]

        # Low Group 預測
        if len(df_low) > 0:
            print(f"   - 轉換 Low Group 特徵")
            df_low_transformed = self._transform_features_to_binary(df_low, 'Low Group')

            # 對齊特徵欄位
            low_features = self.lr_metadata['low_group_features']
            for col in low_features:
                if col not in df_low_transformed.columns:
                    df_low_transformed[col] = 0
            df_low_transformed = df_low_transformed[low_features]

            print(f"\n[4/5] 邏輯回歸預測 - Low Group（閾值: {low_threshold:.2f}）")
            low_proba = self.lr_low_model.predict_proba(df_low_transformed)[:, 1]
            low_pred = (low_proba >= low_threshold).astype(int)

            # 填入結果
            for i, idx in enumerate(low_indices):
                results.loc[idx, 'Group'] = 'Low'
                results.loc[idx, 'DT_PredictedProba'] = df_low.loc[df_low.index[i], 'PredictedProba_Complete']
                results.loc[idx, 'LR_PredictedProba'] = low_proba[i]
                results.loc[idx, 'Prediction'] = low_pred[i]

        print(f"\n[5/5] 預測完成")
        print(f"   - Complete 預測: {results['Prediction'].sum()} 人 ({results['Prediction'].mean()*100:.1f}%)")
        print(f"   - Quit 預測: {(results['Prediction'] == 0).sum()} 人 ({(results['Prediction'] == 0).mean()*100:.1f}%)")

        print("\n" + "=" * 80)

        if return_details:
            return results
        else:
            return results[['Prediction']]


def predict_from_file(input_path: str,
                      output_path: str = None,
                      threshold: str | float | dict = 'best_f1'):
    """
    從檔案讀取資料並預測

    參數:
        input_path: 輸入檔案路徑（Excel 或 CSV）
        output_path: 輸出檔案路徑（可選，預設為 predictions/predictions_{timestamp}.xlsx）
        threshold: 分類閾值（參見 StudentCompletionPredictor.predict 的說明）
    """

    print(f"讀取資料: {input_path}")

    # 讀取資料
    if input_path.endswith('.csv'):
        df = pd.read_csv(input_path)
    else:
        df = pd.read_excel(input_path)

    print(f"資料筆數: {len(df)}")

    # 初始化預測器
    predictor = StudentCompletionPredictor()

    # 預測
    results = predictor.predict(df, threshold=threshold, return_details=True)

    # 合併原始資料和預測結果
    df_with_predictions = pd.concat([df, results], axis=1)

    # 儲存結果
    if output_path is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"predictions/predictions_{timestamp}.xlsx"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_with_predictions.to_excel(output_path, index=False, engine='openpyxl')

    print(f"\n已儲存預測結果: {output_path}")

    return df_with_predictions


if __name__ == "__main__":
    # 範例：使用預處理後的資料進行測試
    # 實際使用時應該用新的、未見過的資料

    input_file = "output/1_preprocessed_data.xlsx"
    output_file = "predictions/test_predictions.xlsx"

    print("測試預測功能（使用訓練資料的前 100 筆）")

    # 讀取資料
    df = pd.read_excel(input_file)

    # 取前 100 筆作為測試
    df_test = df.head(100).copy()

    # 移除 ServiceStatus 欄位（模擬新資料）
    if 'ServiceStatus' in df_test.columns:
        actual_status = df_test['ServiceStatus'].copy()
        df_test = df_test.drop(columns=['ServiceStatus'])

    # 預測
    results = predict_from_file(input_file, output_file)

    print("\n預測結果摘要:")
    print(results[['Group', 'DT_PredictedProba', 'LR_PredictedProba', 'Prediction']].head(10))
