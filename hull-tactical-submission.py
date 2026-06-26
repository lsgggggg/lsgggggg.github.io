# Kaggle — Hull Tactical: Market Prediction
# 推理 / 提交脚本：LightGBM + XGBoost + CatBoost + Ridge 四模型加权集成，
# 配合滚动流式推理与动态波动率仓位策略（120% 波动率约束下控制杠杆）。
# 个人主页附件：https://lsgggggg.github.io/
# Author: Li Shiguang (github.com/lsgggggg)

"""
Kaggle提交脚本
修复：
1. 正确处理is_scored=False的情况（保持历史连续性）
2. 确保使用预测的forward_returns进行滚动预测
3. 动态市场波动率调整（而非固定15%）
"""

import os
import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
import joblib
import torch
import sys
sys.path.append('/kaggle/input/hull-tactical-dataset/')
import data_processor
import config
import models
import train
import metrics

import kaggle_evaluation.default_inference_server

# ============ GPU检测和打印 ============
USE_GPU = torch.cuda.is_available()
if USE_GPU:
    print(f"=" * 60)
    print(f"GPU DETECTED IN INFERENCE:")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"=" * 60)
else:
    print("=" * 60)
    print("WARNING: No GPU detected in inference, using CPU")
    print("=" * 60)

# 配置：选择使用哪个模型
USE_FINAL_MODEL = True
MODEL_PATH = '/kaggle/input/hull-tactical-dataset/saved_models/final/' if USE_FINAL_MODEL else '/kaggle/input/hull-models/cv/'

MODEL_WEIGHTS = {
    'lightgbm': 0.40,
    'xgboost': 0.30,
    'catboost': 0.20,
    'ridge': 0.10
}

class Predictor:
    def __init__(self):
        self.processor_data = None
        self.feature_engineer = None
        self.final_fitter = None
        self.models = {}
        self.is_loaded = False
        self.history = []
        self.use_final = USE_FINAL_MODEL

        # 【改进】追踪策略收益和市场收益（用于动态波动率调整）
        self.recent_strategy_returns = []
        self.recent_market_returns = []  # 新增：追踪市场收益

    def load(self):
        if self.is_loaded:
            return

        print(f"Loading models from {MODEL_PATH}...")
        print(f"GPU Available: {USE_GPU}")

        # 加载处理器
        processor_data = joblib.load(f'{MODEL_PATH}processor.pkl')
        self.feature_engineer = processor_data['feature_engineer']
        self.final_fitter = processor_data['final_fitter']

        # 加载元信息
        meta = joblib.load(f'{MODEL_PATH}meta.pkl')
        print(f"Models trained in {meta['training_mode']} mode")

        # 根据模式加载模型
        if self.use_final:
            model_files = {
                'lightgbm': 'lightgbm_final.txt',
                'xgboost': 'xgboost_final.json',
                'catboost': 'catboost_final.cbm',
                'ridge': 'ridge_final.pkl'
            }
        else:
            last_fold = len(meta.get('cv_scores', [])) - 1
            model_files = {
                'lightgbm': f'lightgbm_fold{last_fold}.txt',
                'xgboost': f'xgboost_fold{last_fold}.json',
                'catboost': f'catboost_fold{last_fold}.cbm',
                'ridge': f'ridge_fold{last_fold}.pkl'
            }

        # 加载各模型
        for model_type, filename in model_files.items():
            filepath = f'{MODEL_PATH}{filename}'
            if os.path.exists(filepath):
                if model_type == 'lightgbm':
                    self.models[model_type] = lgb.Booster(model_file=filepath)
                    print(f"  ✓ Loaded LightGBM (CPU mode)")
                elif model_type == 'xgboost':
                    self.models[model_type] = xgb.Booster()
                    self.models[model_type].load_model(filepath)
                    print(f"  ✓ Loaded XGBoost (GPU: {USE_GPU})")
                elif model_type == 'catboost':
                    self.models[model_type] = CatBoostRegressor()
                    self.models[model_type].load_model(filepath)
                    print(f"  ✓ Loaded CatBoost (GPU: {USE_GPU})")
                elif model_type == 'ridge':
                    self.models[model_type] = joblib.load(filepath)
                    print(f"  ✓ Loaded Ridge (CPU only)")

        self.is_loaded = True
        print(f"Loaded {len(self.models)} models successfully")

    def predict_single(self, test_row):
        """
        预测单条数据

        关键修复：
        1. 无论is_scored与否，都要更新历史（保持连续性）
        2. 使用上一次预测的forward_returns更新lagged_forward_returns
        3. 【改进】动态市场波动率调整（而非固定15%）
        """
        if not self.is_loaded:
            self.load()

        # 转换格式
        if isinstance(test_row, pl.DataFrame):
            current = test_row.to_pandas()
        else:
            current = test_row.copy()

        # 【修复1】先检查is_scored，但仍要处理历史
        is_scored = True
        if 'is_scored' in current.columns:
            is_scored = current['is_scored'].iloc[0] if len(current) > 0 else True

        # 【修复2】无论is_scored与否，都要更新历史（保持连续性）
        # 如果这不是第一条数据，使用上一次的预测结果
        if len(self.history) > 0:
            last_history = self.history[-1].copy()

            # 如果上一条记录有我们预测的forward_returns，使用它更新lagged_forward_returns
            if 'predicted_forward_returns' in last_history.columns:
                current['lagged_forward_returns'] = last_history['predicted_forward_returns'].values[0]

        # 【改进】记录市场收益（从lagged_forward_returns）
        if 'lagged_forward_returns' in current.columns:
            market_return = current['lagged_forward_returns'].iloc[0]
            if not pd.isna(market_return):
                self.recent_market_returns.append(market_return)
                if len(self.recent_market_returns) > 100:
                    self.recent_market_returns = self.recent_market_returns[-100:]

        # 添加到历史（无论is_scored与否）
        MAX_HISTORY = 400
        self.history.append(current.copy())
        if len(self.history) > MAX_HISTORY:
            self.history = self.history[-MAX_HISTORY:]

        # 【修复3】如果is_scored=False，快速返回默认值（但已更新历史）
        if not is_scored:
            # 仍然记录一个"虚拟预测"（使用简单策略）
            dummy_pred = 0.0  # 对应position=1.0（持有现金）
            self.history[-1]['predicted_forward_returns'] = dummy_pred
            return 1.0

        # ===== 正常预测流程 =====
        # 构建完整DataFrame（包含所有历史）
        full_df = pd.concat(self.history, ignore_index=True)

        # 使用feature_engineer创建基础特征
        # 注意：create_lag_rolling_features会为lagged_forward_returns创建滚动特征
        features = self.feature_engineer.create_lag_rolling_features(full_df)

        # 取最后一行（当前要预测的行）
        current_features = features.iloc[-1:].copy()

        # 应用final_fitter的转换
        if self.final_fitter:
            # GroupBy特征
            current_features = self.final_fitter.transform_groupby_features(
                full_df.iloc[-1:], current_features
            )

            # PCA特征
            current_features = self.final_fitter.transform_pca_features(
                full_df.iloc[-1:], current_features
            )

            # 确保没有NaN
            current_features = current_features.fillna(0)

            # 选择特征
            if self.final_fitter.selected_features:
                for feat in self.final_fitter.selected_features:
                    if feat not in current_features.columns:
                        current_features[feat] = 0
                current_features = current_features[self.final_fitter.selected_features]

            # 标准化
            if self.final_fitter.scaler:
                current_features = pd.DataFrame(
                    self.final_fitter.scaler.transform(current_features.fillna(0)),
                    columns=current_features.columns
                )

        # 预测
        predictions = []

        if 'lightgbm' in self.models:
            pred = self.models['lightgbm'].predict(current_features)[0]
            predictions.append(('lightgbm', pred))

        if 'xgboost' in self.models:
            pred = self.models['xgboost'].predict(xgb.DMatrix(current_features))[0]
            predictions.append(('xgboost', pred))

        if 'catboost' in self.models:
            pred = self.models['catboost'].predict(current_features)[0]
            predictions.append(('catboost', pred))

        if 'ridge' in self.models:
            pred = self.models['ridge'].predict(current_features)[0]
            predictions.append(('ridge', pred))

        # 【可选】异常值检查
        pred_values = [pred for _, pred in predictions]
        if len(pred_values) > 1:
            pred_std = np.std(pred_values)
            # 如果模型预测差异过大，可能有问题
            if pred_std > 0.05:  # 阈值可调
                # 记录警告（但不中断预测）
                pass

        # 加权集成
        ensemble_pred = sum(
            MODEL_WEIGHTS.get(name, 0) * pred
            for name, pred in predictions
        )

        # 【修复4】保存这次预测的forward_returns到最后一条历史记录中
        # 这样下次预测时可以使用（保持连续性）
        self.history[-1]['predicted_forward_returns'] = ensemble_pred

        # ===== 【改进】动态市场波动率调整 =====
        # 计算基础仓位
        base_position = 1.0 + ensemble_pred * 400

        # 如果有足够历史，检查波动率
        if len(self.recent_strategy_returns) > 20 and len(self.recent_market_returns) > 20:
            # 近20天策略波动率（年化）
            recent_vol = np.std(self.recent_strategy_returns[-20:]) * np.sqrt(252)

            # 【改进】使用实际市场波动率（而非假设的15%）
            market_vol = np.std(self.recent_market_returns[-20:]) * np.sqrt(252)

            # 避免除零
            if market_vol < 0.01:
                market_vol = 0.15  # 回退到默认值

            vol_ratio = recent_vol / market_vol

            # 如果接近或超过120%限制，降低杠杆
            if vol_ratio > 1.1:
                scale_factor = 1.1 / vol_ratio
                adjusted_position = 1.0 + (base_position - 1.0) * scale_factor
            else:
                adjusted_position = base_position
        else:
            adjusted_position = base_position

        # Clip到允许范围
        position = np.clip(adjusted_position, 0, 2)

        # 记录策略收益（用于下次计算波动率）
        # 简化：用预测的收益估算策略收益
        strategy_return = (position - 1) * ensemble_pred
        self.recent_strategy_returns.append(strategy_return)
        if len(self.recent_strategy_returns) > 100:
            self.recent_strategy_returns = self.recent_strategy_returns[-100:]

        return float(position)

# 创建预测器
predictor = Predictor()

def predict(test: pl.DataFrame) -> float:
    """Kaggle API要求的预测函数"""
    return predictor.predict_single(test)

# 启动服务器
inference_server = kaggle_evaluation.default_inference_server.DefaultInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(('/kaggle/input/hull-tactical-market-prediction/',))
