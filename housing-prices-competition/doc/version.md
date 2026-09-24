## 1.ipynb

- 只有 holdout MAE（RF 17,350 / XGB 17,328）、CV MAE ≈ 0.09
- 用 MAE 而非比賽的 RMSLE，無 OOF、無模型融合 → 最不正確、分數最低

## 2.ipynb

- 10-fold OOF RMSLE：XGB 0.12236、Lasso/Ridge 0.13836，固定權重融合 0.12009
- 流程完整、可從頭跑完，預測值合理（mean ≈ 177k）→ 最正確

## 3_kaggle_12691.94159 -> 4_kaggle_12499.26636.ipynb

1. 先修早停泄漏：eval_set 用驗證折 → OOF 偏樂觀。改成 eval_set=[(X_train_trans, y_train)] 或固定 n_estimators。這是「CV 好、LB 未必好」的主因。
2. 融合改 Stacking：目前 5 個手動權重（且權重是在整份 OOF 上最佳化，會過擬合）。改成 RidgeCV/ElasticNetCV 當 meta-model 吃 OOF，並用 10×5 seeds 平均。
3. 多種子平均：同一模型跑 5~10 個 random_state 取平均，通常直接降 0.002~0.005。
4. 補 Ordinal 編碼：現在只編品質欄。把 BsmtExposure, BsmtFinType1/2, GarageFinish, Functional, Fence, CentralAir, PavedDrive, LotShape, LandSlope, Utilities 依 data_description 順序改成 OrdinalEncoder（別用 one-hot）。

## 4_kaggle_12499.26636 -> 5_kaggle_572081.48834.ipynb

1. 特徵再加強：

- LotFrontage 用「同 Neighborhood 中位數」補值（比全體中位數好）
- zero-inflated 欄位（PoolArea/WoodDeckSF/...）加 HasXxx 0/1 flag
- Neighborhood 做 CV-safe 目標編碼（fold 內 fit）
- OverallQual × OverallCond、GrLivArea/TotalSF 比率、YearBuilt 分箱
- 偏態修正改用 Yeo-Johnson（可處理 0／負值），比現在 log1p(max(0,·)) 好

2. 換/加模型：HistGradientBoostingRegressor、KernelRidge(RBF)、SVR、MLP；線性端把共線欄位先砍一輪（1stFlrSF vs TotalBsmtSF）。
3. 超參調優：Optuna 對 XGB/LGB/Cat 各跑 30~50 trials（LGB num_leaves 15~63、XGB max_depth 3~6、Cat 用 RMSE+verbose=0），搭配 10 折 repeat=2。
4. 收尾：預測值 np.clip(preds, train_min, train_max)；CV 從 0.1066 壓到 0.113 以下再上傳（同時留意 CV/LB gap）。
