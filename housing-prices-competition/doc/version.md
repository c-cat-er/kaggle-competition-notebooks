## 1.ipynb (引進 OOF 與基礎融合)

- 用 10-fold 交叉驗證並收集 OOF 預測值
- 結果：
    - 10-fold OOF RMSLE：XGB 跑出 0.12236，Lasso/Ridge 落在 0.13836
    - 用固定權重把這幾隻融合起來，分數推進到 0.12009，驗證了 Ensemble 方向是對的

## 2.ipynb (12691.94) ➔ 3.ipynb (12499.26)

- 修正早停洩漏（Early Stopping Leakage）：
    - 問題：舊版 eval_set 直接餵 Validation Fold，導致樹模型在訓練時等於偷看了驗證集，以致 OOF 分數虛高、偏樂觀
    - 解法：將 train_test_split 包進 CV 內圈，改用內圈的 validation 做早停，亦可直接改用固定 n_estimators
- 融合策略升級（Blending ➔ Stacking）：
    - 問題：舊版用 scipy.optimize 的 SLSQP 找固定權重，但因權重是在整份 OOF 上全局最佳化，容易 Overfitting
    - 解法：改用更強健的 Stacking：直接把第一層 5 個模型 OOF 預測值當作特徵，塞給第二層 Meta-Model（用 RidgeCV / ElasticNetCV）去自動學權重，並搭配 10×5 seeds 平均
- Multiple Seed Averages (防過擬合加強)：讓同一模型跑 5~10 個不同的 random_state 再取平均，用時間換分數，能更穩 LB 降
- 補齊 Ordinal 編碼：
    - 舊版只編了幾欄 Quality 欄位，其餘通通丟 One-Hot，導致欄位爆增且樹模型很難抓規律。這次對照 data_description.txt 的順序，把 BsmtExposure, BsmtFinType1/2, GarageFinish, Functional, Fence, CentralAir, PavedDrive, LotShape, LandSlope, Utilities 完整重寫成 OrdinalEncoder

## 3.ipynb (12499.26) ➔ 4.ipynb

- 特徵工程再加強：
    - 進階補值：LotFrontage 欄位原本用全體中位數補，太粗糙。改用 Neighborhood 的分群中位數補，更符合真實房價邏輯
    - 稀疏欄位 flag：針對一堆 0 的 Zero-inflated 欄位（如 PoolArea, WoodDeckSF 等），直接多衍生一欄 HasXxx 的 0/1 二元特徵
    - 安全目標編碼：對 Neighborhood 這種高基數分類欄位做 Target Encoding，但須嚴格在 fold 內 fit，否則一定洩漏
    - 組合與分箱：補上 OverallQual × OverallCond 這種直覺的狀態組合特徵，計算居住面積比率（GrLivArea / TotalSF），並對 YearBuilt 做分箱
    - 偏態轉換換代：特徵修正不再用死板的 log1p(max(0,·))，全面改成 Yeo-Johnson 轉換，能自動尋找最佳參數且更完美處理 0 或負值
- 擴充模型多樣性：
    - 加入 HistGradientBoostingRegressor、KernelRidge(RBF)、SVR、MLP 提高模型多樣性
    - 線性模型端要特別注意，強共線性的欄位（如 1stFlrSF vs TotalBsmtSF）在餵給線性模型（Linear Model）之前要先砍掉一輪，避免干擾線性迴歸
- 自動化超參調優：引入 Optuna，針對 XGB/LGB/Cat 這三個核心主力各跑 30~50 代的 tuning（如鎖定 LGB 的 num_leaves 15~63、XGB 的 max_depth 3~6，Cat 關掉 verbose）。搭配 10 折交叉驗證重複跑 2 次，確保超參點足夠強健
- 最後預測輸出前，強制用 np.clip(preds, train_min, train_max) 把超出合理範圍的極端外推值剪掉
- 目標是把真正的 CV 壓到 0.113 以下，並嚴格盯緊 CV 跟 LB 之間的落差（Gap），不再掉進過擬合陷阱
