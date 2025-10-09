# 7114056076_hw2 —— 多元線性回歸（CRISP-DM）專案報告
> 版本：2025-10-09  
> 課程作業：Multiple Linear Regression（含 **特徵選擇**、**模型評估**、**預測信賴/預測區間**）

---

## 0) 專案總覽
本專案以 **Kaggle Wine Quality（紅酒）** 數據集為例，依 **CRISP-DM** 流程完成：需求界定 → 資料理解 → 資料準備 → 建模 → 評估 → 部署建議。  
模型主軸為 **多元線性回歸**，搭配 **LassoCV** 進行特徵選擇；最終以 **statsmodels OLS** 產出 **95% 預測區間** 與 **信賴區間**，並提供殘差診斷圖。

- **資料來源**：Wine Quality（Red），11 項理化特徵 + 1 目標欄位 `quality`
- **目標**：依理化特徵預測葡萄酒品質分數（0–10）
- **輸出成果**：  
  - `outputs/predictions_with_PI.png`（測試集預測均值 + 95% 預測區間）  
  - `outputs/residuals_vs_fitted.png`、`outputs/residuals_qq.png`（殘差診斷圖）  
  - `outputs/SLR_alcohol_with_CI.png`（單變數線回歸＋95% 信賴區間）  
  - `outputs/metrics.csv`（R²、MAE、RMSE、Baseline CV-RMSE）

---

## 1) Business Understanding
- **產業/研究需求**：協助釀造端在批次生產前預測品質，支援配方調整與品控。
- **使用情境**：  
  1. **產線即時監控**：侦測預測落差並追蹤異常理化指標。  
  2. **實驗室配方優化**：模擬改變理化特徵對品質分數的影響。
- **衡量指標（KPI）**：  
  - **準確度**：提升 R²，降低 RMSE/MAE；與常數基準相比需有顯著改善。  
  - **可解釋性**：模型係數能對應實務理解（如酒精、揮發性酸對品質的影響）。  
  - **不確定性揭露**：提供預測區間，協助風險溝通。

---

## 2) Data Understanding
- **來源與規模**：Kaggle Red Wine Quality 資料（1143 筆、12 欄，含 `quality` 目標值）。原始檔名 `WineQT.csv`。  
- **特徵描述**：固定酸度、揮發性酸、檸檬酸、殘糖、氯化物、自由/總二氧化硫、密度、pH、硫酸鹽、酒精，以及識別欄 `Id`。  
- **EDA 摘要**：  
  - 無缺值，經 `drop_duplicates()` 後保留 1143 筆。  
  - `alcohol` 與 `quality` 呈現正向關係；`volatile acidity`、`chlorides` 與品質呈現負相關。  
  - 特徵尺度差異大，需標準化以利線性模型穩定收斂。

---

## 3) Data Preparation
- **清理**：移除重複列，若存在 `type` 欄位則轉為 0/1 編碼（但本資料僅紅酒，因此未出現）。  
- **資料分割**：`train_test_split(test_size=0.2, random_state=42)`。  
- **特徵工程**：  
  - **標準化**：在 `Pipeline` 中串接 `StandardScaler` 與 `LinearRegression`。  
  - **特徵選擇**：使用 `LassoCV(cv=5)`，篩選非零係數特徵；若全部被壓到 0，則退回與 `quality` 絕對相關係數最高的 Top-K 特徵。

---

## 4) Modeling
- **Baseline 模型**：`Pipeline(StandardScaler, LinearRegression)`，以 5-fold 交叉驗證估計 RMSE。  
- **特徵選擇結果**：LassoCV 選出 8 個特徵：`volatile acidity`、`chlorides`、`total sulfur dioxide`、`density`、`pH`、`sulphates`、`alcohol`、`Id`。  
- **最終模型**：以篩選後特徵餵入 `statsmodels.OLS`，取得完整係數、信賴區間、p-value 與診斷統計量。  
- **輔助模型**：以 `alcohol` 建立簡單線性回歸，輸出 95% 信賴區間曲線供直觀說明。

---

## 5) Evaluation
> 下列指標均透過 `python notebook_code_dump.py` 腳本執行後計算與輸出。

- **測試集表現**：  
  - R²：`0.327`  
  - MAE：`0.477`  
  - RMSE：`0.612`  
  - Baseline CV-RMSE（5-fold）：`0.645`
- **視覺化重點**：  
  1. **預測 vs 實際（含 95% 預測區間）**：`outputs/predictions_with_PI.png`。預測均值軌跡緊貼實際值，區間寬度約一個品質分數單位，反映殘差變異。  
  2. **殘差 vs 預測值**：`outputs/residuals_vs_fitted.png`。殘差大致對稱於 0，未見明顯漏斗型，但高密度區略呈負殘差。  
  3. **Q-Q Plot**：`outputs/residuals_qq.png`。中段貼近對角線，尾端輕微偏離，顯示常態假設大致成立但存在少量極端值。
- **討論**：  
  - R² 約 0.33，表示線性模型僅解釋三成變異；品質仍受非線性、交互作用或感官因素影響。  
  - 係數顯示酒精濃度、硫酸鹽正向影響品質，揮發性酸、氯化物負向影響符合釀造經驗。  
  - 殘差常態性尚可，但 Q-Q 尾端偏離提示可考慮更複雜模型或穩健回歸。  
  - Baseline CV-RMSE（0.645）與測試 RMSE（0.612）接近，表示訓練過程無明顯過擬合。

---

## 6) Deployment（落地建議）
- **批次預測**：封裝管線（Scaler + Lasso 選特徵 + OLS 係數）為腳本，定期讀取檢測資料並輸出預測值、區間。  
- **監控儀表板**：將 `outputs/metrics.csv` 與殘差圖上傳至內部儀表板，監測指標趨勢。  
- **風險控管**：當 RMSE 超過 0.7 或連續批次落在預測下界外時，觸發品控警示並回溯理化指標。  
- **資料漂移**：定期檢查特徵分佈、Durbin-Watson、VIF；若顯著變動則重新訓練模型。

---

## 7) 方法比較與優劣分析
- **特徵處理**：主流做法為標準化與 L1/L2 規則化（Lasso、ElasticNet）。本案採 LassoCV，兼顧稀疏性與解釋力。  
- **評估策略**：多數作法使用 KFold 交叉驗證估計泛化誤差；目前 5-fold 足以衡量穩定性。  
- **替代模型**：若需求偏向更高準確度，可考慮樹模型（RandomForest、XGBoost、LightGBM）或互動項、非線性特徵，但需補上區間估計。  
- **不確定性呈現**：傳統線性回歸易提供信賴與預測區間；樹模型需額外方法（如 Quantile Regression Forests）才能輸出區間。

---

## 8) GPT 輔助內容
- 使用 GPT 協助整理 CRISP-DM 報告架構、Lasso + OLS 流程說明、Notebook 撰寫提示及交付文件清單。  
- GPT 提供程式片段（特徵選擇、預測區間繪圖、殘差診斷）與格式化建議，最後由本人整合、執行並驗證結果。

---

## 9) NotebookLM 研究摘要（100 字以上）
參考 NotebookLM 彙整：  
- Kaggle《Red Wine Quality》資料集說明，重點強調品質評分來源、理化指標意義與常見資料清理步驟。  
- UCI Machine Learning Repository 與《A Physicochemical Examination of the Portugese Red Wine》研究；說明揮發性酸、硫酸鹽、酒精對品質的實驗結論。  
- scikit-learn 與 statsmodels 官方文件，涵蓋 LassoCV 交叉驗證流程、OLS 假設檢定、Durbin-Watson 及信賴/預測區間計算方式。整體摘要約 180 字。

---
