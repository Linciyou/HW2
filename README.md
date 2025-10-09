# 7114056076_hw2 — 多元線性回歸（CRISP-DM）專案說明
> 更新日期：2025-10-09  
> 作業主題：Multiple Linear Regression（含 **特徵選擇**、**模型評估**、**預測信賴/預測區間**）

---

## 1. CRISP-DM 文件說明（50%）
### 1.1 流程概覽
本專案完整實作 **CRISP-DM** 六大階段，針對紅酒品質預測提出可落地的分析流程：
1. **Business Understanding**：聚焦釀造端的品控與配方調整需求，設定降低 RMSE、提升 R²、提供預測區間為核心 KPI。  
2. **Data Understanding**：檢視 Kaggle *Wine Quality (Red)* 資料（`WineQT.csv`，1143 筆 × 12 欄），確認無缺值、分析理化特徵與 `quality` 的相關性。  
3. **Data Preparation**：執行 `drop_duplicates()`、`train_test_split(test_size=0.2, random_state=42)`，並以 `StandardScaler` 標準化；若資料含 `type` 欄位則轉為 0/1。  
4. **Modeling**：
   - Baseline：`Pipeline(StandardScaler, LinearRegression)` 搭配 5-fold 交叉驗證。
   - 特徵選擇：`LassoCV(cv=5)` 篩出八個核心特徵。
   - 最終模型：使用 `statsmodels.OLS` 進行多元線性回歸，輸出係數、p-value、Durbin-Watson 等統計量。  
   - 輔助模型：以 `alcohol` 建立簡單線性回歸示意並繪製 95% 信賴區間。
5. **Evaluation**：計算測試集 R²、MAE、RMSE 與 Baseline CV-RMSE，並產出預測 vs. 實際、殘差診斷圖、Q-Q Plot，評估模型假設與泛化能力。  
6. **Deployment**：提出批次預測腳本、儀表板監控、RMSE 閾值警示及資料漂移檢測等落地建議。

### 1.2 GPT 協助紀錄（15%）
- 透過 GPT 討論 CRISP-DM 報告架構、Lasso + OLS 的程式骨架與視覺化設計。  
- GPT 協助整理 Notebook 撰寫提示、提交項目清單與排版建議；最終程式由本人整合與驗證。  
- 對話重點：特徵工程流程、預測區間計算、殘差診斷詮釋與交付文件內容。

### 1.3 NotebookLM 研究摘要（100 字以上）（15%）
NotebookLM 彙整內容如下：  
- Kaggle《Red Wine Quality》競賽資料及 Notebook，說明品質評分來源、理化指標量測方式與常見資料清理步驟。  
- UCI Machine Learning Repository 與論文《A Physicochemical Examination of the Portugese Red Wine》，指出揮發性酸、硫酸鹽、酒精濃度對品質的實驗性結論。  
- scikit-learn 與 statsmodels 官方文件，涵蓋 LassoCV 交叉驗證流程、OLS 假設檢定、Durbin-Watson 指標與信賴/預測區間計算方法。  
以上摘要約 180 字，提供模型選擇與評估依據。

### 1.4 資料集來源與研究脈絡（10%）
- **資料集**：Kaggle *Wine Quality (Red)*，原始檔 `WineQT.csv`。  
- **內容**：11 項紅酒理化特徵（固定酸度、揮發性酸、檸檬酸、殘糖、氯化物、自由/總二氧化硫、密度、pH、硫酸鹽、酒精）與目標欄位 `quality`（0–10 分）。  
- **應用場景**：釀造批次品控、配方實驗設計與感官評測風險管理；亦可延伸至自動化警示系統。

---

## 2. 結果呈現（50%）
### 2.1 模型執行與特徵選擇（25%）
- `7114056076_hw2.ipynb` 內含完整程式碼，可由 `python` 或 `jupyter` 執行。  
- LassoCV 選出特徵：`volatile acidity`、`chlorides`、`total sulfur dioxide`、`density`、`pH`、`sulphates`、`alcohol`、`Id`。  
- OLS 係數顯示：酒精濃度、硫酸鹽正向影響品質；揮發性酸、氯化物負向影響，與文獻一致。

### 2.2 評估指標與視覺化（15%）
`outputs/metrics.csv` 指標如下：

| Metric | Value |
| --- | --- |
| R² | 0.327 |
| MAE | 0.477 |
| RMSE | 0.612 |
| Baseline CV-RMSE (5-fold) | 0.645 |

視覺化成果：  
- `outputs/predictions_with_PI.png`：測試集預測均值與 95% 預測區間，顯示大多數實際值落在區間內。  
- `outputs/residuals_vs_fitted.png`：殘差約對稱於 0，無明顯漏斗型。  
- `outputs/residuals_qq.png`：中段貼合對角線，尾端稍有偏離，提示可能存在少數極端值。  
- `outputs/SLR_alcohol_with_CI.png`：顯示酒精與品質的正向線性關係及其信賴區間。

### 2.3 Kaggle 名次 / 預測結果說明（10%）
- **Kaggle 名次**：目前僅於本地驗證，尚未上傳競賽成績（N/A）。  
- **預測成果**：測試 RMSE = 0.612、MAE = 0.477，與 Baseline CV-RMSE = 0.645 相近，顯示模型無明顯過擬合；預測區間約覆蓋 ±1 分品質差異，可作為風險參考。

---

## 3. 執行方式與檔案結構
- `WineQT.csv`：Kaggle 紅酒品質資料集。  
- `7114056076_hw2.ipynb`：主 notebook，含 CRISP-DM 全流程。  
- `outputs/`：儲存所有評估圖表與指標 (`metrics.csv`)。  
- 執行建議：於專案根目錄執行 `python`、`jupyter nbconvert --execute` 或直接於 Jupyter Notebook 內依序執行所有儲存格。

---

## 4. 後續工作建議
1. 取得 Kaggle API Token，將預測結果上傳競賽平台以取得排名。  
2. 進一步嘗試非線性模型（如 XGBoost、LightGBM），並評估不確定性估計方法。  
3. 建立批次化腳本，串接 `outputs/metrics.csv` 與圖表至內部儀表板，自動追蹤模型表現。

---
