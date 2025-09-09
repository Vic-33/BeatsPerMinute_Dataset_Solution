# Prediction of Beats Per Minute (BPM) Using Machine Learning  

## Table of Contents  
1. [Introduction](#introduction)  
2. [Objective](#objective)  
3. [Dataset](#dataset)  
   - Train Dataset  
   - Test Dataset  
   - Features  
4. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)  
5. [Feature Engineering](#feature-engineering)  
6. [Model Development](#model-development)  
   - Baseline Models  
   - Advanced Models  
   - Evaluation Metrics  
7. [Results](#results)  
8. [Requirements](#requirements)  
9. [How to Run the Project](#how-to-run-the-project)  
10. [Repository Structure](#repository-structure)  
11. [Future Scope](#future-scope)  
12. [Conclusion](#conclusion)  
13. [Authors](#authors)  

---

## Introduction  
Beats per minute (**BPM**) is one of the most important characteristics of music, defining its **tempo, rhythm, and genre**. Predicting BPM from audio features is useful in:  
- **Music recommendation systems** (Spotify, YouTube Music, etc.)  
- **DJ mixing and beat matching**  
- **Music therapy applications**  
- **Audio content analysis for mood classification**  

In this project, we leverage **machine learning (ML)** techniques to predict the **BPM of a song** using its extracted audio features.  

---

## Objective  
- Perform **exploratory data analysis (EDA)** to understand the audio dataset  
- Build and compare **multiple ML models** to predict BPM  
- Optimize model performance using **feature engineering** and **cross-validation**  
- Generate final BPM predictions for the **test dataset** (`test.csv`) and store results in `submission.csv`  

---

## Dataset  

### Train Dataset  
The training dataset contains several audio features (numeric and categorical) with **BeatsPerMinute** as the target column.  

### Test Dataset  
The `test.csv` contains the same features but **without BPM**, used for prediction.  

### Features  
Some example features commonly included in audio datasets (based on music information retrieval):  
- `RhythmScore` – strength of rhythmic patterns  
- `AudioLoudness` – perceived sound intensity  
- `VocalContent` – amount of vocal dominance  
- `AcousticQuality` – acoustic instrument presence  
- `InstrumentalScore` – non-vocal score proportion  
- `LivePerformanceLikelihood` – probability of live recording  
- `MoodScore` – emotion/mood representation  
- `TrackDurationMs` – song length in milliseconds  
- `Energy` – measure of intensity and activity  
- `BeatsPerMinute` – **target variable**  

---

## Exploratory Data Analysis (EDA)  
Performed in **`EDA_[Bonus].ipynb`**:  
- **Data Cleaning**  
  - Checked for missing values and handled them appropriately  
  - Removed obvious outliers (e.g., BPM > 400)  
- **Feature Distributions**  
  - Histograms and boxplots for all numeric features  
- **Correlation Analysis**  
  - Heatmap to detect strongest predictors of BPM  
- **Key Observations**  
  - Features like `Energy`, `RhythmScore`, and `TrackDurationMs` showed stronger correlations with BPM  
  - Some categorical variables needed encoding  

---

## ⚙️ Feature Engineering  
- **Encoding**: Converted categorical features into numerical (Label Encoding / One-Hot Encoding)  
- **Scaling**: StandardScaler/MinMaxScaler used for features with large magnitude differences  
- **Derived Features**:  
  - `Duration_Category` → grouped songs into short/medium/long  
  - Interaction terms between `RhythmScore` and `Energy`  
- **Feature Selection**: Kept only top predictors using feature importance from tree-based models  

---

## 🤖 Model Development  

Implemented in **`model.ipynb`**.  

### 1. Baseline Models  
- **Linear Regression**: simple benchmark  
- **Ridge & Lasso Regression**: for regularization and feature shrinkage  

### 2. Advanced Models  
- **Random Forest Regressor**: non-linear relationships captured  
- **Gradient Boosting (XGBoost, LightGBM, CatBoost)**: handled tabular data effectively  
- **Ensembling**: combined multiple models for improved accuracy  

### 3. Evaluation Metrics  
We used:  
- **Mean Squared Error (MSE)**  
- **Root Mean Squared Error (RMSE)**  
- **R² Score**  

These metrics ensured the model generalizes well.  

---

## Results  

| Model                 | RMSE     |
|-----------------------|----------|
| Linear Regression     | 26.465   |
| Random Forest         | 26.6747  |
| XGBoost               | 26.444   |
| CatBoost              | 26.439   |
| Lightgbm              | **26.408** |

- Final predictions are stored in **`submission.csv`**  

---

## 📦 Requirements  

The following dependencies are required to run this project:  

```txt
pandas
numpy
matplotlib
seaborn
scipy
scikit-learn
xgboost
lightgbm
catboost
