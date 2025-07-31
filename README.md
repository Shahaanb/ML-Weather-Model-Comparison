# Comparative Analysis of Machine Learning Models for Weather Prediction

This repository contains the code and resources for the project "Comparative Analysis of Machine Learning Models for Weather Prediction," which evaluates various machine learning algorithms for forecasting six key meteorological parameters.

## 📝 Abstract

Accurate weather prediction plays a crucial role in various domains such as agriculture, disaster management, and energy planning. This study presents a comparative analysis of machine learning models used to predict six weather-related features: temperature, humidity, wind speed, wind direction, precipitation, and solar radiation. Each feature was modeled independently using a variety of algorithms including Random Forest, XGBoost, CatBoost, and Lasso Regression. The models were evaluated using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), the coefficient of determination ($R^2$), and standard classification metrics. Based on empirical performance, the most suitable model was identified for each weather parameter, highlighting the importance of feature-specific model selection.

---

## 📊 Dataset

The study utilizes a publicly available weather dataset sourced from **Kaggle**.
-   **Link:** https://www.kaggle.com/datasets/gurpreetchaggar/mumbai-weather
-   **Location:** Mumbai, India
-   **Time Span:** January 1, 2016, to November 15, 2020
-   **Records:** 1781 daily entries
-   **Features Modeled:**
    1.  Temperature (°C)
    2.  Humidity (%)
    3.  Wind Speed (km/h)
    4.  Wind Direction (°)
    5.  Solar Radiation (W/m²)
    6.  Precipitation (Binary: Rain/No Rain)

---

## ⚙️ Methodology

A systematic pipeline was implemented to preprocess data, train models, and evaluate their performance.

### 1. Data Preprocessing
-   **Train-Test Split:** A chronological 80/20 split was used to prevent data leakage, with the first 80% of data for training and the remaining 20% for testing.
-   **Feature Engineering:** Lag-based features (e.g., t-1, t-2) were created for time-dependent variables like temperature and wind direction to provide historical context.
-   **Data Scaling:** Features were normalized to a [0, 1] range using `MinMaxScaler` to accommodate models sensitive to feature magnitudes (e.g., Lasso, SVM).

### 2. Models Evaluated

A suite of models was trained independently for each weather parameter.

**For Regression Tasks (Temperature, Humidity, Wind Speed, etc.):**
-   Linear Regression
-   Lasso Regression
-   Support Vector Regressor (SVR)
-   Random Forest Regressor
-   XGBoost Regressor
-   LightGBM
-   CatBoost

**For Classification Task (Precipitation):**
-   Logistic Regression
-   Decision Tree Classifier
-   Random Forest Classifier
-   XGBoost Classifier
-   Support Vector Classifier (SVC)

### 3. Evaluation Metrics

-   **Regression:** Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² Score.
-   **Classification:** Accuracy, Precision, Recall, and F1 Score.

---

## 🏆 Results Summary

The study found that no single model was universally superior. The best-performing model was highly dependent on the specific characteristics of the weather parameter being predicted.

| Weather Parameter   | Best Performing Model     | Key Metrics                               |
| ------------------- | ------------------------- | ----------------------------------------- |
| **Temperature** | `Linear Regression`       | R²: 0.836, RMSE: 0.91, MAE: 0.66          |
| **Humidity** | `Lasso Regression`        | R²: 0.840, RMSE: 5.53, MAE: 3.83          |
| **Wind Speed** | `Lasso Regression`        | R²: 0.514, RMSE: 3.79, MAE: 2.86          |
| **Solar Radiation** | `Random Forest Regressor` | R²: 0.556, RMSE: 41.42, MAE: 29.74        |
| **Precipitation** | `Random Forest Classifier`| Accuracy: 0.90, F1 Score: 0.90            |
| **Wind Direction** | `CatBoost`                | R²: 0.50, RMSE: 31.51, MAE: 23.98         |

---


## 🚀 How to Run

To replicate the experiments, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```
    The `requirements.txt` file should contain libraries such as:
    - `pandas`
    - `numpy`
    - `scikit-learn`
    - `xgboost`
    - `catboost`
    - `lightgbm`
    - `matplotlib`
    - `seaborn`

3.  **Run the notebooks:**
    Open the Jupyter Notebooks in the `Different_Features/` directory to see the step-by-step analysis.

---

## 💡 Future Work

-   **Advanced Architectures:** Explore deep learning models like GRU, Transformers, or hybrid CNN-LSTMs.
-   **Expanded Feature Set:** Incorporate additional data like atmospheric pressure, cloud cover, or satellite imagery.
-   **Circular Feature Engineering:** Use sine/cosine transformations for wind direction to better handle its cyclical nature.
-   **Model Ensembling:** Create stacked or voting ensembles to combine the strengths of multiple models.

---

## 👨‍💻 Authors

-   Shahaan Bharucha
-   Ronak Daniels

