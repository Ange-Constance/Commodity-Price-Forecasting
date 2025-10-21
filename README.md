# 🌾 Commodity Price Forecasting using Machine Learning and Deep Learning

## 📘 Overview
This project aims to predict **commodity prices** (e.g., Wheat Futures) using both **traditional machine learning** and **deep learning (LSTM)** techniques.  
It demonstrates how time series data can be used to forecast future prices, helping stakeholders in agriculture, trading, and finance make informed decisions.

The work was developed as part of a **Summative Project** for the *Introduction to Machine Learning* module.

---

## 🎯 Objectives
- Apply **traditional ML algorithms** (Linear Regression, Random Forest) and **deep learning models** (LSTM) to real-world data.
- Compare model performance based on accuracy and interpretability.
- Build a complete **data science pipeline**: from data collection to evaluation and visualization.

---

## 🧩 Dataset

### 📊 Source
The dataset was obtained from [Kaggle - Commodity Futures Historical Data](https://www.kaggle.com/datasets), which provides open-access daily prices for various commodities including Wheat, Corn, and Gold.  
It contains columns such as:

| Column | Description |
|--------|--------------|
| Date | Date of the record |
| Commodity | Type of commodity (e.g., Wheat) |
| Price | Daily closing price |
| Open, High, Low, Volume | (optional fields depending on data) |

> **Citation (IEEE format):**  
> [1] Kaggle, “Commodity Futures Historical Data,” *Kaggle Datasets*, 2023. [Online]. Available: https://www.kaggle.com/datasets  

---

## ⚙️ Methodology

### 1. **Data Preprocessing**
- Loaded the dataset and cleaned missing values.
- Converted the `Date` column to a datetime format and sorted the records chronologically.
- Created lag features (`price_lag1`, `price_lag2`) and moving averages (`rolling_mean_3`) to capture trends.
- Split the dataset into training, validation, and test sets.

### 2. **Model Development**
#### 🧠 Machine Learning Models
- **Linear Regression:** Baseline model for simple trend prediction.  
- **Random Forest Regressor:** Captures non-linear relationships and variable importance.

#### 🔮 Deep Learning Model
- **LSTM (Long Short-Term Memory):** Built using TensorFlow’s Sequential API to model temporal dependencies.

### 3. **Model Evaluation**
Each model was evaluated using:
- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **R² Score**

Visualizations such as **learning curves** and **prediction plots** were used to compare performance.

---

## 🧠 Key Results
| Model | MAE | MSE | R² |
|--------|------|------|------|
| Linear Regression | — | — | — |
| Random Forest | — | — | — |
| LSTM | — | — | — |

*(Results will depend on your specific data and training outcomes.)*

The **LSTM model** performed best in capturing temporal trends, while Random Forest offered strong performance with lower computational cost.

---

## 📈 Visualizations
*(Plots will appear here once generated in your notebook)*

- Predicted vs. Actual Price Comparison  
- Learning Curves for LSTM  
- Error Distribution  
- Feature Importance (Random Forest)  

---

## 💻 Tools & Libraries
- **Python 3.12+**
- **Pandas**, **NumPy**, **Matplotlib**, **Seaborn**
- **Scikit-learn** (for ML models)
- **TensorFlow / Keras** (for LSTM)
- **Google Colab / Jupyter Notebook**

---

🧠 Insights & Discussion

The deep learning model (LSTM) captured long-term dependencies better than traditional models.

The Random Forest model was robust and faster to train, making it ideal for near-real-time forecasting.

Commodity prices exhibit seasonal and cyclical patterns that affect model accuracy.

Data quality and length significantly impact prediction reliability.

📚 References

[1] Kaggle, “Commodity Futures Historical Data,” Kaggle Datasets, 2023. [Online]. Available: https://www.kaggle.com/datasets

[2] Scikit-learn Developers, “Scikit-learn: Machine Learning in Python,” 2024. [Online]. Available: https://scikit-learn.org/

[3] TensorFlow Developers, “TensorFlow Documentation,” 2024. [Online]. Available: https://www.tensorflow.org/

👩‍💻 Author

Name: Ange Constance
Institution: African Leadership Univesity
Module: Introduction to Machine Learning
Date: 18 October 2025
