# House Prices Prediction using Random Forest

## 📌 Project Overview
This project aims to predict house prices using **Machine Learning** techniques.
The model is trained on the famous **Kaggle House Prices Dataset** and deployed using a **Flask Web Application**.

The goal is to estimate the **SalePrice** of a house based on multiple features such as:
- Overall quality
- Living area
- Number of rooms
- Year built
- Neighborhood
- Garage size

---

## 📊 Dataset Description
- Dataset Source: Kaggle – House Prices: Advanced Regression Techniques
- Training data: `train.csv`
- Testing data: `test.csv`
- Target variable: **SalePrice**

The dataset contains:
- Numerical features (e.g. `GrLivArea`, `LotArea`)
- Categorical features (e.g. `Neighborhood`, `HouseStyle`)

Missing values were handled during preprocessing.

---

## 🤖 Machine Learning Model
The model used in this project is **Random Forest Regressor**.

### Why Random Forest?
- Handles non-linearity well
- Reduces overfitting using ensemble learning
- Works well with mixed feature types

### Model Concept
Random Forest is an **ensemble model** that combines multiple **Decision Trees**.

Each tree predicts a value, and the final prediction is the **average** of all trees:

\[
\hat{y} = \frac{1}{N} \sum_{i=1}^{N} T_i(x)
\]

Where:
- \(T_i(x)\) = prediction of the i-th tree
- \(N\) = number of trees

---

## 🧠 Feature Encoding
- Categorical features were encoded using **One-Hot Encoding**
- Model input columns are stored in `model_columns.joblib`
- Categorical allowed values are stored in `categorical_values.json`

---

## 🌐 Web Application
The project includes a **Flask web app** that allows users to:
- Enter house features through a form
- Get real-time price predictions

### Technologies Used
- Flask
- HTML / CSS
- Jinja2 Templates

---

## ⚙️ Installation & Usage

### 1️⃣ Clone the repository
```bash
git clone https://github.com/AliBarakat721/house-prices-advanced-regression-techniques.git
cd house-prices-advanced-regression-techniques
