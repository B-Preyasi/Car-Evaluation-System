# 🚗 Car Evaluation System using Machine Learning

A Machine Learning web application that predicts the acceptability of a car based on its characteristics such as buying price, maintenance cost, number of doors, seating capacity, luggage boot size, and safety level.

The project is built using **Python**, **Scikit-learn**, and **Streamlit**, and uses a **Random Forest Classifier** trained on the **UCI Car Evaluation Dataset**.

---
## 🌐 Live Demo
https://car-evaluation-system-q639wuvp3rrv7wgikee3da.streamlit.app/

## 📌 Features

- 🚗 Predicts car acceptability in real time
- 🤖 Machine Learning based prediction
- 📊 Confidence score for each prediction
- 📈 Confusion Matrix visualization
- 📉 Feature Importance visualization
- 🎨 Interactive Streamlit dashboard
- 💾 Trained model saved using Joblib

---

## 📷 Application Preview

<img width="1373" height="806" alt="Screenshot 2026-06-27 at 9 34 29 PM" src="https://github.com/user-attachments/assets/f861743a-b0ba-40df-b756-af4afd896d00" />


---

## 🏗️ Project Architecture

```
                User
                  │
                  ▼
          Streamlit Web App
                  │
                  ▼
         User Input Features
                  │
                  ▼
         Data Preprocessing
          (Label Encoding)
                  │
                  ▼
      Random Forest Classifier
                  │
                  ▼
 Prediction + Confidence Score
                  │
                  ▼
      Display Result to User
```

---

## 📂 Project Structure

```
Car-Evaluation-System/
│
├── data/
│   └── car.data
│
├── models/
│   ├── car_model.pkl
│   ├── encoders.pkl
│   └── features.pkl
│
├── outputs/
│   ├── confusion_matrix.png
│   └── feature_importance.png
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── predict.py
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🗂 Dataset

**Dataset:** UCI Car Evaluation Dataset

### Features

| Feature | Description |
|----------|-------------|
| Buying | Buying price |
| Maintenance | Maintenance cost |
| Doors | Number of doors |
| Persons | Seating capacity |
| Luggage Boot | Luggage boot size |
| Safety | Safety level |

### Target Classes

- Unacceptable
- Acceptable
- Good
- Very Good

---

## ⚙️ Machine Learning Workflow

```
Dataset
    │
    ▼
Data Preprocessing
(Label Encoding)
    │
    ▼
Train-Test Split
    │
    ▼
Random Forest Training
    │
    ▼
Model Evaluation
    │
    ▼
Save Model
    │
    ▼
Streamlit Prediction App
```

---

## 🤖 Machine Learning Model

**Algorithm Used**

- Random Forest Classifier

### Why Random Forest?

- Handles categorical data effectively after encoding
- High prediction accuracy
- Reduces overfitting
- Robust and reliable
- Provides feature importance

---

## 📊 Model Evaluation

The model is evaluated using:

- Accuracy Score
- Confusion Matrix
- Classification Report
- Feature Importance

---

## 💻 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib
- Matplotlib
- Seaborn

---

## 🚀 Installation

### Clone Repository

```bash
git clone https://github.com/B-Preyasi/Car-Evaluation-System.git
```

### Navigate to Project

```bash
cd Car-Evaluation-System
```

### Create Virtual Environment

Windows

```bash
python -m venv venv
venv\Scripts\activate
```

Mac/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Train the Model

```bash
python -m src.train_model
```

This creates:

```
models/
    car_model.pkl
    encoders.pkl
    features.pkl
```

---

## 📈 Evaluate the Model

```bash
python -m src.evaluate_model
```

Generated files:

```
outputs/
    confusion_matrix.png
    feature_importance.png
```

---

## 🌐 Run the Streamlit App

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 🎯 Sample Prediction

| Input | Value |
|--------|-------|
| Buying | low |
| Maintenance | low |
| Doors | 4 |
| Persons | more |
| Luggage Boot | big |
| Safety | high |

### Prediction

```
Very Good
```

### Confidence Score

```
96.42%
```

---

## 📊 Future Improvements

- Deep Learning model comparison
- Support Vector Machine implementation
- XGBoost implementation
- Dark mode UI
- Deployment on Streamlit Cloud
- Docker support
- REST API using Flask/FastAPI
- User authentication
- Prediction history database

---

## 👨‍💻 Author

**Bandana Preyasi**

B.Tech CSE (AI & ML)

---

## 📄 License

This project is developed for educational and learning purposes.
