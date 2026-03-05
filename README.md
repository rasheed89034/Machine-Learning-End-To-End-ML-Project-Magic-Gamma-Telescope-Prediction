# Machine-Learning-End-To-End-ML-Project-Magic-Gamma-Telescope-Prediction

# 🔭 Magic Gamma Telescope Prediction 🛰️

A machine learning web application designed to classify high-energy gamma-ray signals and differentiate them from background hadrons. This project leverages data from the MAGIC (Major Atmospheric Gamma Imaging Cherenkov) Telescope, utilizing a trained Logistic Regression model deployed via an interactive Streamlit interface.

## 🚀 Project Overview

The MAGIC Gamma Telescope detects Cherenkov radiation produced by electromagnetic showers in the atmosphere. This tool takes 10 physical characteristics of these radiation images (such as ellipse length, width, asymmetry, and concentration) and predicts whether the signal is a primary gamma ray or a background cosmic ray.

**Key Features:**
* **Data Preprocessing:** Explores and standardizes complex astronomical data.
* **Predictive Modeling:** Implements a Logistic Regression classification model achieving **79.08% accuracy**.
* **Interactive UI:** A user-friendly Streamlit dashboard allowing users to input specific telescope feature values and instantly get a classification prediction.
* **Real-time Results:** Processes manual inputs through a saved standard scaler and model to output the final prediction.

## 🛠️ Tech Stack

* **Language:** Python
* **Machine Learning:** Scikit-Learn (Logistic Regression, StandardScaler)
* **Data Manipulation:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Web Framework:** Streamlit
* **Model Serialization:** Pickle

## 📂 Repository Structure

* `Magic Gamma Telescope.ipynb`: Jupyter Notebook detailing the dataset exploration, feature analysis, and data preparation.
* `ModelTraining.ipynb`: Jupyter Notebook handling the model training process, evaluation, and exporting the final model (`logisticModel.pkl`) and scaler (`standard.pkl`).
* `webapplication.py`: The main Streamlit web application script that creates the UI, loads the serialized models, and processes user inputs for predictions.
* `Models/`: Directory containing the saved `.pkl` files required to run the web application.

## 💻 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/magic-gamma-telescope-prediction.git](https://github.com/yourusername/magic-gamma-telescope-prediction.git)
   cd magic-gamma-telescope-prediction
