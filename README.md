Live link: https://customer-churn-prediction-using-ann-te8rpstdi9qhbg8drttlc3.streamlit.app/

# Customer Churn Prediction using Artificial Neural Networks (ANN)

An end-to-end deep learning project that predicts bank customer churn probability using an Artificial Neural Network (ANN) built with TensorFlow/Keras, accompanied by an interactive Streamlit web application for real-time predictions.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [Model Architecture & Training](#model-architecture--training)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [1. Launching the Streamlit Web App](#1-launching-the-streamlit-web-app)
  - [2. TensorBoard Experiment Tracking](#2-tensorboard-experiment-tracking)
  - [3. Running Notebooks](#3-running-notebooks)
- [Inference Pipeline](#inference-pipeline)
- [Future Enhancements](#future-enhancements)

---

## Overview

Customer churn occurs when customers stop doing business with a company. In banking, retaining existing customers is significantly more cost-effective than acquiring new ones.

This project analyzes bank customer profiles (demographics, credit scores, account balances, activity status, product holdings) to forecast whether a customer is likely to churn (`Exited = 1`) or stay (`Exited = 0`). It provides a trained Deep Learning ANN model alongside pre-saved encoders and scalers, exposed through an intuitive Streamlit interface for non-technical stakeholders and loan/account officers.

---

## Key Features

- **Automated Data Preprocessing**:
  - Drops non-predictive identifiers (`RowNumber`, `CustomerId`, `Surname`).
  - Encodes categorical variables (`Gender` via `LabelEncoder`, `Geography` via `OneHotEncoder`).
  - Feature normalization using `StandardScaler`.
  - Serialized encoders and scalers (`.pkl`) for consistent inference.
- **Deep Learning ANN Classifier**:
  - Multi-layer perceptron built with Keras `Sequential` API.
  - Rectified Linear Unit (`ReLU`) hidden activations and `Sigmoid` output activation.
  - Regularized with `EarlyStopping` to prevent overfitting.
  - Integrated `TensorBoard` callback for epoch-by-epoch loss and accuracy metrics.
- **Interactive Streamlit Web Dashboard**:
  - Dynamic user input controls (sliders, numeric fields, dropdowns).
  - Instant inference showing churn probability score and classification verdict.
- **Reproducible Experiments**:
  - Jupyter notebooks detailing data exploration, training, and manual inference testing.

---

## Tech Stack

- **Deep Learning & ML**: TensorFlow 2.x, Keras, Scikit-learn
- **Data Manipulation & Math**: Pandas, NumPy
- **Visualization & Monitoring**: Matplotlib, TensorBoard
- **Web App / UI**: Streamlit
- **Language**: Python 3.12

---

## Dataset

The project uses the `Churn_Modelling.csv` dataset (10,000 records).

### Features:
| Feature | Type | Description |
| :--- | :--- | :--- |
| `CreditScore` | Integer | Customer's credit score |
| `Geography` | Categorical | Country of residence (`France`, `Germany`, `Spain`) |
| `Gender` | Categorical | `Male` or `Female` |
| `Age` | Integer | Customer's age in years |
| `Tenure` | Integer | Number of years the customer has been with the bank |
| `Balance` | Float | Account balance |
| `NumOfProducts`| Integer | Number of bank products used (1 to 4) |
| `HasCrCard` | Binary | Whether the customer holds a credit card (`1` = Yes, `0` = No) |
| `IsActiveMember`| Binary | Whether the customer is an active member (`1` = Yes, `0` = No) |
| `EstimatedSalary`| Float | Estimated annual salary |
| **`Exited` (Target)** | Binary | Customer churn status (`1` = Churned, `0` = Retained) |

---

## Model Architecture & Training

The Artificial Neural Network (ANN) is structured as follows:

```text
Input Features (12)
        │
   ▼
[Dense Layer: 64 Neurons, ReLU]
        │
   ▼
[Dense Layer: 32 Neurons, ReLU]
        │
   ▼
[Dense Layer: 16 Neurons, ReLU]
        │
   ▼
[Output Layer: 1 Neuron, Sigmoid]
```

- **Loss Function**: Binary Cross-Entropy (`binary_crossentropy`)
- **Optimizer**: Adam (`learning_rate=0.001`)
- **Metrics**: Accuracy (~86–87% on validation set)
- **Callbacks**:
  - `EarlyStopping`: Monitored `val_loss` with patience of 10 epochs and `restore_best_weights=True`.
  - `TensorBoard`: Logs histograms and scalar metrics for visualization.

---

## Project Structure

```bash
customer-churn-prediction/
├── app.py                       # Streamlit web application
├── Churn_Modelling.csv          # Bank customer dataset
├── experiments.ipynb            # Notebook: EDA, preprocessing, ANN training & evaluation
├── prediction.ipynb             # Notebook: Sample inference pipeline verification
├── salaryRegression.ipynb       # Notebook: Experimental salary regression exploration
├── model.h5                     # Trained Keras Sequential model weights & architecture
├── label_encoder_gender.pkl     # Pickled LabelEncoder for 'Gender'
├── onehot_encoder_geography.pkl # Pickled OneHotEncoder for 'Geography'
├── scaler.pkl                   # Pickled StandardScaler for feature scaling
├── logs/                        # TensorBoard training log directory
│   └── fit/
├── requirements.txt             # Project dependencies
├── runtime.txt                  # Python runtime specification (python-3.12)
└── README.md                    # Project documentation
```

---

## Getting Started

### Prerequisites

- Python 3.10+ (Python 3.12 recommended)
- `pip` package manager

### Installation

1. **Clone the repository or navigate to the project directory:**
   ```bash
   cd customer-churn-prediction
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate       # On Linux/macOS
   # or: venv\Scripts\activate   # On Windows
   ```

3. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Launching the Streamlit Web App

Start the interactive web application locally:

```bash
streamlit run app.py
```

Once running, open your browser at `http://localhost:8501`. Enter customer parameters (credit score, geography, gender, age, balance, etc.) to receive real-time churn predictions.

### 2. TensorBoard Experiment Tracking

To inspect training curves, loss, and accuracy progression:

```bash
tensorboard --logdir logs/fit
```

Open `http://localhost:6006` in your browser.

### 3. Running Notebooks

Launch Jupyter Notebook or Jupyter Lab to inspect or re-train the models:

```bash
jupyter notebook experiments.ipynb
```

---

## Inference Pipeline

When a prediction is requested (either via `app.py` or `prediction.ipynb`):
1. **Raw Inputs Collected**: Demographic and banking details.
2. **Categorical Encoding**:
   - `Gender` is encoded to `0` or `1` using `label_encoder_gender.pkl`.
   - `Geography` is transformed into one-hot columns (`Geography_France`, `Geography_Germany`, `Geography_Spain`) using `onehot_encoder_geography.pkl`.
3. **Feature Alignment & Scaling**: Features are aligned with the training schema and scaled using `scaler.pkl`.
4. **ANN Evaluation**: `model.h5` predicts a continuous probability $p \in [0, 1]$.
5. **Thresholding**:
   - $p > 0.5 \implies$ **Customer likely to churn**
   - $p \le 0.5 \implies$ **Customer not likely to churn**

---

## Future Enhancements

- [ ] Add model explainability (SHAP / LIME values) in the Streamlit UI to highlight the top features driving churn risk.
- [ ] Implement hyperparameter tuning using Keras Tuner.
- [ ] Containerize application using Docker for one-click deployment.
- [ ] Complete the regression model for salary estimation in `salaryRegression.ipynb`.
