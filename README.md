## Bankruptcy Prediction using Machine Learning

End-to-end Machine Learning project that predicts whether a company will go bankrupt using the **Taiwan Company Bankruptcy Prediction** dataset.

It includes:

- Clean preprocessing pipeline (imputation + scaling)
- Multiple ML models + evaluation
- Feature importance visualization
- A professional **Streamlit dashboard** (manual input + CSV prediction + dataset analysis + model insights)

---

### Demo (Screenshots)

After you run training and the dashboard, you can add screenshots here:

- `notebooks/model_performance.png`
- `notebooks/feature_importance.png`
- `notebooks/class_distribution.png`
- `notebooks/correlation_heatmap.png`

---

### Project Structure

```text
bankruptcy-prediction-ml/
├── data/
├── notebooks/
├── models/
├── app/
├── main.py
├── requirements.txt
└── README.md
```

- **`data/`**: dataset CSV goes here
- **`notebooks/`**: generated plots (PNG) are saved here
- **`models/`**: trained model file `bankruptcy_model.pkl`
- **`app/`**: Streamlit dashboard `streamlit_app.py`
- **`main.py`**: training, evaluation, model comparison, saving model

---

### Dataset

Download the **Taiwan Company Bankruptcy Prediction** dataset (CSV).

Place it at:

```text
data/company_bankruptcy_prediction.csv
```

Target column must be named **`Bankrupt?`**:

- `0` = Not Bankrupt
- `1` = Bankrupt

---

### Tech Stack

- Python
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- pickle
- streamlit (dashboard)

---

### Setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
```

Windows PowerShell:

```bash
.\.venv\Scripts\activate
pip install -r requirements.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

---

### Train Models + Save the Best Model

From the project root:

```bash
python main.py
```

This will:

- Run EDA and save plots to `notebooks/`
- Train and compare multiple models:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Extra Trees
  - SVM (RBF)
- Evaluate each model using:
  - Accuracy
  - Confusion Matrix
  - Classification Report
  - F1 (macro)
- Select the **best model by accuracy**
- Save the trained pipeline + metadata bundle to:

```text
models/bankruptcy_model.pkl
```

---

### Streamlit Prediction Dashboard

Run the dashboard (recommended way on Windows):

```bash
python -m streamlit run app/streamlit_app.py
```

Dashboard features:

- **Predict**
  - Manual input (single company)
  - CSV prediction (multiple companies) + download predictions
  - If your CSV includes `Bankrupt?`, the dashboard also shows **evaluation metrics**
- **Analyze Dataset**
  - Upload any dataset CSV and get quick EDA (stats + correlation heatmap)
- **Model Insights**
  - Model comparison table (accuracy, F1)
  - Feature importance visualization (top 20)

---

### Results

The dataset is often **imbalanced** (many more non-bankrupt companies), so accuracy can be high even with simple models.

This project:

- Uses **class weighting** for more robust learning
- Includes quick **hyperparameter tuning** (small grid search) to improve performance
- Reports **F1 (macro)** along with accuracy

---

### Notes

- If your dataset file name is different, edit `DATA_PATH` in `main.py`.
- The model file `models/bankruptcy_model.pkl` contains a full **pipeline** (preprocessing + model), so you can load it and predict directly.

---

### License

Free to use for learning/portfolio. Please also follow the dataset provider’s license/terms.

