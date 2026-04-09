# Fraud Detection Project

This project contains:
- A full data analysis and model training workflow in Jupyter Notebook
- A trained fraud detection model file
- A Streamlit web app for real-time prediction

## Project Structure

- `notebook.ipynb` - EDA, preprocessing, model training, evaluation
- `Fraud_Detection_Dataset.csv` - Dataset used for training
- `fraud_detection_model.pkl` - Saved trained model
- `app.py` - Streamlit UI for user input and prediction
- `requirements.txt` - Python dependencies

## What The App Predicts

The app predicts whether a transaction is likely:
- Fraudulent
- Non-fraudulent

### User Inputs in the App

1. Transaction Type
2. Amount
3. Old Balance (Sender)
4. New Balance (Sender)
5. Old Balance (Receiver)
6. New Balance (Receiver)

Then click **Predict**.

## Dataset Source

Kaggle dataset used in this project:

https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset

---

## Run Locally (Step-by-Step)

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd "Fraud Detection"
```

### 2. Create a virtual environment

#### Windows (PowerShell)

```powershell
python -m venv .venv
```

#### macOS/Linux

```bash
python3 -m venv .venv
```

### 3. Activate the virtual environment

#### Windows (PowerShell)

```powershell
.\.venv\Scripts\Activate.ps1
```

#### Windows (cmd)

```bat
.venv\Scripts\activate.bat
```

#### macOS/Linux

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit app

```bash
streamlit run app.py
```

Open the URL shown in terminal (usually `http://localhost:8501`).

---

## Recommended Windows Command (No Activation Needed)

If virtual environment activation is blocked by policy, run directly with:

```powershell
& ".\.venv\Scripts\python.exe" -m pip install -r .\requirements.txt
& ".\.venv\Scripts\python.exe" -m streamlit run .\app.py
```

---

## Optional: Run on a Different Port

If 8501 is already in use:

```bash
streamlit run app.py --server.port 8502
```

Windows direct command:

```powershell
& ".\.venv\Scripts\python.exe" -m streamlit run .\app.py --server.port 8502
```

---

## How Model Loading Works

- The app first tries to load `fraud_detection_model.pkl`.
- If model loading fails due to scikit-learn version incompatibility, the app automatically retrains a compatible model from `Fraud_Detection_Dataset.csv` and saves it back to `fraud_detection_model.pkl`.

This makes first run robust across different local environments.

---

## Reproduce Model Training (Notebook)

1. Open `notebook.ipynb` in Jupyter/VS Code.
2. Run all cells.
3. Exported model will be saved as `fraud_detection_model.pkl`.

---

## Troubleshooting

### 1) Command fails with `streamlit app.py`
Use:

```bash
streamlit run app.py
```

### 2) `Port 8501 is not available`
Use another port:

```bash
streamlit run app.py --server.port 8502
```

### 3) `ModuleNotFoundError` / imports not found
Install dependencies inside the same environment:

```bash
pip install -r requirements.txt
```

### 4) PowerShell execution policy blocks activation
Use direct Python path commands:

```powershell
& ".\.venv\Scripts\python.exe" -m pip install -r .\requirements.txt
& ".\.venv\Scripts\python.exe" -m streamlit run .\app.py
```

---

## Tech Stack

- Python
- Pandas
- Scikit-learn
- Joblib
- Streamlit

---

## License

Add your preferred license here (for example: MIT).
