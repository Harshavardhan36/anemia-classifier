# 🩸 Anemia Classification Using CBC Data

A Flask web application that classifies anemia from Complete Blood Count (CBC) values using three machine learning models.

## 📊 Dataset
- **Source:** [Kaggle — Anemia Dataset](https://www.kaggle.com/datasets/biswaranjanrao/anemia-dataset)
- **Samples:** 1,421 clinical records
- **Features:** Gender, Hemoglobin, MCH, MCHC, MCV
- **Target:** Result (0 = No Anemia, 1 = Anemia)

## 🤖 Models
| Model | Test Accuracy |
|-------|--------------|
| Logistic Regression (baseline) | 99% |
| SVM (RBF kernel) | ~99% |
| Random Forest (200 trees) | ~99% |
| Gradient Boosting (200 estimators) | ~99% |

5-fold cross-validation applied on all models.

## 📁 Project Structure
```
anemia-classifier/
├── app.py               # Flask backend
├── train_models.py      # Model training script
├── requirements.txt
├── anemia.csv           # Dataset (download from Kaggle)
├── model/               # Saved .pkl files (generated after training)
│   ├── svm.pkl
│   ├── random_forest.pkl
│   ├── gradient_boosting.pkl
│   └── scaler.pkl
└── templates/
    └── index.html       # Frontend UI
```

## 🚀 Setup & Run

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/anemia-classifier.git
cd anemia-classifier
```

### 2. Create virtual environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add the dataset
Download `anemia.csv` from [Kaggle](https://www.kaggle.com/datasets/biswaranjanrao/anemia-dataset) and place it in the project root.

### 5. Train the models
```bash
python train_models.py
```

### 6. Run the app
```bash
python app.py
```

Open your browser at **http://127.0.0.1:5000**

## 🛠 Tech Stack
- **Backend:** Python, Flask
- **ML:** Scikit-learn (SVM, Random Forest, Gradient Boosting)
- **Frontend:** HTML, CSS, Vanilla JS
- **Data:** Pandas, NumPy

## ⚠️ Disclaimer
This tool is for **educational and research purposes only**. It is not a substitute for professional medical diagnosis.
