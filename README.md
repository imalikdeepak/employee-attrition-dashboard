# 🤖 AI-Driven Employee Attrition Dashboard

This project predicts employee attrition using a Machine Learning model (Random Forest) and provides interactive HR analytics dashboards for data-driven retention decisions.

---

## 🔍 What This Dashboard Does
- Predicts employee attrition probability
- Segments employees into Low / Medium / High risk
- Displays key HR KPIs and visual insights
- Provides single-employee attrition analysis
- Allows manual attrition prediction for new employees
- Identifies top employees most likely to leave
- Generates downloadable attrition reports

---

## 🧠 Machine Learning Model Used
- Random Forest Classifier (Accuracy ~89%)

---

## 📊 Dataset Used
This project uses a **public sample HR attrition dataset** sourced from **Kaggle**, commonly used for learning and analytics practice.

- The dataset is **synthetic/sample data**, not real employee data
- Used only for **model training and demonstration**
- Actual employee data is not shared due to privacy considerations

Dataset source: Kaggle

---

## 📂 Input Data (CSV File Required)

### Mandatory Columns
The uploaded CSV file **must contain** the following columns:

- EmployeeNumber  
- Attrition (Yes / No)  
- Age  
- Department  
- JobRole  
- MonthlyIncome  
- YearsAtCompany  
- JobSatisfaction  
- WorkLifeBalance  
- EnvironmentSatisfaction  
- PerformanceRating  
- OverTime  

### System-Handled Columns
- EmployeeCount  
- Over18  
- StandardHours  

---

## 🛠 Tech Stack
- Python
- Streamlit
- Pandas, NumPy
- Scikit-learn
- Plotly

---

## ▶ How to Run This Project
```bash
pip install -r requirements.txt
streamlit run app.py
