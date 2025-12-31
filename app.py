import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="AI Employee Attrition Dashboard",
    layout="wide"
)

st.markdown(
    """<h1 style='text-align:center;'>🤖 AI-Driven Employee Attrition Dashboard</h1><p style="text-align:center;color:#6b7280;"> • Predict • Analyze • Retain Talent</p>""",
    unsafe_allow_html=True
)

st.markdown("## 📘 Instructions for HR (Before Uploading File)")

st.info("""
### ✅ File Upload Guidelines

Please upload the **Employee Attrition CSV file** in the correct format to ensure accurate analysis.

### 📂 File Requirements
- File must be in **CSV format (.csv)**
- Each row must represent **one employee**
- Column names must **not be changed**

### 📌 Mandatory Columns Required
The file must contain the following key columns:

• EmployeeNumber (unique ID for each employee)  
• Attrition (Yes / No)  
• Age  
• Department  
• JobRole  
• MonthlyIncome  
• YearsAtCompany  
• JobSatisfaction  
• WorkLifeBalance  
• EnvironmentSatisfaction  
• PerformanceRating  
• OverTime  

### ⚙️ System-Handled Columns
The following columns are **automatically handled by the system** and require no action:
• EmployeeCount  
• Over18  
• StandardHours  

### 🎯 Data Quality Guidelines (For Best Accuracy ~89%)
✔ No missing values  
✔ Original categorical values (Yes/No)  
✔ Unique EmployeeNumber for each employee  

### ⚠️ Important Disclaimer
This AI system provides **predictive insights only**.  
Final HR decisions should always combine AI insights with **managerial judgment and employee discussions**.
""")

st.markdown("## 📊 Dashboard Output Explanation")

st.success("""
### 🔍 What This Dashboard Shows

After uploading the employee dataset, the dashboard automatically generates the following insights:

### 📈 Key HR KPIs
• **Total Employees** – Total number of employees analyzed  
• **High Risk Employees** – Employees with high attrition probability  
• **Low Risk Employees** – Employees with low attrition probability  
• **Average Attrition Probability** – Overall workforce risk level  
• **Model Accuracy (89%)** – Performance of the trained AI model  

### 📊 Visual Insights
• **Histogram** – Distribution of attrition probabilities  
• **Pie Chart** – Employee risk segmentation (Low / Medium / High)  
• **Bar Chart** – Average attrition risk by department  
• **Box Plot** – Attrition variation across job roles  

### 👤 Single Employee Attrition Analysis
• Select an employee using **Employee Number**
• View key employee details
• A **large meter gauge** displays the attrition risk percentage  

### 🧠 Employee Attrition Predictor (Manual Entry)
• HR can manually enter employee details  
• System predicts future attrition risk  
• Useful for new hires, promotions, or internal transfers  

### 🔝 Top 10 Employees Likely to Leave
• Automatically ranked list based on attrition probability  
• Helps HR prioritize retention strategies  

### ⬇ Downloadable Report
• Full employee attrition report in CSV format  
• Sorted by highest attrition probability  
• Can be used for HR reviews and management meetings  
""")

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "📂 Upload Employee CSV File",
    type=["csv"]
)

if uploaded_file is None:
    st.info("⬆ Please upload the Employee Attrition CSV file to proceed")
    st.stop()

df = pd.read_csv(uploaded_file)

# --------------------------------------------------
# STORE LABEL ENCODERS (CRITICAL FIX)
# --------------------------------------------------
encoders = {}

for col in df.select_dtypes(include='object'):
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le


# --------------------------------------------------
# PREPROCESSING (MATCHES TRAINING)
# --------------------------------------------------
df = df.drop(['EmployeeCount', 'Over18', 'StandardHours'], axis=1)
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

le = LabelEncoder()
for col in df.select_dtypes(include='object'):
    if col != 'Attrition':
        df[col] = le.fit_transform(df[col])

X = df.drop(['Attrition', 'EmployeeNumber'], axis=1)
df['Attrition_Probability'] = model.predict_proba(X)[:, 1]

df['Risk_Level'] = pd.cut(
    df['Attrition_Probability'],
    bins=[0, 0.35, 0.7, 1],
    labels=['Low Risk', 'Medium Risk', 'High Risk']
)

# --------------------------------------------------
# KPI SECTION
# --------------------------------------------------
st.markdown("## 📊 Key HR KPIs")

k1, k2, k3, k4, k5 = st.columns(5)

k1.metric("👥 Total Employees", len(df))
k2.metric("🔥 High Risk", (df['Risk_Level'] == 'High Risk').sum())
k3.metric("✅ Low Risk", (df['Risk_Level'] == 'Low Risk').sum())
k4.metric("📈 Avg Attrition Probability", f"{df['Attrition_Probability'].mean():.2%}")
k5.metric("🎯 Model Accuracy", "89%")

# --------------------------------------------------
# INSIGHT CHARTS
# --------------------------------------------------
st.markdown("## 📈 Attrition Insights")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.plotly_chart(
        px.histogram(df, x="Attrition_Probability",
                     template="plotly_dark",
                     title="Attrition Probability Distribution"),
        use_container_width=True
    )

with c2:
    st.plotly_chart(
        px.pie(df, names="Risk_Level",
               template="plotly_dark",
               title="Risk Segmentation"),
        use_container_width=True
    )

with c3:
    dept = df.groupby("Department")['Attrition_Probability'].mean().reset_index()
    st.plotly_chart(
        px.bar(dept, x="Department", y="Attrition_Probability",
               template="plotly_dark",
               title="Avg Attrition by Department"),
        use_container_width=True
    )

with c4:
    st.plotly_chart(
        px.box(df, x="JobRole", y="Attrition_Probability",
               template="plotly_dark",
               title="Attrition by Job Role"),
        use_container_width=True
    )

# --------------------------------------------------
# 🔍 SINGLE EMPLOYEE ATTRITION ANALYSIS (CORRECT & CONSISTENT)
# --------------------------------------------------
st.markdown("## 🔍 Single Employee Attrition Analysis")

left, right = st.columns([3, 2])

with left:
    emp_id = st.selectbox(
        "Select Employee Number",
        df['EmployeeNumber'],
        key="single_emp"
    )

    emp = df[df['EmployeeNumber'] == emp_id].iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Age", emp['Age'])
    c2.metric("Monthly Income", emp['MonthlyIncome'])
    c3.metric("Years at Company", emp['YearsAtCompany'])

    c4, c5 = st.columns(2)
    c4.metric("Job Satisfaction", emp['JobSatisfaction'])
    c5.metric("Work Life Balance", emp['WorkLifeBalance'])

with right:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=emp['Attrition_Probability'] * 100,  # ✅ SAME MODEL OUTPUT
        number={'suffix': '%'},
        title={"text": "Attrition Risk"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#ef4444"},
            'steps': [
                {'range': [0, 35], 'color': "#22c55e"},
                {'range': [35, 70], 'color': "#f59e0b"},
                {'range': [70, 100], 'color': "#dc2626"}
            ]
        }
    ))

    fig.update_layout(
        template="plotly_dark",
        height=420,
        margin=dict(l=10, r=10, t=60, b=10)
    )

    st.plotly_chart(fig, use_container_width=True)



# --------------------------------------------------
# 🧠 EMPLOYEE ATTRITION PREDICTOR (MANUAL ENTRY)
# --------------------------------------------------
st.markdown("## 🧠 Employee Attrition Predictor (Manual Entry)")

with st.form("manual_attrition_predictor"):

    c1, c2, c3, c4 = st.columns(4)

    age = c1.number_input("Age", 18, 60, 30)
    income = c2.number_input("Monthly Income", 1000, 200000, 30000)
    years = c3.number_input("Years at Company", 0, 40, 3)
    overtime = c4.selectbox("OverTime", encoders['OverTime'].classes_)

    dept = c1.selectbox("Department", encoders['Department'].classes_)
    role = c2.selectbox("Job Role", encoders['JobRole'].classes_)
    travel = c3.selectbox("Business Travel", encoders['BusinessTravel'].classes_)

    js = c4.slider("Job Satisfaction", 1, 4, 2)
    wlb = c1.slider("Work Life Balance", 1, 4, 2)
    env = c2.slider("Environment Satisfaction", 1, 4, 2)
    perf = c3.slider("Performance Rating", 1, 4, 3)

    predict_btn = st.form_submit_button("Predict Attrition Risk")

if predict_btn:

    # Step 1: Create empty row with ALL training columns
    manual_emp = pd.DataFrame(columns=X.columns)
    manual_emp.loc[0] = 0  # initialize safely

    # Step 2: Assign numeric features
    manual_emp.at[0, 'Age'] = age
    manual_emp.at[0, 'MonthlyIncome'] = income
    manual_emp.at[0, 'YearsAtCompany'] = years
    manual_emp.at[0, 'JobSatisfaction'] = js
    manual_emp.at[0, 'WorkLifeBalance'] = wlb
    manual_emp.at[0, 'EnvironmentSatisfaction'] = env
    manual_emp.at[0, 'PerformanceRating'] = perf

    # Step 3: Encode categorical features EXACTLY like training
    manual_emp.at[0, 'OverTime'] = encoders['OverTime'].transform([overtime])[0]
    manual_emp.at[0, 'Department'] = encoders['Department'].transform([dept])[0]
    manual_emp.at[0, 'JobRole'] = encoders['JobRole'].transform([role])[0]
    manual_emp.at[0, 'BusinessTravel'] = encoders['BusinessTravel'].transform([travel])[0]

    # Step 4: Predict probability
    prob = model.predict_proba(manual_emp)[:, 1][0]


    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={'suffix': '%'},
        title={"text": "Predicted Attrition Risk"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#ef4444"},
            'steps': [
                {'range': [0, 35], 'color': "#22c55e"},
                {'range': [35, 70], 'color': "#f59e0b"},
                {'range': [70, 100], 'color': "#dc2626"}
            ]
        }
    ))

    fig.update_layout(
        template="plotly_dark",
        height=380,
        margin=dict(l=10, r=10, t=60, b=10)
    )

    st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------
# TOP 10 EMPLOYEES
# --------------------------------------------------
st.markdown("## 🔝 Top 10 Employees Likely to Leave")

st.dataframe(
    df.sort_values("Attrition_Probability", ascending=False).head(10)[
        ['EmployeeNumber','Age','Department','JobRole',
         'MonthlyIncome','YearsAtCompany','Attrition_Probability']
    ]
)

# --------------------------------------------------
# DOWNLOAD CSV
# --------------------------------------------------
st.markdown("## ⬇ Download Attrition Report")

csv = df.sort_values(
    "Attrition_Probability", ascending=False
)[
    ['EmployeeNumber','Department','JobRole',
     'MonthlyIncome','YearsAtCompany',
     'Attrition_Probability','Risk_Level']
].to_csv(index=False)

st.download_button(
    "📥 Download CSV Report",
    csv,
    "employee_attrition_report.csv",
    "text/csv"
)


st.markdown(""" <hr style="border:1px solid #444;"> <p style="text-align:center; color:grey; font-size:14px;"> Empowered with Machine Learning & HR Analytics | © 2025 Deepak Malik. All Rights Reserved. </p> """, unsafe_allow_html=True)
