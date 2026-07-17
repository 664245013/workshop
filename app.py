import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==============================
# Page Configuration
# ==============================
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="💰",
    layout="wide"
)

# ==============================
# Load Model
# ==============================
@st.cache_resource
def load_model():
    model = joblib.load('svm_loan_model.pkl')
    info = joblib.load('model_info.pkl')
    return model, info

model, model_info = load_model()

# ==============================
# Sidebar
# ==============================
st.sidebar.title("💰 Loan Approval Predictor")
st.sidebar.markdown("---")
st.sidebar.info("🤖 โมเดล SVM สำหรับทำนายการอนุมัติสินเชื่อ")

# ==============================
# Main UI
# ==============================
st.title("🏦 ระบบทำนายการอนุมัติสินเชื่อ")
st.markdown("กรอกข้อมูลด้านล่างเพื่อตรวจสอบโอกาสการอนุมัติสินเชื่อของคุณ")
st.markdown("---")

# ==============================
# Input Form
# ==============================
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 ข้อมูลส่วนตัว")
    age = st.number_input("อายุ (ปี)", min_value=18, max_value=100, value=30, step=1)
    gender = st.selectbox("เพศ", model_info['categories']['gender'])
    education = st.selectbox("ระดับการศึกษา", model_info['categories']['education'])
    num_children = st.number_input("จำนวนบุตร", min_value=0, max_value=20, value=0, step=1)
    
    st.subheader("🏠 ข้อมูลที่อยู่อาศัย")
    home_ownership = st.selectbox("สถานะที่อยู่อาศัย", model_info['categories']['home_ownership'])

with col2:
    st.subheader("💵 ข้อมูลทางการเงิน")
    income = st.number_input("รายได้ต่อปี (บาท)", min_value=0, value=50000, step=1000)
    loan_amount = st.number_input("จำนวนเงินกู้ (บาท)", min_value=0, value=10000, step=1000)
    loan_purpose = st.selectbox("วัตถุประสงค์การกู้", model_info['categories']['loan_purpose'])
    interest_rate = st.number_input("อัตราดอกเบี้ย (%)", min_value=0.0, max_value=50.0, value=10.0, step=0.1)
    credit_score = st.number_input("คะแนนเครดิต", min_value=300, max_value=900, value=650, step=1)
    loan_term = st.number_input("ระยะเวลาผ่อน (ปี)", min_value=1, max_value=30, value=5, step=1)
    credit_limit = st.number_input("วงเงินเครดิตที่มี (บาท)", min_value=0, value=5000, step=500)

# ==============================
# Predict Button
# ==============================
st.markdown("---")
if st.button("🔮 ทำนายผล", type="primary", use_container_width=True):
    # สร้าง DataFrame จากข้อมูล input
    input_data = pd.DataFrame({
        'age': [age],
        'gender': [gender],
        'education': [education],
        'income': [income],
        'num_children': [num_children],
        'home_ownership': [home_ownership],
        'loan_amount': [loan_amount],
        'loan_purpose': [loan_purpose],
        'interest_rate': [interest_rate],
        'credit_score': [credit_score],
        'loan_term': [loan_term],
        'credit_limit': [credit_limit]
    })
    
    # ทำนาย
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    
    # แสดงผล
    st.markdown("---")
    result_col1, result_col2 = st.columns(2)
    
    with result_col1:
        if prediction == 1:
            st.success("✅ ผลทำนาย: **ได้รับการอนุมัติ**")
            st.metric("ความน่าจะเป็นที่จะอนุมัติ", f"{probability[1]*100:.2f}%")
        else:
            st.error("❌ ผลทำนาย: **ไม่ได้รับการอนุมัติ**")
            st.metric("ความน่าจะเป็นที่จะอนุมัติ", f"{probability[1]*100:.2f}%")
    
    with result_col2:
        st.subheader("📊 ข้อมูลที่คุณกรอก")
        st.dataframe(input_data.T.rename(columns={0: 'ค่า'}), use_container_width=True)
    
    # คำแนะนำ
    st.markdown("---")
    st.subheader("💡 คำแนะนำ")
    if probability[1] < 0.5:
        st.warning("""
        **เพื่อเพิ่มโอกาสในการอนุมัติ:**
        - 📈 เพิ่มคะแนนเครดิต (Credit Score)
        - 💰 ลดจำนวนเงินกู้หรือเพิ่มรายได้
        - 📉 ลดอัตราดอกเบี้ยที่ขอ
        - 🏠 พิจารณาประเภทที่อยู่อาศัยที่เหมาะสม
        """)
    else:
        st.info("ข้อมูลของคุณอยู่ในเกณฑ์ที่ดี มีโอกาสสูงที่จะได้รับการอนุมัติ!")

# ==============================
# Footer
# ==============================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    <small>Developed with ❤️ using Streamlit + SVM | Machine Learning Project</small>
    </div>
    """, 
    unsafe_allow_html=True
)