import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="ระบบทำนายการอนุมัติสินเชื่อ",
    page_icon="💰",
    layout="wide"
)

# โหลดโมเดล
@st.cache_resource
def load_model():
    try:
        model = joblib.load('svm_loan_model.pkl')
        info = joblib.load('model_info.pkl')
        return model, info
    except Exception as e:
        st.error(f" ไม่สามารถโหลดโมเดลได้: {e}")
        return None, None

model, model_info = load_model()

# UI
st.title("🏦 ระบบทำนายการอนุมัติสินเชื่อ")
st.markdown("---")

if model is None:
    st.error("❌ ไม่พบไฟล์โมเดล กรุณาตรวจสอบว่ามี svm_loan_model.pkl และ model_info.pkl")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("ℹ️ เกี่ยวกับระบบ")
    st.info("โมเดล SVM สำหรับทำนายการอนุมัติสินเชื่อ")
    st.markdown("---")
    st.write("**Features ที่ใช้:**")
    st.write("- ข้อมูลส่วนตัว")
    st.write("- ข้อมูลทางการเงิน")
    st.write("- ประวัติเครดิต")

# แบ่งคอลัมน์
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 ข้อมูลส่วนตัว")
    person_age = st.number_input("อายุ (ปี)", min_value=18, max_value=100, value=25, step=1)
    person_gender = st.selectbox("เพศ", model_info['categories'].get('person_gender', ['male', 'female']))
    person_education = st.selectbox("ระดับการศึกษา", model_info['categories'].get('person_education', ['High School', 'Bachelor', 'Master', 'Associate', 'Doctorate']))
    person_income = st.number_input("รายได้ต่อปี (บาท)", min_value=0, value=50000, step=1000)
    person_emp_exp = st.number_input("ประสบการณ์ทำงาน (ปี)", min_value=0, max_value=50, value=0, step=1)
    person_home_ownership = st.selectbox("สถานะที่อยู่อาศัย", model_info['categories'].get('person_home_ownership', ['RENT', 'OWN', 'MORTGAGE', 'OTHER']))

with col2:
    st.subheader("💵 ข้อมูลการกู้")
    loan_amnt = st.number_input("จำนวนเงินกู้ (บาท)", min_value=0, value=10000, step=1000)
    loan_intent = st.selectbox("วัตถุประสงค์", model_info['categories'].get('loan_intent', ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'DEBTCONSOLIDATION', 'HOMEIMPROVEMENT']))
    loan_int_rate = st.number_input("อัตราดอกเบี้ย (%)", min_value=0.0, max_value=50.0, value=10.0, step=0.1)
    loan_percent_income = st.number_input("อัตราส่วนหนี้ต่อรายได้", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
    cb_person_cred_hist_length = st.number_input("ประวัติเครดิต (ปี)", min_value=0, max_value=50, value=3, step=1)
    credit_score = st.number_input("คะแนนเครดิต", min_value=300, max_value=900, value=650, step=1)
    previous_loan_defaults_on_file = st.number_input("จำนวนการผิดนัดชำระก่อนหน้านี้", min_value=0, max_value=10, value=0, step=1)

# ปุ่มทำนาย
st.markdown("---")
if st.button("🔮 ทำนายผล", type="primary", use_container_width=True):
    # สร้าง DataFrame
    input_data = pd.DataFrame({
        'person_age': [person_age],
        'person_gender': [person_gender],
        'person_education': [person_education],
        'person_income': [person_income],
        'person_emp_exp': [person_emp_exp],
        'person_home_ownership': [person_home_ownership],
        'loan_amnt': [loan_amnt],
        'loan_intent': [loan_intent],
        'loan_int_rate': [loan_int_rate],
        'loan_percent_income': [loan_percent_income],
        'cb_person_cred_hist_length': [cb_person_cred_hist_length],
        'credit_score': [credit_score],
        'previous_loan_defaults_on_file': [previous_loan_defaults_on_file]
    })
    
    try:
        # ทำนาย
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        
        # แสดงผล
        st.markdown("---")
        col_result1, col_result2 = st.columns(2)
        
        with col_result1:
            if prediction == 1:
                st.success("✅ **ผลการทำนาย: อนุมัติ**")
                st.metric("ความน่าจะเป็นที่จะอนุมัติ", f"{probability[1]*100:.2f}%")
            else:
                st.error("❌ **ผลการทำนาย: ไม่อนุมัติ**")
                st.metric("ความน่าจะเป็นที่จะอนุมัติ", f"{probability[1]*100:.2f}%")
        
        with col_result2:
            st.subheader("📋 ข้อมูลที่กรอก")
            st.dataframe(input_data.T.rename(columns={0: 'ค่า'}), use_container_width=True)
        
        # คำแนะนำ
        st.markdown("---")
        st.subheader("💡 คำแนะนำ")
        if probability[1] < 0.5:
            st.warning("""
            **วิธีเพิ่มโอกาสการอนุมัติ:**
            - 📈 เพิ่มคะแนนเครดิต (Credit Score)
            - 💰 ลดจำนวนเงินกู้
            - 📉 ลดอัตราส่วนหนี้ต่อรายได้
            - ⏰ เพิ่มประวัติเครดิต
            - ❌ ลดการผิดนัดชำระก่อนหน้านี้
            """)
        else:
            st.info("✅ ข้อมูลของคุณอยู่ในเกณฑ์ดี มีโอกาสสูงที่จะได้รับการอนุมัติ!")
    
    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาดในการทำนาย: {e}")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "<small>Developed with ❤️ using Streamlit + SVM</small>"
    "</div>", 
    unsafe_allow_html=True
)