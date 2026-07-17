import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("🚀 เริ่มต้นการเทรนโมเดล SVM")
print("=" * 60)

# 1. โหลดข้อมูล (ไฟล์ไม่มี header)
print("\n กำลังโหลดข้อมูล...")
column_names = [
    'person_age',
    'person_gender', 
    'person_education',
    'person_income',
    'person_emp_exp',
    'person_home_ownership',
    'loan_amnt',
    'loan_intent',
    'loan_int_rate',
    'loan_percent_income',
    'cb_person_cred_hist_length',
    'credit_score',
    'loan_status',
    'dummy'
]

df = pd.read_csv('loan_data.csv', header=None, names=column_names)
df = df.drop('dummy', axis=1)  # ลบคอลัมน์ที่ไม่ต้องการ
print(f"✅ โหลดข้อมูลสำเร็จ: {df.shape[0]} แถว, {df.shape[1]} คอลัมน์")

# 2. แสดงข้อมูลเบื้องต้น
print(f"\n📊 ข้อมูล Target (loan_status):")
print(df['loan_status'].value_counts())

# 3. แปลง loan_status เป็น 0/1
df['loan_status'] = df['loan_status'].map({'Yes': 1, 'No': 0})

# ลบ rows ที่มี NaN
df = df.dropna()
print(f"\n✅ ข้อมูลหลังทำความสะอาด: {df.shape[0]} แถว")

# 4. แยก Features และ Target
X = df.drop('loan_status', axis=1)
y = df['loan_status']

# 5. ระบุคอลัมน์
categorical_cols = ['person_gender', 'person_education', 'person_home_ownership', 'loan_intent']
numerical_cols = ['person_age', 'person_income', 'person_emp_exp', 'loan_amnt', 
                  'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 
                  'credit_score']

print(f"\n📋 Categorical columns: {categorical_cols}")
print(f" Numerical columns: {numerical_cols}")

# 6. สร้าง Preprocessor
print("\n️ กำลังสร้าง Preprocessor...")
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ])

# 7. สร้าง Pipeline
print("🏗️ กำลังสร้าง SVM Pipeline...")
svm_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC(kernel='rbf', C=1.0, probability=True, random_state=42))
])

# 8. Split Data
print("\n🔀 แบ่งข้อมูล Train/Test (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Train: {X_train.shape[0]} samples")
print(f"   Test: {X_test.shape[0]} samples")

# 9. Train Model
print("\n🚀 กำลังเทรนโมเดล...")
svm_pipeline.fit(X_train, y_train)
print("✅ เทรนโมเดลสำเร็จ!")

# 10. Evaluate
print("\n ประเมินผลโมเดล...")
y_pred = svm_pipeline.predict(X_test)
y_prob = svm_pipeline.predict_proba(X_test)[:, 1]

print(f"\n   Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"\n   Confusion Matrix:")
print(f"   {confusion_matrix(y_test, y_pred)}")
print(f"\n   Classification Report:")
print(f"   {classification_report(y_test, y_pred, target_names=['Not Approved', 'Approved'])}")

# 11. บันทึกโมเดล
print("\n💾 กำลังบันทึกโมเดล...")
joblib.dump(svm_pipeline, 'svm_loan_model.pkl')

# บันทึกข้อมูล categories สำหรับใช้ใน app
model_info = {
    'numerical_cols': numerical_cols,
    'categorical_cols': categorical_cols,
    'categories': {
        'person_gender': df['person_gender'].unique().tolist(),
        'person_education': df['person_education'].unique().tolist(),
        'person_home_ownership': df['person_home_ownership'].unique().tolist(),
        'loan_intent': df['loan_intent'].unique().tolist()
    }
}
joblib.dump(model_info, 'model_info.pkl')

print("✅ บันทึกไฟล์สำเร็จ:")
print("   - svm_loan_model.pkl")
print("   - model_info.pkl")
print("\n" + "=" * 60)
print("🎉 เสร็จสิ้น! พร้อมใช้งานแล้ว")
print("=" * 60)