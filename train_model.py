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

# 1. โหลดข้อมูล
print("\n📂 กำลังโหลดข้อมูล...")
try:
    # ลองอ่านแบบมี header ก่อน
    df = pd.read_csv('loan_data.csv')
    print(f"✅ โหลดข้อมูลสำเร็จ (มี header): {df.shape[0]} แถว, {df.shape[1]} คอลัมน์")
except:
    # ถ้าไม่มี header ให้อ่านแบบไม่มี header
    columns = ['person_age', 'person_gender', 'person_education', 'person_income', 
               'person_emp_exp', 'person_home_ownership', 'loan_amnt', 'loan_intent', 
               'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 
               'credit_score', 'previous_loan_defaults_on_file', 'loan_status']
    df = pd.read_csv('loan_data.csv', header=None, names=columns)
    print(f"✅ โหลดข้อมูลสำเร็จ (ไม่มี header): {df.shape[0]} แถว, {df.shape[1]} คอลัมน์")

print(f"\n📋 ชื่อคอลัมน์ทั้งหมด:")
print(df.columns.tolist())

# 2. ตรวจสอบข้อมูล
print(f"\n📊 ข้อมูลเบื้องต้น:")
print(df.head())
print(f"\n ข้อมูล Target (loan_status):")
print(df['loan_status'].value_counts())

# 3. แปลง loan_status เป็นตัวเลข
# ตรวจสอบว่ามีค่าอะไรบ้าง
unique_values = df['loan_status'].unique()
print(f"\n ค่าที่เป็นไปได้ใน loan_status: {unique_values}")

# แปลง Yes/No หรือ 1/0
if 'Yes' in unique_values or 'No' in unique_values:
    df['approved'] = df['loan_status'].map({'Yes': 1, 'No': 0})
elif 1 in unique_values or 0 in unique_values:
    df['approved'] = df['loan_status']
else:
    # ถ้าเป็นค่าอื่น ให้ลองแปลง
    df['approved'] = pd.to_numeric(df['loan_status'], errors='coerce')

# ลบแถวที่มีค่า NaN ใน approved
df = df.dropna(subset=['approved'])
df['approved'] = df['approved'].astype(int)

print(f"\n✅ แปลง Target สำเร็จ: {df['approved'].value_counts().to_dict()}")

# 4. แยก Features และ Target
X = df.drop(['loan_status', 'approved'], axis=1, errors='ignore')
y = df['approved']

# 5. ระบุคอลัมน์
categorical_cols = ['person_gender', 'person_education', 'person_home_ownership', 'loan_intent']
numerical_cols = ['person_age', 'person_income', 'person_emp_exp', 'loan_amnt', 
                  'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 
                  'credit_score', 'previous_loan_defaults_on_file']

# ลบคอลัมน์ที่ไม่มีอยู่จริง
categorical_cols = [col for col in categorical_cols if col in X.columns]
numerical_cols = [col for col in numerical_cols if col in X.columns]

print(f"\n📋 Categorical columns: {categorical_cols}")
print(f"📋 Numerical columns: {numerical_cols}")

# 6. Handle missing values
print("\n กำลังจัดการกับ missing values...")
for col in numerical_cols:
    if X[col].isnull().sum() > 0:
        print(f"   - เติมค่า NaN ใน {col} ด้วย median")
        X[col] = X[col].fillna(X[col].median())

for col in categorical_cols:
    if X[col].isnull().sum() > 0:
        print(f"   - เติมค่า NaN ใน {col} ด้วย mode")
        X[col] = X[col].fillna(X[col].mode()[0])

# 7. สร้าง Preprocessor
print("\n️ กำลังสร้าง Preprocessor...")
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ])

# 8. สร้าง Pipeline
print("🏗️ กำลังสร้าง SVM Pipeline...")
svm_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC(kernel='rbf', C=1.0, probability=True, random_state=42))
])

# 9. Split Data
print("\n แบ่งข้อมูล Train/Test (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Train: {X_train.shape[0]} samples")
print(f"   Test: {X_test.shape[0]} samples")

# 10. Train Model
print("\n🚀 กำลังเทรนโมเดล...")
svm_pipeline.fit(X_train, y_train)
print("✅ เทรนโมเดลสำเร็จ!")

# 11. Evaluate
print("\n📊 ประเมินผลโมเดล...")
y_pred = svm_pipeline.predict(X_test)
y_prob = svm_pipeline.predict_proba(X_test)[:, 1]

print(f"\n   Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"\n   Confusion Matrix:")
print(f"   {confusion_matrix(y_test, y_pred)}")
print(f"\n   Classification Report:")
print(f"   {classification_report(y_test, y_pred, target_names=['Not Approved', 'Approved'])}")

# 12. บันทึกโมเดล
print("\n กำลังบันทึกโมเดล...")
joblib.dump(svm_pipeline, 'svm_loan_model.pkl')

# บันทึกข้อมูล categories สำหรับใช้ใน app
model_info = {
    'numerical_cols': numerical_cols,
    'categorical_cols': categorical_cols,
    'categories': {}
}

for col in categorical_cols:
    model_info['categories'][col] = df[col].unique().tolist()

joblib.dump(model_info, 'model_info.pkl')

print("✅ บันทึกไฟล์สำเร็จ:")
print("   - svm_loan_model.pkl")
print("   - model_info.pkl")
print("\n" + "=" * 60)
print("🎉 เสร็จสิ้น! พร้อมใช้งานแล้ว")
print("=" * 60)