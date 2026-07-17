import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("🚀 เริ่มต้นการเทรนโมเดล SVM")
print("=" * 60)

# 1. โหลดข้อมูล
print("\n📂 กำลังโหลดข้อมูล...")
df = pd.read_csv('loan_data.csv')
print(f"✅ โหลดข้อมูลสำเร็จ: {df.shape[0]} แถว, {df.shape[1]} คอลัมน์")

# 2. กำหนดชื่อคอลัมน์
df.columns = ['age', 'gender', 'education', 'income', 'num_children', 
              'home_ownership', 'loan_amount', 'loan_purpose', 'interest_rate', 
              'loan_to_income', 'loan_term', 'credit_score', 'approved']

print(f"\n📊 ข้อมูล Target:")
print(df['approved'].value_counts())

# 3. แยก Features และ Target
X = df.drop('approved', axis=1)
y = df['approved'].map({'Yes': 1, 'No': 0})

# 4. ระบุคอลัมน์
categorical_cols = ['gender', 'education', 'home_ownership', 'loan_purpose']
numerical_cols = ['age', 'income', 'num_children', 'loan_amount', 
                  'interest_rate', 'loan_to_income', 'loan_term', 'credit_score']

# 5. สร้าง Preprocessor
print("\n กำลังสร้าง Preprocessor...")
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ])

# 6. สร้าง Pipeline
print("🏗️  กำลังสร้าง SVM Pipeline...")
svm_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC(kernel='rbf', C=1.0, probability=True, random_state=42))
])

# 7. Split Data
print("\n แบ่งข้อมูล Train/Test (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Train: {X_train.shape[0]} samples")
print(f"   Test: {X_test.shape[0]} samples")

# 8. Train Model
print("\n🚀 กำลังเทรนโมเดล...")
svm_pipeline.fit(X_train, y_train)
print("✅ เทรนโมเดลสำเร็จ!")

# 9. Evaluate
print("\n ประเมินผลโมเดล...")
y_pred = svm_pipeline.predict(X_test)
y_prob = svm_pipeline.predict_proba(X_test)[:, 1]

print(f"\n   Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"\n   Confusion Matrix:")
print(f"   {confusion_matrix(y_test, y_pred)}")
print(f"\n   Classification Report:")
print(f"   {classification_report(y_test, y_pred, target_names=['Not Approved', 'Approved'])}")

# 10. บันทึกโมเดล
print("\n💾 กำลังบันทึกโมเดล...")
joblib.dump(svm_pipeline, 'svm_loan_model.pkl')

# บันทึกข้อมูล categories สำหรับใช้ใน app
model_info = {
    'numerical_cols': numerical_cols,
    'categorical_cols': categorical_cols,
    'categories': {
        'gender': df['gender'].unique().tolist(),
        'education': df['education'].unique().tolist(),
        'home_ownership': df['home_ownership'].unique().tolist(),
        'loan_purpose': df['loan_purpose'].unique().tolist()
    }
}
joblib.dump(model_info, 'model_info.pkl')

print("✅ บันทึกไฟล์สำเร็จ:")
print("   - svm_loan_model.pkl")
print("   - model_info.pkl")
print("\n" + "=" * 60)
print(" เสร็จสิ้น! พร้อมใช้งานแล้ว")
print("=" * 60)