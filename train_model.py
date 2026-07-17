import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

# ==============================
# 1. โหลดข้อมูล
# ==============================
print("=" * 50)
print("📂 Loading data...")
df = pd.read_csv('loan_data.csv')
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(df.head())

# ==============================
# 2. สำรวจข้อมูลเบื้องต้น
# ==============================
print("\n" + "=" * 50)
print("🔍 Data Info:")
print(df.info())
print("\n📊 Missing values:")
print(df.isnull().sum())
print("\n📈 Target distribution:")
print(df['approved'].value_counts())

# ==============================
# 3. Preprocessing
# ==============================
print("\n" + "=" * 50)
print("🔧 Preprocessing...")

# แยก features และ target
X = df.drop('approved', axis=1)
y = df['approved'].map({'Yes': 1, 'No': 0})  # แปลงเป็น 0/1

# แยก categorical และ numerical features
categorical_cols = ['gender', 'education', 'home_ownership', 'loan_purpose']
numerical_cols = ['age', 'income', 'num_children', 'loan_amount', 
                  'interest_rate', 'credit_score', 'loan_term', 'credit_limit']

# ตรวจสอบ missing values และเติมค่า
for col in numerical_cols:
    if X[col].isnull().sum() > 0:
        X[col] = X[col].fillna(X[col].median())

for col in categorical_cols:
    if X[col].isnull().sum() > 0:
        X[col] = X[col].fillna(X[col].mode()[0])

# ==============================
# 4. Split Data
# ==============================
print("\n" + "=" * 50)
print("🔀 Splitting data (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# ==============================
# 5. สร้าง Pipeline (Preprocessing + SVM)
# ==============================
print("\n" + "=" * 50)
print("🏗️  Building SVM Pipeline...")

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Preprocessor: OneHotEncode categorical + Scale numerical
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# SVM Pipeline
svm_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC(kernel='rbf', C=1.0, probability=True, random_state=42))
])

# ==============================
# 6. Train Model
# ==============================
print("\n" + "=" * 50)
print("🚀 Training SVM model...")
svm_pipeline.fit(X_train, y_train)
print("✅ Training completed!")

# ==============================
# 7. Evaluate Model
# ==============================
print("\n" + "=" * 50)
print("📊 Model Evaluation:")
y_pred = svm_pipeline.predict(X_test)
y_prob = svm_pipeline.predict_proba(X_test)[:, 1]

print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Approved', 'Approved']))

# ==============================
# 8. Save Model และ Preprocessor
# ==============================
print("\n" + "=" * 50)
print("💾 Saving model...")
joblib.dump(svm_pipeline, 'svm_loan_model.pkl')

# บันทึก column names สำหรับใช้ใน Streamlit
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
print("✅ Model saved as 'svm_loan_model.pkl'")
print("✅ Model info saved as 'model_info.pkl'")
print("\n🎉 Done! Ready for Streamlit deployment.")