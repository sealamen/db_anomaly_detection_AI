import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
import joblib

# CSV 불러오기
df = pd.read_csv("./csv_files/normal.csv")

# 숫자형 데이터만 선택, time 컬럼은 제외
df_numeric = df.select_dtypes(include=['float64', 'int64']).drop(columns=['time'], errors='ignore')

# 데이터 스케일링
scaler = StandardScaler()
normal_scaled = scaler.fit_transform(df_numeric)

# One-Class SVM 모델 정의 및 학습
oc_svm = OneClassSVM(nu=0.01, kernel="rbf", gamma="scale")
oc_svm.fit(normal_scaled)

# 모델 + 스케일러 + 컬럼 저장
bundle = {
    "svm_model": oc_svm,
    "scaler": scaler,
    "feature_columns": df_numeric.columns.tolist()
}

joblib.dump(bundle, "../models/one_class_svm.pkl")

print("✅ One-Class SVM + Scaler 저장 완료")
