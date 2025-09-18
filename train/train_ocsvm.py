import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
import joblib

# CSV 불러오기
df = pd.read_csv("./csv_files/train_anomaly.csv")

# 숫자형 데이터만 선택, time 컬럼은 제외
df_numeric = df.select_dtypes(include=['float64', 'int64']).drop(columns=['time'], errors='ignore')

# 학습용 데이터 분리
normal_train, _ = train_test_split(df_numeric, test_size=0.1, random_state=42)

# 데이터 스케일링
scaler = StandardScaler()
normal_scaled = scaler.fit_transform(normal_train)

# One-Class SVM 모델 정의 및 학습
oc_svm = OneClassSVM(nu=0.01, kernel="rbf", gamma="scale")
oc_svm.fit(normal_scaled)

# 모델 + 스케일러 저장
bundle = {
    "svm_model": oc_svm,
    "scaler": scaler
}

joblib.dump(bundle, "../models/ocsvm_scaler.pkl")

print("✅ One-Class SVM + Scaler 저장 완료")
