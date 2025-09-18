import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

# CSV 불러오기
df = pd.read_csv("./csv_files/train_anomaly.csv")

# 숫자형 컬럼만 추출
df_numeric = df.select_dtypes(include=['float64','int64']).drop(columns=['time'], errors='ignore')

# 학습 데이터 분리
normal_train, _ = train_test_split(df_numeric, test_size=0.1, random_state=42)

# Isolation Forest 학습
iso_forest = IsolationForest(contamination=0.01, random_state=42)
iso_forest.fit(normal_train)

# 저장
joblib.dump(iso_forest, "../models/isolation_forest.pkl")
print("✅ Isolation Forest 모델 저장 완료")
