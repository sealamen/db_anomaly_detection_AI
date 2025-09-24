import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest

# CSV 불러오기
df = pd.read_csv("./csv_files/normal.csv")

# 숫자형 컬럼만 추출
df_numeric = df.select_dtypes(include=['float64','int64']).drop(columns=['time'], errors='ignore')

# Isolation Forest 학습
iso_forest = IsolationForest(
    n_estimators=200,
    max_samples=1.0,
    contamination=0.05,
    max_features=1.0,
    random_state=42
)
iso_forest.fit(df_numeric)

# 모델 + 컬럼 저장
bundle = {
    "iso_model": iso_forest,
    "feature_columns": df_numeric.columns.tolist()
}

# 저장
joblib.dump(bundle, "../models/isolation_forest.pkl")
print("✅ Isolation Forest 모델 저장 완료")
