import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# ===============================
# 1. 데이터 불러오기
# ===============================
df = pd.read_csv("./anomaly.csv")

# 숫자형 컬럼만 선택 (time 제거)
df_numeric = df.select_dtypes(include=['float64', 'int64']).drop(columns=['time'], errors='ignore')

# 원래 feature 컬럼 보관
feature_cols = df_numeric.columns.tolist()

# ===============================
# 2. One-Class SVM 불러오기 & 예측
# ===============================
svm_bundle = joblib.load("ocsvm_scaler.pkl")
oc_svm = svm_bundle["svm_model"]
svm_scaler = svm_bundle["scaler"]

svm_scaled = svm_scaler.transform(df_numeric[feature_cols])
svm_preds = oc_svm.predict(svm_scaled)
df_numeric["SVM_Pred"] = np.where(svm_preds == -1, 1, 0)

# ===============================
# 3. Isolation Forest 불러오기 & 예측
# ===============================
iso_forest = joblib.load("isolation_forest.pkl")

if_preds = iso_forest.predict(df_numeric[feature_cols])  # feature만 사용
df_numeric["IF_Pred"] = np.where(if_preds == -1, 1, 0)

# ===============================
# 4. Autoencoder 불러오기 & 예측
# ===============================
autoencoder = load_model("autoencoder.h5",compile=False)
ae_scaler = joblib.load("autoencoder_scaler.pkl")

ae_scaled = ae_scaler.transform(df_numeric[feature_cols])
recon = autoencoder.predict(ae_scaled, verbose=0)
mse = np.mean(np.square(ae_scaled - recon), axis=1)

# 임계값: 평균 + 3*표준편차
train_recon = autoencoder.predict(ae_scaled, verbose=0)
train_mse = np.mean(np.square(ae_scaled - train_recon), axis=1)
recon_thresh = train_mse.mean() + 3 * train_mse.std()

df_numeric["AE_Pred"] = (mse > recon_thresh).astype(int)

# ===============================
# 5. Final Alert (과반수 투표)
# ===============================
def final_alert(row):
    votes = row[['IF_Pred', 'SVM_Pred', 'AE_Pred']].sum()
    return 1 if votes >= 2 else 0

df_numeric['Final_Alert'] = df_numeric.apply(final_alert, axis=1)

# ===============================
# 6. 결과 저장
# ===============================
print("✅ 최종 결과 샘플:")
print(df_numeric[['IF_Pred', 'SVM_Pred', 'AE_Pred', 'Final_Alert']].head())

df_numeric.to_csv("./evaluation_results.csv", index=False)
print("📂 evaluation_results.csv 저장 완료")
