import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


def batch_detect_and_evaluate(df, model_path):
    df_numeric = df.copy()

    # 0. 데이터 현황 출력
    total = len(df)
    anomaly_count = (df['ANOMALY_YN'] == 'Y').sum()
    normal_count = (df['ANOMALY_YN'] == 'N').sum()
    print("==== 테스트 데이터 현황 ====")
    print(f"전체: {total}, 정상(N): {normal_count}, 이상(Y): {anomaly_count}\n")


    # 모델별 예측 생성 (detect_anomaly 내용 벡터화)
    # 1. SVM
    svm_bundle = joblib.load(model_path + "one_class_svm.pkl")
    oc_svm = svm_bundle["svm_model"]
    svm_scaler = svm_bundle["scaler"]
    svm_features = svm_bundle["feature_columns"]
    df_numeric = df_numeric.reindex(columns=df_numeric.columns.union(svm_features), fill_value=0)
    svm_scaled = svm_scaler.transform(df_numeric[svm_features])
    df_numeric["SVM_Pred"] = np.where(oc_svm.predict(svm_scaled) == -1, 1, 0)

    # 2. Isolation Forest
    if_bundle = joblib.load(model_path + "isolation_forest.pkl")
    iso_forest = if_bundle["iso_model"]
    if_features = if_bundle["feature_columns"]
    df_numeric["IF_Pred"] = np.where(iso_forest.predict(df_numeric[if_features]) == -1, 1, 0)

    # 3. AutoEncoder
    autoencoder = load_model(model_path + "autoencoder.h5", compile=False)
    ae_bundle = joblib.load(model_path + "autoencoder_scaler_columns.pkl")
    ae_scaler = ae_bundle["scaler"]
    ae_features = ae_bundle["feature_columns"]
    df_numeric = df_numeric.reindex(columns=df_numeric.columns.union(ae_features), fill_value=0)
    ae_scaled = ae_scaler.transform(df_numeric[ae_features])
    recon = autoencoder.predict(ae_scaled, verbose=0)
    mse = np.mean(np.square(ae_scaled - recon), axis=1)
    recon_thresh = mse.mean() + 3 * mse.std()
    df_numeric["AE_Pred"] = (mse > recon_thresh).astype(int)

    # 4. 최종 Alert
    df_numeric['Final_Alert'] = df_numeric.apply(final_alert, axis=1)

    # 5. 평가
    results = []

    y_true = df['ANOMALY_YN'].map({'N': 0, 'Y': 1})
    results.append(evaluate_model(y_true, df_numeric['SVM_Pred'], "One-class SVM"))
    results.append(evaluate_model(y_true, df_numeric['IF_Pred'], "Isolation Forest"))
    results.append(evaluate_model(y_true, df_numeric['AE_Pred'], "AutoEncoder"))
    results.append(evaluate_model(y_true, df_numeric['Final_Alert'], "Final Alert"))

    # 요약표 DataFrame 반환
    results_df = pd.DataFrame(results)

    # 요약표 출력 시 생략되지 않도록 설정
    pd.set_option("display.max_columns", None)  # 컬럼 모두 출력
    pd.set_option("display.max_rows", None)  # 행 모두 출력
    pd.set_option("display.width", None)  # 한 줄에 다 보여주기

    print("==== 성능 요약표 ====")
    print(results_df)

    return df_numeric, results_df


def final_alert(row):
    votes = row[['IF_Pred', 'SVM_Pred', 'AE_Pred']].sum()
    return 1 if votes >= 2 else 0


def evaluate_model(y_true, y_pred, model_name="Model"):
    print(f"\n[모델 성능 평가 시작] {model_name}")
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}\n")
    return {
        "Model": model_name,
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "Accuracy": accuracy, "Precision": precision,
        "Recall": recall, "F1": f1
    }


# 모델을 평가할 데이터 입력, ANOMALY_YN 컬럼이 포함되어야 평가 가능
data_path = "C:\\OCI\\data\\anomaly_40000_default_2_percent.csv"
model_path = "C:\\OCI\\repository\\db_anomaly_detection_AI\\models\\"

df_test = pd.read_csv(data_path)
print(df_test.info())
batch_detect_and_evaluate(df_test, model_path)