import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

st.title("Prediksi Risiko CKD (Chronic Kidney Disease) dengan XAI")

model = joblib.load("model_ckd.pkl")

explainer = shap.TreeExplainer(model)

# Input data dari user
with st.form("input_form"):
    hemoglobin = st.slider("Hemoglobin", min_value=7.9, max_value=16.8, value=10.0)
    albumin = st.slider("Albumin", min_value=0.0, max_value=5.0, value=3.0, step=1.0)
    serum_creatinine = st.slider("Serum Creatinine", value=2.0)
    rc = st.slider("Red Blood Cells", min_value=1.13, max_value=2.20, value=1.50, step=0.01)
    pcv = st.slider("Packed Cell Volume", min_value=24.0, max_value=52.0, value=38.0) 
    sg = st.slider("Specific Gravity", min_value=1.005, max_value=1.025, value=1.01, step=0.001, format="%.3f")
    bgr = st.slider("Blood Glucose Random", min_value=3.13, max_value=6.20, value=5.00)
    bu = st.slider("Blood Urea", min_value=0.91, max_value=6.00, value=3.00)
    dm = st.selectbox("Diabetes Mellitus", ('Yes', 'No'))
    dm = dm.lower()
    htn = st.selectbox("Hypertension", ('Yes', 'No'))
    htn = htn.lower()
    dm = 1 if dm == 'yes' else 0
    htn = 1 if htn == 'yes' else 0

    submitted = st.form_submit_button("Prediksi")

if submitted:
    input_df = pd.DataFrame([{
        'hemo': hemoglobin,
        'pcv': pcv,
        'sg': sg,
        'rc': rc,
        'bgr': bgr,
        'bu': bu,
        'al': albumin,
        'dm': dm,
        'sc': serum_creatinine,
        'htn': htn,
        # 'classification': 0.0  # atau abaikan kalau bukan inputan user
    }])

    # Prediksi
    proba = model.predict_proba(input_df)[0][1]
    proba_percent = round(proba * 100, 2)
    prediction = model.predict(input_df)[0]
    status = "✅ Tidak Terkena CKD" if prediction == 0 else "❌ Terkena CKD"

    # Kategori risiko
    if proba_percent < 40:
        risk_level = "Rendah"
    elif 40 <= proba_percent < 70:
        risk_level = "Sedang"
    else:
        risk_level = "Tinggi"

    # Tampilkan hasil
    st.subheader("🔍 Hasil Prediksi")
    st.write(f"**Status:** {status}")
    st.write(f"**Probabilitas terkena CKD:** {proba_percent}%")
    st.write(f"**Kategori Risiko:** {risk_level}")


    import sklearn
    print(sklearn.__version__)
    # SHAP Explanation
    st.subheader("📌 Penjelasan Model (SHAP)")
    shap_values = explainer.shap_values(input_df)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    shap.initjs()
    shap.force_plot(explainer.expected_value[1], shap_values[1], input_df, matplotlib=True, show=False)
    st.pyplot(bbox_inches='tight')