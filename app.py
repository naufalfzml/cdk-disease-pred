import streamlit as st
from utils.predict import prediksi_dan_penjelasan

st.title("Prediksi Risiko CKD (Chronic Kidney Disease) dengan XAI")

# Input data dari user
hemoglobin = st.silder("Hemoglobin", min_value=7.9, max_value=16.8, value=10.0)
albumin = st.silder("Albumin", min_value=0.0, max_value=5.0, value=3.0, step=1.0)
serum_creatinine = st.silder("Serum Creatinine", value=2.0)
rc = st.silder("Red Blood Cells", min_value=1.13, max_value=2.20, value=1.50, step=0.01)
pcv = st.silder("Packed Cell Volume", min_value=24.0, max_value=52.0, value=38.0) 
sg = st.silder("Specific Gravity", min_value=1.005, max_value=1.025, value=1.01)
bgr = st.silder("Blood Glucose Random", min_value=3.13, max_value=6.20, value=5.00)
bu = st.silder("Blood Urea", min_value=0.91, max_value=6.00, value=3.00)
dm = st.selectbox("Diabetes Mellitus", ('Yes', 'No'))
dm = dm.lower()
htn = st.selectbox("Hypertension", ('Yes', 'No'))
htn = htn.lower()

if st.button("Prediksi"):
    data = {
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
        'classification': 0.0  # atau abaikan kalau bukan inputan user
    }

    prob, kategori, penyebab_df = prediksi_dan_penjelasan(data)

    st.write(f"### Probabilitas terkena CKD: {prob*100:.2f}%")
    st.write(f"### Tingkat Risiko: {kategori}")
    st.markdown("### Penyebab Utama Prediksi:")

    for i, row in penyebab_df.iterrows():
        arah = "meningkatkan risiko" if row['SHAP Value'] > 0 else "menurunkan risiko"
        st.markdown(f"- **{row['Fitur']}** = {row['Nilai Input']} → {arah} ({row['SHAP Value']:.3f})")