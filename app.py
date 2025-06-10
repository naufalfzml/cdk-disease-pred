import streamlit as st
from utils.predict import prediksi_pasien

st.title("Prediksi Risiko CKD (Chronic Kidney Disease) dengan XAI")

# Input data dari user
hemoglobin = st.number_input("Hemoglobin", min_value=7.9, max_value=16.8, value=10.0)
albumin = st.number_input("Albumin", min_value=0.0, max_value=5.0, value=3.0)
serum_creatinine = st.number_input("Serum Creatinine", value=2.0)


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
