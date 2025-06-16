import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Cek Ginjal+",
                   layout="wide")

st.markdown("""
    <div style="text-align: center;">
        <h1>Cek Ginjal+ : Aplikasi Prediksi Penyakit Gagal Ginjal Kronis dengan XAI dan Probabilitas </h1>
        <p>Created by: Kelompok 1 Kelas B Informatika UNS 2023</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <div style="text-align: center;">
        <img src="https://www.medicalindependent.ie/wp-content/uploads/2024/09/CKD-image.png" alt="Gambar Ginjal" style="width: 400px;">
        <br> </br>
    </div>
""", unsafe_allow_html=True)


with st.expander("Tentang Aplikasi Ini"):
            st.write("""Aplikasi ini dibuat untuk mendeteksi risiko penyakit gagal ginjal kronis dengan menggunakan model prediksi berbasis Machine Learning. Algoritma yang digunakan adalah Random Forest.
                     Dengan menganalisis data kesehatan pengguna yang sudah diinputkan, seperti hemoglobin, riwayat hipertensi, dan riwayat kesehatan lainnya. Aplikasi ini akan memprediksi ada atau tidaknya risiko penyakit gagal ginjal kronis pada tubuh pengguna.
                    Data diperoleh dari [Chronic Kidney Disease](https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease) by UCIML. 
""")
            
with st.expander("Penjelasan Feature"):
    st.markdown("""
    **1. Kadar Hemoglobin dalam gms (hemo)**

    **2. Tingkat Kandungan Albumin dalam Urin (al)**
                
    **3. Kadar Kreatinin dalam Serum dalam mgs/dl (sc)**
                
    **4. Jumlah Sel Darah Merah dalam millions/cmm (rc)**
                
    **5. Persentase Volume Darah yang ditempati oleh Sel Darah Merah (pcv)**
                
    **6. Berat Jenis Urin (sg)**
                         
    **7. Kadar Gula Darah Acak dalam mgs/dl  (bgr)**
                
    **8. Kadar Urea dalam Darah dalam mgs/dl (bu)**   
                
    **9. Riwayat Diabetes Mellitus (dm):**
    - Yes = Mempunyai riwayat Diabetes Mellitus
    - No = Tidak mempunyai riwayat Diabetes Mellitus   
                
    **10. Riwayat Hipertensi (htn):**
    - Yes = Mempunyai riwayat Hipertensi
    - No = Tidak mempunyai riwayat Hipertensi
    """)

model = joblib.load("model/model_randomforest_ckd.pkl")
explainer = shap.TreeExplainer(model)


st.header('User Input Features:')
uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file is not None:
     input_df = pd.read_csv(uploaded_file)
else:
# Input data dari user
    with st.form("input_form"):
        hemoglobin = st.slider("Hemoglobin (hemo)", min_value=7.9, max_value=16.8, value=10.0)
        albumin = st.slider("Albumin (al)", min_value=0.0, max_value=5.0, value=3.0, step=1.0)
        serum_creatinine = st.slider("Serum Creatinine (sc)", min_value=0.33, max_value=4.50, value=2.0, step= 0.01)
        rc = st.slider("Red Blood Cells (rc)", min_value=1.13, max_value=2.20, value=1.50, step=0.01)
        pcv = st.slider("Packed Cell Volume (pcv)", min_value=24.0, max_value=52.0, value=38.0)
        sg = st.slider("Specific Gravity (sg)", min_value=1.005, max_value=1.025, value=1.01, step=0.001, format="%.3f")
        bgr = st.slider("Blood Glucose Random (bgr)", min_value=3.13, max_value=6.20, value=5.00)
        bu = st.slider("Blood Urea (bu)", min_value=0.91, max_value=6.00, value=3.00)
        dm = st.selectbox("Diabetes Mellitus (dm)", ('Yes', 'No'))
        htn = st.selectbox("Hipertensi (htn)", ('Yes', 'No'))
        
        dm = 1 if dm.lower() == 'yes' else 0
        htn = 1 if htn.lower() == 'yes' else 0

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
        }])

        try:
            # Prediksi
            proba = model.predict_proba(input_df)[0][1]
            proba_percent = round(proba * 100, 2)
            prediction = model.predict(input_df)[0]
            status = "Kamu Tidak Terdeteksi Penyakit Gagal Ginjal Kronis" if prediction == 0 else "Kamu Terdeteksi Penyakit Gagal Ginjal Kronis"

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

            # SHAP Explanation
            st.subheader("📌 Penjelasan Model (SHAP)")

            # PERBAIKAN: Hitung SHAP values dengan error handling yang lebih baik
            try:
                shap_values = explainer(input_df)
                
                # Extract SHAP values berdasarkan tipe output
                if hasattr(shap_values, 'values'):
                    if len(shap_values.values.shape) == 3:
                        # Binary classification dengan 2 kelas
                        shap_vals = shap_values.values[0, :, 1]  # Kelas CKD (1)
                    elif len(shap_values.values.shape) == 2:
                        # Output langsung untuk satu kelas
                        shap_vals = shap_values.values[0, :]
                    else:
                        raise ValueError("Format SHAP values tidak dikenali")
                else:
                    # Jika tidak ada atribut values, coba akses langsung
                    shap_vals = shap_values[0]
                    
            except Exception as shap_error:
                st.warning(f"Error dengan SHAP Explainer: {shap_error}")
                
                # FALLBACK: Gunakan Linear/Kernel explainer sebagai alternatif
                try:
                    st.write("Mencoba metode alternatif...")
                    
                    # Untuk model logistic regression, gunakan LinearExplainer
                    if hasattr(model, 'coef_'):
                        explainer_alt = shap.LinearExplainer(model, sample_data)
                        shap_values_alt = explainer_alt.shap_values(input_df)
                        
                        if isinstance(shap_values_alt, list):
                            shap_vals = shap_values_alt[1][0]  # Kelas 1
                        else:
                            shap_vals = shap_values_alt[0]
                    else:
                        # Gunakan KernelExplainer sebagai last resort
                        explainer_kernel = shap.KernelExplainer(
                            lambda X: model.predict_proba(X)[:, 1], 
                            sample_data
                        )
                        shap_values_kernel = explainer_kernel.shap_values(input_df, nsamples=100)
                        shap_vals = shap_values_kernel[0]
                        
                except Exception as fallback_error:
                    st.error(f"Semua metode SHAP gagal: {fallback_error}")
                    
                    # FINAL FALLBACK: Gunakan feature importance jika tersedia
                    if hasattr(model, 'coef_'):
                        st.write("Menggunakan koefisien model sebagai pengganti SHAP:")
                        feature_names = input_df.columns.tolist()
                        feature_values = input_df.iloc[0].values.tolist()
                        coefficients = model.coef_[0]
                        
                        coef_df = pd.DataFrame({
                            'Fitur': feature_names,
                            'Nilai Fitur': feature_values,
                            'Koefisien': coefficients,
                            'Kontribusi': np.array(feature_values) * coefficients
                        })
                        
                        coef_df['Kontribusi Absolut'] = coef_df['Kontribusi'].abs()
                        total_kontribusi = coef_df['Kontribusi Absolut'].sum()
                        
                        if total_kontribusi > 0:
                            coef_df['Persentase Pengaruh (%)'] = (coef_df['Kontribusi Absolut'] / total_kontribusi * 100).round(2)
                        else:
                            coef_df['Persentase Pengaruh (%)'] = 0
                        
                        top3 = coef_df.sort_values(by='Kontribusi Absolut', ascending=False).head(3)
                        st.write("**🔝 3 Fitur yang Paling Mempengaruhi (Berdasarkan Koefisien):**")
                        st.table(top3[['Fitur', 'Nilai Fitur', 'Kontribusi', 'Persentase Pengaruh (%)']])
                        
                    else:
                        st.error("Model tidak memiliki koefisien yang dapat diakses")
                    
                    st.stop()

            # Jika SHAP berhasil, lanjutkan dengan analisis
            feature_names = input_df.columns.tolist()
            feature_values = input_df.iloc[0].values.tolist()
            
            # Validasi panjang array
            if len(feature_names) != len(feature_values) or len(feature_names) != len(shap_vals):
                st.error(f"Length mismatch - Features: {len(feature_names)}, Values: {len(feature_values)}, SHAP: {len(shap_vals)}")
                st.stop()
            
            shap_df = pd.DataFrame({
                'Fitur': feature_names,
                'Nilai Fitur': feature_values,
                'SHAP Value': shap_vals
            })

            # Hitung kontribusi absolut dan persentase
            shap_df['Kontribusi Absolut'] = shap_df['SHAP Value'].abs()
            total_kontribusi = shap_df['Kontribusi Absolut'].sum()
            
            if total_kontribusi > 0:
                shap_df['Persentase Pengaruh (%)'] = (shap_df['Kontribusi Absolut'] / total_kontribusi * 100).round(2)
            else:
                shap_df['Persentase Pengaruh (%)'] = 0

            # Ambil 3 teratas
            top3 = shap_df.sort_values(by='Kontribusi Absolut', ascending=False).head(3)

            # Tampilkan hasil
            st.write("**🔝 3 Fitur yang Paling Mempengaruhi Prediksi CKD:**")
            st.table(top3[['Fitur', 'Nilai Fitur', 'SHAP Value', 'Persentase Pengaruh (%)']])

            # Visualisasi SHAP
            try:
                st.write("**📊 SHAP Visualization:**")
                
                # Bar plot sederhana yang lebih reliable
                fig, ax = plt.subplots(figsize=(10, 6))
                top_features = shap_df.sort_values(by='SHAP Value', key=abs, ascending=False).head(5)
                
                colors = ['red' if x < 0 else 'blue' for x in top_features['SHAP Value']]
                bars = ax.barh(top_features['Fitur'], top_features['SHAP Value'], color=colors, alpha=0.7)
                
                ax.set_xlabel('SHAP Value')
                ax.set_title('Top 5 Feature Contributions (SHAP Values)')
                ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                
                # Add value labels
                for bar, value in zip(bars, top_features['SHAP Value']):
                    width = bar.get_width()
                    ax.text(width + (0.01 if width >= 0 else -0.01), 
                        bar.get_y() + bar.get_height()/2, 
                        f'{value:.3f}', 
                        ha='left' if width >= 0 else 'right', 
                        va='center')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
            except Exception as viz_error:
                st.warning(f"Tidak dapat menampilkan visualisasi: {viz_error}")

        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.write("Silakan coba lagi atau periksa model dan data input.")