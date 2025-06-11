import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np

st.title("Prediksi Risiko CKD (Chronic Kidney Disease) dengan XAI")

# Load model
try:
    model = joblib.load("model/model_ckd")
    explainer = shap.Explainer(model)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Input data dari user
with st.form("input_form"):
    hemoglobin = st.slider("Hemoglobin", min_value=7.9, max_value=16.8, value=10.0)
    albumin = st.slider("Albumin", min_value=0.0, max_value=5.0, value=3.0, step=1.0)
    serum_creatinine = st.slider("Serum Creatinine", min_value=0.33, max_value=4.50, value=2.0, step= 0.01)
    rc = st.slider("Red Blood Cells", min_value=1.13, max_value=2.20, value=1.50, step=0.01)
    pcv = st.slider("Packed Cell Volume", min_value=24.0, max_value=52.0, value=38.0)
    sg = st.slider("Specific Gravity", min_value=1.005, max_value=1.025, value=1.01, step=0.001, format="%.3f")
    bgr = st.slider("Blood Glucose Random", min_value=3.13, max_value=6.20, value=5.00)
    bu = st.slider("Blood Urea", min_value=0.91, max_value=6.00, value=3.00)
    dm = st.selectbox("Diabetes Mellitus", ('Yes', 'No'))
    htn = st.selectbox("Hypertension", ('Yes', 'No'))
    
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

        # SHAP Explanation
        st.subheader("📌 Penjelasan Model (SHAP)")

        # Hitung SHAP values
        shap_values = explainer(input_df)
        
        # Coba berbagai cara mengakses SHAP values
        try:
            # Metode 1: Untuk model binary classification
            if hasattr(shap_values, 'values') and len(shap_values.values.shape) == 3:
                # Shape: (n_samples, n_features, n_classes)
                shap_vals = shap_values.values[0, :, 1]  # Class 1 (CKD)
            elif hasattr(shap_values, 'values') and len(shap_values.values.shape) == 2:
                # Shape: (n_samples, n_features)
                shap_vals = shap_values.values[0, :]
            else:
                # Fallback: gunakan TreeExplainer khusus untuk Decision Tree
                explainer_tree = shap.TreeExplainer(model)
                shap_values_tree = explainer_tree.shap_values(input_df)
                
                if isinstance(shap_values_tree, list):
                    # Binary classification returns list of arrays
                    shap_vals = shap_values_tree[1][0]  # Class 1 values
                else:
                    shap_vals = shap_values_tree[0]
                    
        except Exception as e:
            st.error(f"Error calculating SHAP values: {e}")
            # Fallback sederhana tanpa SHAP
            st.write("Menggunakan feature importance dari model sebagai alternatif:")
            feature_importance = model.feature_importances_
            feature_names = input_df.columns.tolist()
            
            importance_df = pd.DataFrame({
                'Fitur': feature_names,
                'Nilai Fitur': input_df.iloc[0].values,
                'Feature Importance': feature_importance,
                'Persentase Pengaruh (%)': (feature_importance / feature_importance.sum() * 100).round(2)
            })
            
            top3 = importance_df.sort_values(by='Feature Importance', ascending=False).head(3)
            st.write("**🔝 3 Fitur yang Paling Penting (Feature Importance):**")
            st.table(top3)
            st.stop()

        # Buat DataFrame dari SHAP values dan fitur
        feature_names = input_df.columns.tolist()
        feature_values = input_df.iloc[0].values.tolist()
        
        # Pastikan semua array memiliki panjang yang sama
        if len(feature_names) != len(feature_values) or len(feature_names) != len(shap_vals):
            st.error(f"Length mismatch - Features: {len(feature_names)}, Values: {len(feature_values)}, SHAP: {len(shap_vals)}")
            st.stop()
        
        shap_df = pd.DataFrame({
            'Fitur': feature_names,
            'Nilai Fitur': feature_values,
            'SHAP Value': shap_vals
        })

        # Hitung kontribusi absolut dan persentase kontribusi
        shap_df['Kontribusi Absolut'] = shap_df['SHAP Value'].abs()
        total_kontribusi = shap_df['Kontribusi Absolut'].sum()
        
        if total_kontribusi > 0:
            shap_df['Persentase Pengaruh (%)'] = (shap_df['Kontribusi Absolut'] / total_kontribusi * 100).round(2)
        else:
            shap_df['Persentase Pengaruh (%)'] = 0

        # Ambil 3 teratas
        top3 = shap_df.sort_values(by='Kontribusi Absolut', ascending=False).head(3)

        # Tampilkan ke Streamlit
        st.write("**🔝 3 Fitur yang Paling Mempengaruhi Prediksi CKD:**")
        st.table(top3[['Fitur', 'Nilai Fitur', 'SHAP Value', 'Persentase Pengaruh (%)']])

        # Visualisasi SHAP (dengan error handling)
        try:
            st.write("**📊 SHAP Waterfall Plot:**")
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Coba buat waterfall plot
            if hasattr(shap_values, 'values'):
                # Gunakan shap_values object
                if len(shap_values.values.shape) == 3:
                    shap_single = shap_values[0, :, 1]
                else:
                    shap_single = shap_values[0]
            else:
                # Buat manual explanation object
                shap_single = shap.Explanation(
                    values=shap_vals,
                    base_values=explainer.expected_value[1] if hasattr(explainer.expected_value, '__getitem__') else explainer.expected_value,
                    data=feature_values
                )
            
            shap.plots.waterfall(shap_single, max_display=5, show=False)
            st.pyplot(fig)
            plt.close()
            
        except Exception as e:
            st.warning(f"Tidak dapat menampilkan waterfall plot: {e}")
            
            # Alternative: Bar plot sederhana
            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                top_features = shap_df.sort_values(by='SHAP Value', key=abs, ascending=False).head(5)
                
                colors = ['red' if x < 0 else 'blue' for x in top_features['SHAP Value']]
                bars = ax.barh(top_features['Fitur'], top_features['SHAP Value'], color=colors, alpha=0.7)
                
                ax.set_xlabel('SHAP Value')
                ax.set_title('Top 5 Feature Contributions (SHAP Values)')
                ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                
                # Add value labels on bars
                for bar, value in zip(bars, top_features['SHAP Value']):
                    width = bar.get_width()
                    ax.text(width + (0.01 if width >= 0 else -0.01), bar.get_y() + bar.get_height()/2, 
                           f'{value:.3f}', ha='left' if width >= 0 else 'right', va='center')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
            except Exception as e2:
                st.error(f"Error creating alternative plot: {e2}")

    except Exception as e:
        st.error(f"Error during prediction or explanation: {e}")
        st.write("Silakan coba lagi atau periksa model dan data input.")