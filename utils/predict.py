import shap
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# Load model & feature
model = joblib.load('model/model_ckd.pkl')
feature_names = ['hemo', 'pcv', 'sg', 'rc', 'bgr', 'bu', 'al', 'dm', 'sc', 'htn']

def prediksi_dan_penjelasan(data_dict):
    input_array = np.array([data_dict[f] for f in feature_names]).reshape(1, -1)
    
    # Prediksi probabilitas
    prob = model.predict_proba(input_array)[0][1]
    risiko = "Rendah" if prob < 0.34 else "Sedang" if prob < 0.67 else "Tinggi"

    # Gunakan SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_array)
    
    # Ambil top fitur penyebab (nilai absolut tertinggi)
    shap_df = pd.DataFrame({
        'Fitur': feature_names,
        'SHAP Value': shap_values[1][0],
        'Nilai Input': input_array[0]
    })
    shap_df['Kontribusi'] = shap_df['SHAP Value'].abs()
    shap_df = shap_df.sort_values(by='Kontribusi', ascending=False)

    return prob, risiko, shap_df.head(3)  # 3 fitur utama penyebab