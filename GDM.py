import streamlit as st
import pandas as pd
import numpy as np
import pickle
from catboost import CatBoostClassifier

from PIL import Image
image = Image.open('gusto.jpg')
st.image(image,use_column_width=True)

st.write("""
# Gestational Diabetes Mellitus (GDM) 
This app predicts Gestational Diabetes Mellitus at first trimester of pregnancy.
""")

st.sidebar.header('Patient Input Features')

# Collects patient input features into dataframe
def patient_input_features():
    sbp_first_antenatal = st.sidebar.number_input('Systolic Blood Pressure (mmHg)', 50.0,250.0,120.0)
    dbp_first_antenatal = st.sidebar.number_input('Diastolic Blood Pressure (mmHg)', 50.0,250.0,100.0)
    map_est_first_antenatal = (1/3)*sbp_first_antenatal + (2/3)*dbp_first_antenatal
    mother_age_recruitment = st.sidebar.number_input('Maternal Age (years)', 18.0,60.0,35.0)
    pw11_any_gdm_outcome_cat = st.sidebar.selectbox('Previous History of GDM',('0.0','1.0'))
    m_ethnicity_malay_bin = st.sidebar.selectbox('Malay Ethnicity vs Chinese/Indian Ethnicity',('0.0','1.0'))
    data = {'map_est_first_antenatal': map_est_first_antenatal,
            'mother_age_recruitment': mother_age_recruitment,
            'pw11_any_gdm_outcome_cat': pw11_any_gdm_outcome_cat,
            'm_ethnicity_malay_bin': m_ethnicity_malay_bin}
    features = pd.DataFrame(data, index=[0])
    return features
df = patient_input_features()

df = df[:1] # Selects only the first row (the patient input data)

# Displays the patient input features
st.subheader('Patient Input features :')

st.write(df)

# Reads in saved classification model
load_clf = pickle.load(open('ni4f_auc_cb2.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)

st.subheader('Prediction :')
df1=pd.DataFrame(prediction,columns=['0'])
df1.loc[df1['0'] == 0, 'Chances of GDM'] = 'No'
df1.loc[df1['0'] == 1, 'Chances of GDM'] = 'Yes'
st.write(df1)

prediction_proba = load_clf.predict_proba(df)
st.subheader('Prediction Probability in % :')
st.write(prediction_proba * 100)
