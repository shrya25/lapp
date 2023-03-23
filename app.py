import streamlit as st
import pickle
import numpy as np

# import the model
pipe = pickle.load(open('expipe (2).pkl','rb'))
data = pickle.load(open('data.pkl','rb'))

st.title("Laptop Predictor")

# brand
company = st.selectbox('Brand',data['Brand'].unique())

# type of laptop
type = st.selectbox('Type',data['type'].unique())

# Ram
ram = st.selectbox('RAM(in GB)',[2,4,6,8,12,16,24,32,64])

# screen size
screen_size = st.number_input('Screen Size')

#cpu
cpu = st.selectbox('CPU',data['cpu brand'].unique())

hdd = st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])

gpu = st.selectbox('GPU',data['gpu'].unique())

os = st.selectbox('OS',data['operatingsys'].unique())

if st.button('Predict Price'):
    query = np.array([company,type,ram,cpu,screen_size,hdd,gpu,os])

    query = query.reshape(1,8)
    st.title("The predicted price of this configuration is " + str(int(np.exp(pipe.predict(query)[0]))))
