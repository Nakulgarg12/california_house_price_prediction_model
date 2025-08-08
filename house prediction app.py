import streamlit as st 
import pandas as pd 
import random
from sklearn.preprocessing import StandardScaler
import pickle
import warnings 
warnings.filterwarnings('ignore')
# Title 
col = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']
st.title('California Housing Price Prediction')
st.image('https://mir-s3-cdn-cf.behance.net/projects/404/0aa2df180622213.Y3JvcCw0MzE0LDMzNzUsODQzLDA.jpg')
st.header('A model of housing prices to predict median house values in California',divider=True)
st.sidebar.title('Select House Features 🏠')
st.sidebar.image('https://blog.architizer.com/wp-content/uploads/Untitled-design.gif')
temp_df = pd.read_csv('california.csv')
random.seed(10)
all_values=[]
for i in temp_df[col]:
    min_value, max_value = temp_df[i].agg([min,max])
    var =st.sidebar.slider(f'select {i} value' , int(min_value) , int(max_value),
                           random.randint(int(min_value),int(max_value)))
    all_values.append(var)
ss = StandardScaler()
ss.fit(temp_df[col])
final_value = ss.transform([all_values]) 
with open('house_price_pred_ridge_model.pkl','rb') as f:
    chatgpt = pickle.load(f)

price = chatgpt.predict(final_value)[0]
import time 
st.write(pd.DataFrame(dict(zip(col,all_values)),index =[1]))
progress_bar = st.progress(0)
placeholder = st.empty()
placeholder.subheader('Predicting Price')
place = st.empty()
place.image('https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExcmoxdnBlb3Y3Z21vamhyaHFpemdzeTJtaXl0c3d2N2pyaTl3azI2NiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3o7bu3XilJ5BOiSGic/giphy.gif',width = 40)
if price>0:
    for i in range (100):
        time.sleep(0.05)
        progress_bar.progress(i+1)
    body= f'Predicted Median House Price: ${round(price,2)} Thousand Dollars'
    placeholder.empty()
    place.empty()
    st.success(body)
else:
    body = 'Invalid House Feature Values'
    st.warning(body)