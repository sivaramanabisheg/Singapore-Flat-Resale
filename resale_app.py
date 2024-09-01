import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import datetime
import pickle
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt


# In[8]:


def main():
    st.set_page_config(
        page_title="Singapore Flat Resale Price Predictor",
        page_icon='🏠',
        initial_sidebar_state='expanded',
        layout='wide',
        menu_items={"about":"This Streamlit application is developed for Singapore flat resale price prediction"}
    )
    st.title(":green[Singapore flat Resale Price Predictor]")
    
    selected=option_menu("Singapore Flat Resale | Comprehensive Analysis and Predictive Modelling",
                            options=["Home", "Prediction", "Explore"],
                            icons=["house", "lightbulb", "bar-chart-line"],
                            default_index=1, menu_icon="globe",
                            orientation="horizontal")
    if selected=="Home":
        title_text='''<h1 style='font-size: 30px;text-align: center; color: grey;'>Singapore Resale Flat Price Predictor</h1>'''
        st.markdown(title_text, unsafe_allow_html=True)
        col1, col2=st.columns([2, 1.5], gap="large")
        with col1:
            st.markdown("### :green[Skills Takeaway]:")
            st.markdown('<h5> Data Wrangling, EDA, Model Building, Model Deployment </h5>', unsafe_allow_html=True)

            st.markdown("### :green[Domain]:")
            st.markdown('<h5> Real Estate </h5>', unsafe_allow_html=True)

            st.markdown("### :green[Overview]:")
            st.markdown('''<h4> 
                                <li> Collected and processed Singapore HDB resale flat transction data (1990-Present) using Python,<br>
                                <li> Cleaned and structured data for Machine Learning,<br>
                                <li> Analyzed Pricing trends and predictions,<br>
                                <li> Developed a user-friendly application for resale price predictions.
                        </h4>''', unsafe_allow_html=True)
            st.info('''
                ### :green[Problem Statement]: ###
                Predicting resale flat prices in Singapore can be challenging due to various factors such as location, flat type, floor area and lease duration.
                A Machine Learning model can provide accurate price estimates using historical resale flat transaction data, the model aims to assist potential buyers and sellers in estimating the resale value of a flat.
            ''')

            st.markdown("### :green[Solution steps]: ###")
            st.markdown("""
                - 🔍 Collect and preprocess resale flat transaction data
                - 🔄 Extract relevant features and create additional features
                - 📈 Train a regression model on historical data
                - 🎯 Develop a user-friendly web application for price predictions
            """)

            st.markdown("### :greeen[Data Source] ###")
            st.markdown("### [Singapore Government Data](https://beta.data.gov.sg/collections/189/view)")

        with col2:
            st.image("https://j.gifs.com/66jXYL.gif", use_column_width=True)
            st.write("----")
            st.markdown("  ")
            
        co1,col2=st.columns([2,2])
        with col1:
            st.image('https://media2.malaymail.com/uploads/articles/2020/2020-07/20200725_Singapore-HDB.jpg', use_column_width=True)
        with col2:
            st.image('https://miro.medium.com/v2/resize:fit:1400/0*hn4nICHk9Cq-tugt.jpeg', use_column_width=True)
    
    class option:
        option_months=["January","February","March","April","May","June","July","August","September","October","November","December"]
        
        current_year=datetime.datetime.now().year
        option_year=[str(year) for year in range(1990, current_year+1)]
        encoded_month= {"January" : 1,"February" : 2,"March" : 3,"April" : 4,"May" : 5,"June" : 6,"July" : 7,"August" : 8,"September" : 9,
                "October" : 10 ,"November" : 11,"December" : 12}

        option_town=['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH','BUKIT TIMAH', 'CENTRAL AREA', 'CHOA CHU KANG',
                     'CLEMENTI', 'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST','KALLANG/WHAMPOA', 'MARINE PARADE', 'QUEENSTOWN', 
                     'SENGKANG','SERANGOON', 'TAMPINES', 'TOA PAYOH', 'WOODLANDS', 'YISHUN','LIM CHU KANG', 'SEMBAWANG', 'BUKIT PANJANG',
                     'PASIR RIS','PUNGGOL']
        
        encoded_town={'ANG MO KIO' : 0 ,'BEDOK' : 1,'BISHAN' : 2,'BUKIT BATOK' : 3,'BUKIT MERAH' : 4,'BUKIT PANJANG' : 5,'BUKIT TIMAH' : 6,
            'CENTRAL AREA' : 7,'CHOA CHU KANG' : 8,'CLEMENTI' : 9,'GEYLANG' : 10,'HOUGANG' : 11,'JURONG EAST' : 12,'JURONG WEST' : 13,
            'KALLANG/WHAMPOA' : 14,'LIM CHU KANG' : 15,'MARINE PARADE' : 16,'PASIR RIS' : 17,'PUNGGOL' : 18,'QUEENSTOWN' : 19,
            'SEMBAWANG' : 20,'SENGKANG' : 21,'SERANGOON' : 22,'TAMPINES' : 23,'TOA PAYOH' : 24,'WOODLANDS' : 25,'YISHUN' : 26}
        
        option_flat_type=['1 ROOM', '2 ROOM','3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE','MULTI-GENERATION']

        encoded_flat_type={'1 ROOM': 0,'2 ROOM' : 1,'3 ROOM' : 2,'4 ROOM' : 3,'5 ROOM' : 4,'EXECUTIVE' : 5,'MULTI-GENERATION' : 6}

        option_flat_model=['2-ROOM','3GEN','ADJOINED FLAT', 'APARTMENT' ,'DBSS','IMPROVED' ,'IMPROVED-MAISONETTE', 'MAISONETTE',
                        'MODEL A', 'MODEL A-MAISONETTE','MODEL A2' ,'MULTI GENERATION' ,'NEW GENERATION', 'PREMIUM APARTMENT',
                        'PREMIUM APARTMENT LOFT', 'PREMIUM MAISONETTE','SIMPLIFIED', 'STANDARD','TERRACE','TYPE S1','TYPE S2']

        encoded_flat_model={'2-ROOM' : 0,'3GEN' : 1,'ADJOINED FLAT' : 2,'APARTMENT' : 3,'DBSS' : 4,'IMPROVED' : 5,'IMPROVED-MAISONETTE' : 6,
                    'MAISONETTE' : 7,'MODEL A' : 8,'MODEL A-MAISONETTE' : 9,'MODEL A2': 10,'MULTI GENERATION' : 11,'NEW GENERATION' : 12,
                    'PREMIUM APARTMENT' : 13,'PREMIUM APARTMENT LOFT' : 14,'PREMIUM MAISONETTE' : 15,'SIMPLIFIED' : 16,'STANDARD' : 17,
                    'TERRACE' : 18,'TYPE S1' : 19,'TYPE S2' : 20}
    if selected=="Prediction":
        st.write('')
        title_text='''<h2 style='font-size:32px; text-align:center; color:grey;'>Resale Flat Price Prediction</h2>'''
        st.markdown(title_text, unsafe_allow_html=True)
        st.markdown("<h5 style=color:orange> To predict the resale price of a flat, Please provide the following information:", unsafe_allow_html=True)
        st.write('')

        with st.form('prediction'):
            col1, col2=st.columns(2)
            with col1:
                user_month=st.selectbox(label='Month', options=option.option_months, index=None)
                user_town=st.selectbox(label='Town', options=option.option_town, index=None)
                user_flat_type=st.selectbox(label='Flat Type', options=option.option_flat_type, index=None)
                user_flat_model=st.selectbox(label='Flat Model', options=option.option_flat_model, index=None)
                floor_area_sqm=st.number_input(label='Floor area sqm (10 to 307)', min_value=10.0)
                price_per_sqm=st.number_input(label='Price per sqm', min_value=100.0)
            with col2:
                year=st.selectbox(label='Year', options=option.option_year, index=None)
                block=st.number_input(label='Block (1 to 999)', min_value=1, max_value=999, step=1)
                lease_commence_date=st.text_input(label='Year of lease commence (1996 to 2020)', max_chars=4)
                remaining_lease=st.number_input(label='Remaining lease year (0 to 99)', min_value=0, max_value=99, step=1)
                years_holding=st.number_input(label='Years holding (0 to 99)', min_value=0, max_value=99, step=1)
                
                c1,c2=st.columns(2)
                with c1:
                    storey_start=st.number_input(label='Storey start (1 to 50)', min_value=1, max_value=50)
                with c2:
                    storey_end=st.number_input(label='Storey end (1 to 51)', min_value=1, max_value=51)
                st.markdown('<br>', unsafe_allow_html=True)
                
                button=st.form_submit_button('Predict price', use_container_width=True)
                
                st.markdown("""
                            <style>
                            div.stButton> button:first-child {
                                background-color: #009999;
                                color: white;
                                width: 100%
                            }
                            </style>
                        """, unsafe_allow_html=True)
        if button:
            with st.spinner("Predicting....."):
            
                if not all([user_month,user_town,user_flat_type,user_flat_model,floor_area_sqm,price_per_sqm,year,block,
                        lease_commence_date,remaining_lease,years_holding,storey_start,storey_end]):
                    st.error("Please fill in all required fields")
                else:
                    current_year=datetime.datetime.now().year
                    current_remaining_lease=remaining_lease-(current_year-(int(year)))
                    age_of_property=current_year-int(lease_commence_date)
                    month=option.encoded_month[user_month]
                    town=option.encoded_town[user_town]
                    flat_type=option.encoded_flat_type[user_flat_type]
                    flat_model=option.encoded_flat_model[user_flat_model]
                    floor_area_sqm_log=np.log(floor_area_sqm)
                    remaining_lease_log=np.log1p(remaining_lease)
                    price_per_sqm_log=np.log(price_per_sqm)

                    with open('Decisiontreemodel.pkl', 'rb') as files:
                        model=pickle.load(files)
                    user_data=np.array([[month, town, flat_type, block, flat_model, lease_commence_date, year, storey_start,
                                        storey_end, years_holding, current_remaining_lease, age_of_property, floor_area_sqm_log,
                                        remaining_lease_log, price_per_sqm_log]])
                    predict=model.predict(user_data)
                    resale_price_sgd=np.exp(predict[0])
                    resale_price_inr=resale_price_sgd * 64.13
    
                    st.subheader(f"Predicted Resale price is : :green[${resale_price_sgd:.2f}]")
                    st.subheader(f"Predicted Resale price in INR is : :green[₹{resale_price_inr:.2f}]")

    if selected == "Explore":
        st.markdown('<br>', unsafe_allow_html=True)
        st.subheader(':green[About Housing and Development Board]')
        col1, col2 = st.columns([3,1])
        with col1:
            st.info('''
            -The Housing and Development Board (HDB : Often referred to as the Housing Board), is a statutory board under the Ministry of National Development responsible for the public housing in Singapore.
            -Established in 1960 as a result of efforts in the late 1950s to setup an authority to take over the Singapore Improvement Trust's (SIT) public housing responsibilities.
            -The HDB focused on the construction of emergency housing and the resettlement of Kampong residents into publi housing in the first few years of its existence.
            -In the 1990s and 2000s, the HDB introduced the upgrading and re-development schemes for mature estates, as well as new type of housing intended to cater to different income groups in partnership with private developers.
            -The HDB was recogonized in 2003 to better suit Singapore's housing market in the 2000s.
            ''')

    st.markdown(" ")
    st.markdown(" ")

if __name__ == "__main__":
    main()