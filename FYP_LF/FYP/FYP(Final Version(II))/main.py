import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import base64
from Datacleaning_Nike import *
from Datacleaning_Colmgate import *
from Appearl_XGB import *
from Equipment_XGB import *
from Footwear_XGB import *
from Others_XGB import *
from STD_XGB import *
def add_logo(logo_path, width, height):
    """Read and return a resized logo"""
    logo = Image.open(logo_path)
    modified_logo = logo.resize((width, height))
    return modified_logo

def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

st.markdown(
    """
    <style>
        [data-testid=stSidebar] [data-testid=stImage]{
            text-align: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
        }
    </style>
    """, unsafe_allow_html=True
)

with st.sidebar:
    st.image(add_logo(logo_path="photo/download.png", width=268, height=128)) 
    choose = option_menu("Volume Prediction", ['LF Logistics'],
                        icons=['bi bi-cloud-arrow-up','bi bi-cloud-arrow-up'],
                        menu_icon="bi bi-clipboard-data", default_index=0,
                        styles={
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "20px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02ab21"},
    }
    )

img1 = get_img_as_base64("photo/download-1.jpg")
img2 = get_img_as_base64("photo/Picture 1.jpg")
imgs = f"""
            <style>
            [data-testid="stAppViewContainer"] > .main {{
            background-image: url("data:image/png;base64,{img1}");
            background-size: 100%;
            background-position: top left ;
            background-repeat: no-repeat;
            background-size: cover;
            background-attachment: local;
            
            }}
            [data-testid="stSidebar"] > div:first-child {{
            background-image: url("data:image/png;base64,{img2}");
            background-position: center; 
            background-size: 100%;
            background-repeat: no-repeat;
            background-attachment: fixed;
            }}
            [data-testid="stHeader"] {{
            background: rgba(0,0,0,0);
            }}
            [data-testid="stToolbar"] {{
            right: 2rem;
            }}
            </style>
            """
st.markdown(imgs, unsafe_allow_html=True)



#Nike
if choose == 'LF Logistics':
    st.subheader("Upload your CSV")
    uploaded_data = st.file_uploader(
                "Drag and Drop or Click to Upload", type=".csv", accept_multiple_files=True)   
    tab1, tab2= st.tabs(['NIKE (Hong Kong)', "COLMGATE (Philippines)"])

    st.write('---') 
    with tab1:
        st.header("LF Logistics-Nike")

        if len(uploaded_data) > 0:
            st.success('Upload successfully', icon="✅")
            df = pd.read_csv(uploaded_data[0])
            df_ten=df.head(10)
            st.write(df_ten)
            pf_APPAREL,pf_Equipemnt,pf_FOOTWEAR = Datacleaning(df)
            data_APPAREL=pf_APPAREL['shippedqty']
            data_Equipemnt=pf_Equipemnt['shippedqty']
            data_FOOTWEAR=pf_FOOTWEAR['shippedqty']
                
            data=data_APPAREL
            Q1,Q3,IQR,lower_outlier,higher_outlier =Outiler(data)

            Q3_Appeal=Q3
            IQR_Appeal=IQR
            pf_APPAREL.loc[data > (Q3_Appeal + 1.5 * IQR_Appeal) ] = np.nan
            pf_APPAREL.to_csv('csv/Appeal_outlier_dataset.csv',index=False)
            pf_APPAREL.dropna(axis = 0).to_csv('csv/Appeal_outlier_dataset.csv',index=False)

            data=data_Equipemnt
            Q1,Q3,IQR,lower_outlier,higher_outlier =Outiler(data)

            Q3_EQUIPMENT=Q3
            IQR_EQUIPMENT=IQR
            pf_Equipemnt.loc[data > (Q3_EQUIPMENT + 1.5 * IQR_EQUIPMENT) ] = np.nan
            pf_Equipemnt.to_csv('csv/EQUIPMENT_outlier_dataset.csv',index=False)
            pf_Equipemnt.dropna(axis = 0).to_csv('csv/EQUIPMENT_outlier_dataset.csv',index=False)

            data=data_FOOTWEAR
            Q1,Q3,IQR,lower_outlier,higher_outlier =Outiler(data)

            Q3_FOOTWEAR=Q3
            IQR_FOOTWEAR=IQR
            pf_FOOTWEAR.loc[data > (Q3_FOOTWEAR+ 1.5 * IQR_FOOTWEAR) ] = np.nan
            pf_FOOTWEAR.to_csv('csv/FOOTWEAR_outlier_dataset.csv',index=False)
            pf_FOOTWEAR.dropna(axis = 0).to_csv('csv/FOOTWEAR_outlier_dataset.csv',index=False)
            st.write('---')
            option = st.selectbox('Which one would you like to be predicted?',
                ('Footwear', 'Equipment', 'Appearl'))
            st.write('You selected:', option)
        #Appearl
            if option == 'Appearl':
                df_appearl=pd.read_csv('csv/Appeal_outlier_dataset.csv')
                sum,graph,df_pred1,prediction,actual = modelappearl(df_appearl)
                st.plotly_chart(sum)
                if (df_pred1['Predicted'].sum()-df_pred1['Actual'].sum())<0:
                    st.write("we have",((df_pred1['Predicted'].sum()-df_pred1['Actual'].sum())/df_pred1['Predicted'].sum())*(-100), "% of opportunity lost will happen by this model.")
                else:
                    st.write("we have",((df_pred1['Predicted'].sum()-df_pred1['Actual'].sum())/df_pred1['Predicted'].sum())*(100), "% of items will not sale by this model.")
                st.write('---')
                st.plotly_chart(graph)
                st.write('The Predicton value is:',prediction[-1])
                st.write('The Actual value is:',actual[-1])
        #Equipment
            elif option == 'Equipment':
                df_EQUIPMENT=pd.read_csv('csv/EQUIPMENT_outlier_dataset.csv')
                sum,graph,df_pred1,prediction,actual = modelEQUIPMENT(df_EQUIPMENT)
                st.plotly_chart(sum)
                if (df_pred1['Predicted'].sum()-df_pred1['Actual'].sum())<0:
                    st.write("we have",((df_pred1['Predicted'].sum()-df_pred1['Actual'].sum())/df_pred1['Predicted'].sum())*(-100), "% of opportunity lost will happen by this model.")
                else:
                    st.write("we have",((df_pred1['Predicted'].sum()-df_pred1['Actual'].sum())/df_pred1['Predicted'].sum())*(100), "% of items will not sale by this model.")
                st.write('---')
                st.plotly_chart(graph)
                st.write('The Predicton value is:',prediction[-1])
                st.write('The Actual value is:',actual[-1])
        #Footwear
            else: 
                option == 'Footwear' 
                df_FOOTWEAR=pd.read_csv('csv/FOOTWEAR_outlier_dataset.csv')
                sum,graph,df_pred1,prediction,actual = modelFOOTWEAR_(df_FOOTWEAR)
                st.plotly_chart(sum)
                if (df_pred1['Predicted'].sum()-df_pred1['Actual'].sum())<0:
                    st.write("we have",((df_pred1['Predicted'].sum()-df_pred1['Actual'].sum())/df_pred1['Predicted'].sum())*(-100), "% of opportunity lost will happen by this model.")
                else:
                    st.write("we have",((df_pred1['Predicted'].sum()-df_pred1['Actual'].sum())/df_pred1['Predicted'].sum())*(100), "% of items will not sale by this model.")
                st.write('---')
                st.plotly_chart(graph)
                st.write('The Predicton value is:',prediction[-1])
                st.write('The Actual value is:',actual[-1])







#Colmagte
    with tab2:
        st.header("LF Logistics-Colmagate")
        if len(uploaded_data) >0:
            st.success('Upload successfully', icon="✅")
            df = pd.read_csv(uploaded_data[1])
            df_ten=df.head(10)
            st.write(df_ten)
            pf_STD,pf_others = Datacleaning2(df)
            data_STD=pf_STD['shippedqty']
            data_others=pf_others['shippedqty']
                
            data=data_STD
            Q1,Q3,IQR,lower_outlier,higher_outlier =Outiler2(data)
            Q3_STD=Q3
            IQR_STD=IQR
            pf_STD.loc[data > (Q3_STD + 1.5 * IQR_STD) ] = np.nan
            pf_STD.to_csv('csv/STD_outlier_dataset.csv',index=False)
            pf_STD.dropna(axis = 0).to_csv('csv/STD_outlier_dataset.csv',index=False)


            data=data_others
            Q3,Q1,IQR,lower_outlier,higher_outlier=Outiler2(data)
            Q3_others=Q3
            IQR_others=IQR
            pf_others.loc[data > (Q3_others + 1.5 * IQR_others) ] = np.nan
            pf_others.to_csv('csv/others_outlier_dataset.csv',index=False)
            pf_others.dropna(axis = 0).to_csv('csv/others_outlier_dataset.csv',index=False)
            st.write('---')
            option = st.selectbox(
                'Which one would you like to be predicted?',
                ('STD', 'Others'))
            st.write('You selected:', option)
            #STD
            if option == 'STD':
                df_STD=pd.read_csv('csv/STD_outlier_dataset.csv')
                sum,graph,df_pred1,prediction,actual = modelSTD(df_STD)
                st.plotly_chart(sum)
                if (df_pred1['Predicted'].sum()-df_pred1['Actual'].sum())<0:
                    st.write("we have",((df_pred1['Predicted'].sum()-df_pred1['Actual'].sum())/df_pred1['Predicted'].sum())*(-100), "% of opportunity lost will happen by this model.")
                else:
                    st.write("we have",((df_pred1['Predicted'].sum()-df_pred1['Actual'].sum())/df_pred1['Predicted'].sum())*(100), "% of items will not sale by this model.")
                st.write('---')
                st.plotly_chart(graph)
                st.write('The Predicton value is:',prediction[-1])
                st.write('The Actual value is:',actual[-1])
         #Others
            elif option == 'Others':
                df_others=pd.read_csv('csv/others_outlier_dataset.csv')
                sum,graph,df_pred1,prediction,actual = modelOthers(df_others)
                st.plotly_chart(sum)
                if (df_pred1['Predicted'].sum()-df_pred1['Actual'].sum())<0:
                    st.write("we have",((df_pred1['Predicted'].sum()-df_pred1['Actual'].sum())/df_pred1['Predicted'].sum())*(-100), "% of opportunity lost will happen by this model.")
                else:
                    st.write("we have",((df_pred1['Predicted'].sum()-df_pred1['Actual'].sum())/df_pred1['Predicted'].sum())*(100), "% of items will not sale by this model.")
                st.write('---')
                st.plotly_chart(graph)
                st.write('The Predicton value is:',prediction[-1])
                st.write('The Actual value is:',actual[-1])

