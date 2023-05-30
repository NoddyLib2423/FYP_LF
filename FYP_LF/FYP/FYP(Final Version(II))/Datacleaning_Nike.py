import statistics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
import seaborn as sns

sns.set(rc={'figure.figsize':(11, 4)})
from matplotlib import pyplot
from mlxtend.plotting import plot_decision_regions
from numpy import asarray
from pandas import DataFrame, concat, read_csv
from scipy import stats


def Datacleaning(df):
    nike = df
    del nike['region']
    del nike['dbid']
    del nike['storerkey']
    del nike['company']
    del nike['CustomerGroupName']
    del nike['MarketSegment']
    del nike['CountryOfOrigin']
    del nike['containertype']
    del nike['facility']
    del nike['descr']
    del nike['orderkey']
    del nike['style']
    del nike['sku']
    del nike['cube']
    del nike['weight']
    del nike['class']
    del nike['size']
    del nike['descr.1']
    del nike['uom']
    Filter_skugroup=nike["skugroup"].unique()
    nike_APPAREL =nike.loc[nike['skugroup'] == 'APPAREL']
    nike_EQUIPMENT =nike.loc[nike['skugroup'] == 'EQUIPMENT']
    nike_FOOTWEAR =nike.loc[nike['skugroup'] == 'FOOTWEAR']
    del nike_APPAREL['skugroup']
    del nike_EQUIPMENT['skugroup']
    del nike_FOOTWEAR['skugroup']
    #Apparel
    nike_APPAREL['deliverydate']=pd.to_datetime(nike_APPAREL['deliverydate'])
    nike_APPAREL_Groupped = nike_APPAREL.groupby(pd.Grouper(key='deliverydate', axis=0, freq='1D', sort=True))
    result_nike_APPAREL=nike_APPAREL_Groupped.sum()
    Finalresult_nike_APPAREL = result_nike_APPAREL["shippedqty"].to_frame()
    Finaldateframe_nike_APPAREL=Finalresult_nike_APPAREL[~(Finalresult_nike_APPAREL==0).any(axis=1)]
    Finaldateframe_nike_APPAREL.reset_index()[['deliverydate', 'shippedqty']].to_csv('csv/Appeal_outlier_dataset.csv',index=False)
    #Equipemnt
    nike_EQUIPMENT['deliverydate']=pd.to_datetime(nike_EQUIPMENT['deliverydate'])
    nike_EQUIPMENT_Groupped = nike_EQUIPMENT.groupby(pd.Grouper(key='deliverydate', axis=0, freq='1D', sort=True))
    result_nike_EQUIPMENT=nike_EQUIPMENT_Groupped.sum()
    Finalresult_nike_EQUIPMENT = result_nike_EQUIPMENT["shippedqty"].to_frame()
    Finaldateframe_nike_EQUIPMENT=Finalresult_nike_EQUIPMENT[~(Finalresult_nike_EQUIPMENT==0).any(axis=1)]
    Finaldateframe_nike_EQUIPMENT.reset_index()[['deliverydate', 'shippedqty']].to_csv('csv/EQUIPMENT_outlier_dataset.csv',index=False)
    #Footwear
    nike_FOOTWEAR['deliverydate']=pd.to_datetime(nike_FOOTWEAR['deliverydate'])
    nike_FOOTWEAR_Groupped = nike_FOOTWEAR.groupby(pd.Grouper(key='deliverydate', axis=0, freq='1D', sort=True))
    result_nike_FOOTWEAR=nike_FOOTWEAR_Groupped.sum()
    Finalresult_nike_FOOTWEAR = result_nike_FOOTWEAR["shippedqty"].to_frame()
    Finaldateframe_nike_FOOTWEAR=Finalresult_nike_FOOTWEAR[~(Finalresult_nike_FOOTWEAR==0).any(axis=1)]
    Finaldateframe_nike_FOOTWEAR.reset_index()[['deliverydate', 'shippedqty']].to_csv('csv/FOOTWEAR_outlier_dataset.csv',index=False)

    #APPAREL
    pf_APPAREL = pd.read_csv("csv/Appeal_outlier_dataset.csv")
    #Equipemnt
    pf_Equipemnt = pd.read_csv("csv/EQUIPMENT_outlier_dataset.csv")
    #FOOTWEAR
    pf_FOOTWEAR = pd.read_csv("csv/FOOTWEAR_outlier_dataset.csv")
    return pf_APPAREL,pf_Equipemnt,pf_FOOTWEAR

def Outiler(data):
    median = statistics.median(data)
    Q3 = (3/4)*median
    Q1 = (1/4)*median
    IQR = Q3-Q1
    lower_outlier = Q1 - 1.5*IQR
    higher_outlier = Q3 + 1.5*IQR
    plt.boxplot(data)
    plt.axhline(y=lower_outlier, color='blue', linestyle='-')
    plt.axhline(y=higher_outlier, color='green', linestyle='-')
    print('Q3:',Q3)
    print('Q1:',Q1)
    print('IQR:',IQR)
    print('lower_outlier:',lower_outlier)
    print('higher_outlier:',higher_outlier)
    return Q1,Q3,IQR,lower_outlier,higher_outlier

# data_APPAREL = pd.read_csv("Appeal_outlier_dataset.csv")
# data_Equipemnt = pd.read_csv("EQUIPMENT_outlier_dataset.csv")
# data_FOOTWEAR = pd.read_csv("FOOTWEAR_outlier_dataset.csv")

# data=data_APPAREL
# Outiler(data)
# Q1,Q3,IQR,lower_outlier,higher_outlier =Outiler(data)

# Q3_Appeal=Q3
# IQR_Appeal=IQR
# data_APPAREL.loc[data > (Q3_Appeal + 1.5 * IQR_Appeal) ] = np.nan
# data_APPAREL.to_csv('Appeal_outlier_dataset.csv',index=False)
# data_APPAREL.dropna(axis = 0).to_csv('Appeal_outlier_dataset.csv',index=False)

# data=data_Equipemnt
# Outiler(data)
# Q1,Q3,IQR,lower_outlier,higher_outlier =Outiler(data)

# Q3_EQUIPMENT=Q3
# IQR_EQUIPMENT=IQR
# data_Equipemnt.loc[data > (Q3_EQUIPMENT + 1.5 * IQR_EQUIPMENT) ] = np.nan
# data_Equipemnt.to_csv('EQUIPMENT_outlier_dataset.csv',index=False)
# data_Equipemnt.dropna(axis = 0).to_csv('EQUIPMENT_outlier_dataset.csv',index=False)

# data=data_FOOTWEAR
# Outiler(data)
# Q1,Q3,IQR,lower_outlier,higher_outlier =Outiler(data)

# Q3_FOOTWEAR=7510.5
# IQR_FOOTWEAR=5007.0
# data_FOOTWEAR.loc[data > (Q3_FOOTWEAR+ 1.5 * IQR_FOOTWEAR) ] = np.nan
# data_FOOTWEAR.to_csv('FOOTWEAR_outlier_dataset.csv',index=False)
# data_FOOTWEAR.dropna(axis = 0).to_csv('FOOTWEAR_outlier_dataset.csv',index=False)