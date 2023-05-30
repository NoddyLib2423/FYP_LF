import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics 
import seaborn as sns
import scipy.stats as stats
import plotly.graph_objects as go
import plotly.express as px
sns.set(rc={'figure.figsize':(11, 4)})
from scipy import stats
from numpy import asarray
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from matplotlib import pyplot
from mlxtend.plotting import plot_decision_regions

def Datacleaning2(df):
    colgate = df
    del colgate['region']
    del colgate['dbid']
    del colgate['storerkey']
    del colgate['company']
    del colgate['CustomerGroupName']
    del colgate['MarketSegment']
    del colgate['CountryOfOrigin']
    del colgate['containertype']
    del colgate['facility']
    del colgate['descr']
    del colgate['orderkey']
    del colgate['style']
    del colgate['sku']
    del colgate['cube']
    del colgate['weight']
    del colgate['class']
    del colgate['size']
    del colgate['descr.1']
    del colgate['uom']
    colgate_STD =colgate.loc[colgate['skugroup'] == 'STD']
    colgate_others =colgate.loc[colgate['skugroup'] != 'STD']
    del colgate_STD['skugroup']
    del colgate_others['skugroup']

    #STD
    colgate_STD['deliverydate']=pd.to_datetime(colgate_STD['deliverydate'])
    colgate_STD_Groupped = colgate_STD.groupby(pd.Grouper(key='deliverydate', axis=0, freq='1D', sort=True))
    result_colgate_STD=colgate_STD_Groupped.sum()
    Finalresult_colgate_STD = result_colgate_STD["shippedqty"].to_frame()
    Finaldateframe_colgate_STD=Finalresult_colgate_STD[~(Finalresult_colgate_STD==0).any(axis=1)]
    Finaldateframe_colgate_STD.reset_index()[['deliverydate', 'shippedqty']].to_csv('csv/STD_outlier_dataset.csv',index=False)
    #Others
    colgate_others['deliverydate']=pd.to_datetime(colgate_others['deliverydate'])
    colgate_others_Groupped = colgate_others.groupby(pd.Grouper(key='deliverydate', axis=0, freq='1D', sort=True))
    result_colgate_others=colgate_others_Groupped.sum()
    Finalresult_colgate_others = result_colgate_others["shippedqty"].to_frame()
    Finaldateframe_colgate_others=Finalresult_colgate_others[~(Finalresult_colgate_others==0).any(axis=1)]
    Finaldateframe_colgate_others.reset_index()[['deliverydate', 'shippedqty']].to_csv('csv/others_outlier_dataset.csv',index=False)

    pf_STD = pd.read_csv("csv/STD_outlier_dataset.csv")
    pf_others = pd.read_csv("csv/others_outlier_dataset.csv")
    return pf_STD,pf_others


def Outiler2(data):
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
    return Q3,Q1,IQR,lower_outlier,higher_outlier

# data=data_STD
# Outiler2(data)
# Q3,Q1,IQR,lower_outlier,higher_outlier=Outiler2(data)

# Q3_STD=Q3
# IQR_STD=IQR
# pf_STD.loc[data > (Q3_STD + 1.5 * IQR_STD) ] = np.nan
# pf_STD.to_csv('csv/STD_outlier_dataset.csv',index=False)
# pf_STD.dropna(axis = 0).to_csv('csv/STD_outlier_dataset.csv',index=False)


# data=data_others
# Outiler2(data)
# Q3,Q1,IQR,lower_outlier,higher_outlier=Outiler2(data)

# Q3_others=Q3
# IQR_others=IQR
# pf_others.loc[data > (Q3_others + 1.5 * IQR_others) ] = np.nan
# pf_others.to_csv('csv/others_outlier_dataset.csv',index=False)
# pf_others.dropna(axis = 0).to_csv('csv/others_outlier_dataset.csv',index=False)

