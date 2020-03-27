import pandas as pd
from numpy import set_printoptions
from sklearn.preprocessing import MinMaxScaler

filename ='indians-diabetes.data.csv'
hnames=['preg','plas','pres','skin','tets','mass','pedi','age','class']
dataframe=pd.read_csv(filename,names=hnames)
array=dataframe.values
#seperate array into input and outpu componests
x=array[:,0:8]
y=array[:, 8]
scaler=MinMaxScaler(feature_range=(1,5))
#first method
rescaledX= scaler.fit_transform(x)
#summarize transformed data
set_printoptions(precision=2)
print(rescaledX[:10,:])