from sklearn.preprocessing import StandardScaler
from pandas import read_csv
from numpy import set_printoptions

filename ='indians-diabetes.data.csv'
hnames=['preg','plas','pres','skin','tets','mass','pedi','age','class']
dataframe=read_csv(filename,names=hnames)
array=dataframe.values
#seperate array into input and outpu componests
x=array[:,0:8]
y=array[:, 8]
scaler=StandardScaler()
#first method
rescaledX= scaler.fit_transform(x)
#summarize transformed data
set_printoptions(precision=2)
print(rescaledX[:10,:])