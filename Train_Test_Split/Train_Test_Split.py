import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings(action='ignore')
#load data
filename ='indians-diabetes.data.csv'
hnames=['preg','plas','pres','skin','tets','mass','pedi','age','class']
dataframe=pd.read_csv(filename,names=hnames)
array=dataframe.values
#seperate array into input and output componests
x=array[:,0:8]
y=array[:, 8]
test_data_size =0.33
seed=99
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=test_data_size,random_state=seed)
model=LogisticRegression()
model.fit(x_train,y_train)
result=model.score(x_test,y_test)
print('accuracy=%f%%'%(result*100))