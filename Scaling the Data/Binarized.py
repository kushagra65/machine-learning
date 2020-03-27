from sklearn.preprocessing import Binarizer
from pandas import read_csv
from numpy import set_printoptions
filename ='indians-diabetes.data.csv'
hnames=['preg','plas','pres','skin','tets','mass','pedi','age','class']
dataframe=read_csv(filename,names=hnames)
array=dataframe.values
#seperate array into input and outpu componests
x=array[:,0:8]
y=array[:, 8]
scaler=Binarizer
#first method
binarizer= Binarizer(threshold=5)#threshold limits the value in 0 or 1 if the no
                                # is greater than the limit then the no will convert into 1 else 0)
binaryX=binarizer.fit_transform(x)
#summarize transformed data
set_printoptions(precision=2)
print(binaryX[:10,:])