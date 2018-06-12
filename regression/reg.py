import pandas as pd 
import quandl,math
import numpy as np 
from sklearn import preprocessing,cross_validation ,svm
from sklearn.linear_model import LinearRegression

class stockRegression:
	def __init__(self):
		pass
	def getData(self):
		df = quandl.get('WIKI/GOOGL')
		df =df[['Adj. Close','Adj. Open','Adj. High','Adj. Volume','Adj. Low']]
		forecast_col = 'Adj. Close'
		df.fillna(-99999,inplace=True)

		forecast_out= int(math.ceil(0.01*len(df)))

		df['label']=df[forecast_col].shift(-forecast_out)

		df.dropna(inplace=True)
		# print(df.head)

		self.df = df
		print('received Data')
		return self

	def prepData(self,*args,**kargs):
		# df=kargs['df']
		df=self.df
		X=np.array(df.drop(['label'],1))
		Y=np.array(df['label'])

		X=preprocessing.scale(X)
		
		X_train,X_test,Y_train,Y_test=cross_validation.train_test_split(X,Y,test_size=0.2)
		clf = LinearRegression(n_jobs=-1)
		clf.fit(X_train,Y_train)
		print(clf.score(X_test,Y_test))

		return self
		

	def fit(self):


		pass

	def predict(self):


		pass

	def scoreData(self):
		pass



test= stockRegression()
test.getData().prepData()
