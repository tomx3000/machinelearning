import pandas as pd 
import quandl,math,datetime
import numpy as np 
from sklearn import preprocessing,cross_validation ,svm
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt 
from matplotlib import style
import pickle


style.use('ggplot')

class stockRegression:
	def __init__(self):
		pass
	def getData(self):
		df = quandl.get('WIKI/GOOGL')
		df =df[['Adj. Close','Adj. Open','Adj. High','Adj. Volume','Adj. Low']]
		forecast_col = 'Adj. Close'
		df.fillna(-99999,inplace=True)

		self.forecast_out= int(math.ceil(0.01*len(df)))

		df['label']=df[forecast_col].shift(-self.forecast_out)

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
		X_sample=X[:-self.forecast_out]
		Y_sample=Y[:-self.forecast_out]

		self.X_lately=X[-self.forecast_out:]

		X_train,X_test,Y_train,Y_test=cross_validation.train_test_split(X_sample,Y_sample,test_size=0.2)

		# self.clf = LinearRegression(n_jobs=-1)

		
		# self.clf.fit(X_train,Y_train)

		# classifier().save(filename='linear',clf=self.clf)

		self.clf=classifier().get(filename='linear')
		print(self.clf.score(X_test,Y_test))

		return self
		

	def fit(self):


		pass

	def visualize(self):
		lastdate= self.df.iloc[-1].name
		last_unix=lastdate.timestamp()
		oneday=86400
		nex_unix=last_unix+oneday

		for i in self.forcastset:
			next_date=datetime.datetime.fromtimestamp(nex_unix)
			nex_unix+=oneday
			self.df.loc[next_date]=[np.nan for _ in range (len(self.df.columns)-1)]+[i]

		self.df['Adj. Close'].plot()
		self.df['Forecast'].plot()
		plt.legend(loc=4)
		plt.xlabel('Date')
		plt.ylabel('Price')
		plt.show()	

		pass

	def predict(self):
		self.forcastset=self.clf.predict(self.X_lately)
		print(self.forcastset)
		self.df['Forecast']=np.nan

		return self

	def scoreData(self):
		pass

class classifier:
	def __init__(self,*args,**kargs):
		if(kargs.__contains__('clf')):
			self.clf=kargs['clf']
		if(kargs.__contains__('filename')):
			self.file=kargs['fiename']
		
		pass

	def save(self,*args,**kargs):
		filename=kargs['filename'] if kargs.__contains__('filename') else self.file
		clf=kargs['clf'] if kargs.__contains__('clf') else self.clf

		with open(filename+'.pickle','wb') as f:
			pickle.dump(clf,f)
		return self

	def get(self,*args,**kargs):
		filename=kargs['filename'] if kargs.__contains__('filename') else self.file
		clf = open(filename+'.pickle','rb')
		return pickle.load(clf)








test= stockRegression()
test.getData().prepData().predict()
