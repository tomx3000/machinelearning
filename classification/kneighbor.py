import numpy as np 
from sklearn import preprocessing,cross_validation,neighbors
import matplotlib.pyplot as plt 
from matplotlib import style
import pickle
import pandas as pd 

# from classifier import classifier

from statistics import mean
import random


class Kneighbor:
	def __init__(self,*args,**kargs):
		pass

	def get_data(self,*args,**kargs):

		df = pd.read_csv('breast-cancer-wisconsin.data.txt')
		df.replace(['?'],-99999,inplace=True)
		df.drop(['id'],1,inplace=True)	

		
		self.ys=np.array(df['class'])
		self.xs=np.array(df.drop(['class'],1),dtype=np.float64)
		
		return self

	def fit(self,*args,**kargs):
		x_train,self.x_test,y_train,self.y_test=cross_validation.train_test_split(self.xs,self.ys,test_size=0.2)

		self.clf = neighbors.KNeighborsClassifier()
		self.clf.fit(x_train,y_train)
		
		return self

	def score(self,*args,**kargs):
		print('Accuracy :{}'.format(self.clf.score(self.x_test,self.y_test)))
		
		return self

	def predict(self,*args,**kargs):
		value=kargs['data'] if kargs.__contains__('data') else [4,5,2,1,1,1,2,7,4] 
		predict=np.array([value])
		predict=predict.reshape(len(predict),-1)
		print('cancer class :{}'.format(self.clf.predict(predict)))
		return self

	def visualize(self,*args,**kargs):
		pass

Kneighbor().get_data().fit().score().predict()