import pandas as pd 
import quandl,math,datetime
import numpy as np 
from sklearn import preprocessing,cross_validation ,svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt 
from matplotlib import style
import pickle
from classifier import classifier

from statistics import mean
import random




class LinearRegression:
	def __init__(self,*args,**kargs):
		pass

	def fit(self,*args,**kargs):
		self.gradient= (mean(self.sample_xs)*mean(self.sample_ys)-mean(self.sample_xs*self.sample_ys))/(mean(self.sample_xs)**2-mean(self.sample_xs**2))

		self.intercept=mean(self.sample_ys)-self.gradient*mean(self.sample_xs)

		self.y_line=[self.gradient * x + self.intercept for x in self.sample_xs]

		return self
		
	def get_sample_data(self,*args,**kargs):
		self.sample_xs=np.array([1,2,3,4,5,6,7,8,9],dtype=np.float64)
		self.sample_ys=np.array([2,4,6,2,7,8,5,9,10])
		return self

	def predict(self,*args,**kargs):
		self.test_x=11
		if kargs.__contains__('x'):
			self.test_x=kargs['x']

		self.test_y=self.gradient * self.test_x + self.intercept 
		return self
		
	def score(self,*args,**kargs):
		y_mean_line=[mean(self.sample_ys) for _ in self.sample_ys]
		y_best_fit__line=self.y_line
		mean_squared_error=sum((y_mean_line-self.sample_ys)**2)
		best_fit_squared_error=sum((y_best_fit__line-self.sample_ys)**2)

		coeficient_of_determination=1-(best_fit_squared_error/mean_squared_error)

		print(coeficient_of_determination)

		return self
		

	def visualize(self,*args,**kargs):
		style.use('ggplot')

		plt.scatter(self.sample_xs,self.sample_ys,color='g')
		try:
			plt.plot(self.sample_xs,self.y_line)
			plt.scatter(self.test_x,self.test_y,color='b',s=100)
		except AttributeError as err:
			print(err)
			# was not defined in clling object		
		except Exception as inst:
			print(type(inst))
			# any exception

			
		plt.show()
		

	def generate_test_data(self,*args,**kargs):
		start_value= kargs['start'] if kargs.__contains__('start') else 1

		items= kargs['items'] if kargs.__contains__('items') else 50

		correlation= kargs['correlation'] if kargs.__contains__('correlation') else 'pos'

		variance= kargs['variance'] if kargs.__contains__('variance') else 20

		step= kargs['step'] if kargs.__contains__('step') else 2
		ys=[]
		# variance, number of data , step, correlation 

		for i in range(items): 
			y_val=start_value+random.randrange(-variance,variance)
			ys.append(y_val)
			if correlation is 'pos' or '+':
				start_value+=step
			elif correlation is 'neg' or '-':
				start_value-=step

		xs=[x for x in range(items)]

		self.sample_xs=np.array(xs,dtype=np.float64)
		self.sample_ys=np.array(ys,dtype=np.float64)
		
		return self		


LinearRegression().generate_test_data().fit().score().predict().visualize()
