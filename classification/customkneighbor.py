import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import style
import pickle , math
import pandas as pd 
# from classifier import classifier
import random
from collections import Counter


class CustomKneighbor:
	def __init__(self,*args,**kargs):

		self.sample_data={'r':[[1,2],[4,3],[2,4]],'k':[[7,8],[8,7],[8,9]]}

		self.test_data=[10,11]


		print('welcome custom neigbor')

		pass

	def get_data(self,*args,**kargs):
		df = pd.read_csv('breast-cancer-wisconsin.data.txt')
		df.replace(['?'],-99999,inplace=True)
		df.drop(['id'],1,inplace=True)	

		self.sample_data={2:[],4:[]}

		data=df.astype(float).values.tolist()


		for group in data:
			self.sample_data[group[-1]].append(group[:-1])

		return self	




	def fit(self,*args,**kargs):
		k = kargs['k'] if kargs.__contains__('k') else 3
		predict=kargs['predict'] if kargs.__contains__('predict') else self.test_data
		record=[]
		for group in self.sample_data:
			for data in self.sample_data[group]:
				eucledean_distance=np.linalg.norm(np.array(data)-np.array(predict))
				record.append([eucledean_distance,group])
		
		shortest_distance=[val[1] for val in sorted(record)[:k]]

		self.selected_group=Counter(shortest_distance).most_common()[0][0]
		print(self.selected_group)

		return self

	def score(self,*args,**kargs):
		pass

	def predict(self,*args,**kargs):

		pass
		
	def visualize(self,*args,**kargs):
		style.use('ggplot')
		[[plt.scatter(data[0],data[1], color=group) for data in  self.sample_data[group] ]for group in self.sample_data]

		try:
			plt.scatter(self.test_data[0],self.test_data[1],color=self.selected_group)
		except Exception:
			plt.scatter(self.test_data[0],self.test_data[1],color='b')


		

		plt.show()
		

CustomKneighbor().get_data().fit(predict=[4,1,2,1,2,7,5,4,4])