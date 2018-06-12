import pickle

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