from sklearn.svm import SVC
import sys
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.svm import SVC
from sklearn import svm, datasets
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt


def graph(X,Y):
	# newsgroups_train = fetch_20newsgroups(subset='train',categories=['alt.atheism', 'sci.space'])
	# print newsgroups_train.data
	# pipeline = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),])
	# X = pipeline.fit_transform(newsgroups_train.data).todense()
	print X.shape
	X=X.todense()
	print X.shape
	pca = PCA(n_components=2).fit(X)
	data2D = pca.transform(X)
	print data2D
	plt.scatter(data2D[:,0], data2D[:,1],c=Y)
	plt.colorbar(ticks=range(10))
	# plt.show()              #not required if using ipython notebook
	plt.savefig("graph.png")

def get_file_data(str,delim):
	with open(str) as data:
		arr = data
		arr = arr.read()

		arr = arr.split('\n')

	for i in range (len(arr)):
		arr[i] = arr[i].split(delim)


	return arr

def train(train_var,train_text):
	x = []
	y = []

	foo = 0

	for i in train_var:
		y.append(str(i[-1]))

		i.pop(-1)
		i.pop(0)

		# i.append(train_text[foo][-1].split(" "))

		foo+=1
		x.append(str(i))
		# x.append(str(train_text[foo][-1].split(" ")))

	# print x[0]
	# print y[0]
	# print x[0]'
	# print type(x), type(x[0][0])


	tfidf_vect = TfidfVectorizer(stop_words='english')

	X_train = tfidf_vect.fit_transform(x)

	# X_train_tf = tf_transformer.transform(matrix1)
	# X_train_tfidf.shape
	return X_train,y,tfidf_vect


def test(tfidf_vect,arr,arr2):
	x = []

	for i in arr:
		i.append(arr2[-1])
		x.append(str(i))

	return tfidf_vect.transform(x)



def classifier(train_1,train_2,test):

	clf = SVC(kernel = 'linear', probability = True)

	for i in range(len(train_2)):
		train_2[i]=int(train_2[i])

	clf.fit(train_1,train_2)

	print "ID,class1,class2,class3,class4,class5,class6,class7,class8,class9"

	predictions = clf.predict_proba(test)

	for i in range (0,len(predictions)):
		j=i+1
		stri = ','.join(str(x) for x in predictions[i])

		print str(j) + ',' + stri


def main ():


	train_text = get_file_data('training_text','||')
	test_text = get_file_data('stage2_test_text.csv','||')

	train_var = get_file_data('training_variants',',')
	test_var = get_file_data('stage2_test_variants.csv',',')


	graph(x_train,y_train)
	x_train,y_train,tfidf_vect = train(train_var,train_text)
	# print gene,mutation
	x_test = test(tfidf_vect,test_var,test_text)

	# classifier(x_train,y_train,x_test)

main ()