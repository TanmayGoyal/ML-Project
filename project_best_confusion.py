from sklearn.svm import SVC
import sys
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.svm import SVC
from sklearn import svm, datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt2


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
		foo+=1

		# i.append(train_text[foo][-1].split(" "))

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
		i.pop(0)
		i.append(arr2[-1])
		x.append(str(i))

	return tfidf_vect.transform(x)



def classifier(train_1,train_2,test):

	# print "hi"
	clf = SVC(kernel = 'linear', probability = True)

	# print (train_1).shape, len(train_2)

	# for i in range(train_1.shape[0]):
	# 	for j in range(train_1[i].shape[0]):
	# 		print type(train_1[i][j])
	# 		if train_1[i][j]!=0:
	# 			print train_1[i][j]
			# print train_1[i][j]
	# clf = svm.LinearSVC()
	for i in range(len(train_2)):
		train_2[i]=int(train_2[i])
	# print train_1
	clf.fit(train_1,train_2)
	# print "hey"
	# arr = clf.predict(test)
	# print clf.decision_function()

	# # print "yo"
	# print "ID,class1,class2,class3,class4,class5,class6,class7,class8,class9"

	# predictions = clf.predict_proba(test)


	# for i in range (0,len(predictions)):
	# 	# for k in range (0,len(predictions[i])):
	# 	# 	predictions[i][k] = str(predictions[i][k])

	# 	# print predictions[i]

	# 	# break

	# 	# print arr[i]

	# 	# arr2[int(arr[i])-1] = '1'
	# 	j=i+1
	# 	# print arr2
	# 	stri = ','.join(str(x) for x in predictions[i])

	# 	print str(j) + ',' + stri

	# 	# break

	# 	# print str(i) + ',' + str(arr[i])


	return clf.predict(test)



	# # print "hi"
	# clf = LinearSVC()

	# clf.fit(train_1,train_2)
	# # print "hey"

	# arr = clf.predict(test)

	# # print "yo"
	# print "ID,class1,class2,class3,class4,class5,class6,class7,class8,class9"

	# for i in range (0,len(arr)):
		
	# 	arr2 = ['0']*9
	# 	# print arr[i]

	# 	arr2[int(arr[i])-1] = '1'
	# 	j=i+1
	# 	# print arr2
	# 	stri = ','.join(arr2)

	# 	print str(j) + ',' + stri



	# 	# print str(i) + ',' + str(arr[i])
def confuse_graph(mat):

	conf_arr = mat
	conf_arr = np.array(conf_arr)
	norm_conf = []
	for i in conf_arr:
	    a = 0
	    tmp_arr = []
	    a = sum(i, 0)
	    for j in i:
	        tmp_arr.append(float(j)/float(a))
	    norm_conf.append(tmp_arr)

	fig = plt2.figure()
	plt2.clf()
	ax = fig.add_subplot(111)
	ax.set_aspect(1)
	res = ax.imshow(np.array(norm_conf),  interpolation='nearest')

	width, height = conf_arr.shape

	for x in xrange(width):
	    for y in xrange(height):
	        ax.annotate(str(conf_arr[x][y]), xy=(y, x),horizontalalignment='center',verticalalignment='center')

	# cb = fig.colorbar(res)
	alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
	plt2.xticks(range(width), alphabet[:width])
	plt2.yticks(range(height), alphabet[:height])
	plt2.show()



def main ():


	train_text = get_file_data('training_text','||')
	test_text = get_file_data('stage2_test_text.csv','||')

	train_var = get_file_data('training_variants',',')
	test_var = get_file_data('stage2_test_variants.csv',',')


	# train_text = get_train_text()

	x_train,y_train,tfidf_vect = train(train_var,train_text)

	x_test = test(tfidf_vect,test_var,test_text)
	# x=0
	x_train, x_test, y_train, y_test = train_test_split(x_train,y_train,test_size=0.1)


	y_pred = classifier(x_train,y_train,x_test)

	print type(y_test[0]), type(y_pred[0])

	for i in range (len(y_test)):
		y_test[i] = int(y_test[i])
	confuse_graph(confusion_matrix(y_test, y_pred))


main ()