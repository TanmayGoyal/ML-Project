import matplotlib.pyplot as plt
import numpy as np
import operator
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import string

def plot_1():#frequency of each class
	f = open("training_variants","r")
	line = f.readline()
	line = f.readline().split(",")

	class_freq = [0 for i in range(9)]

	while line!=['']:
		class_freq[int(line[-1].strip())-1]+=1
		line = f.readline().split(",")

	class_label = [str(i+1)+"\n("+str(class_freq[i])+")" for i in range(9)]

	y_pos = np.arange(len(class_label))
	 
	plt.bar(y_pos, class_freq, align='center', alpha=0.5)
	plt.xticks(y_pos, class_label)
	plt.xlabel('Classes')
	plt.title("Frequency of each Class")
	plt.tight_layout()
	plt.show()

def plot_2():#top 10 genes
	f = open("training_variants","r")
	line = f.readline()
	line = f.readline().split(",")

	gene_freq={}

	while line!=['']:
		if line[1] in gene_freq:
			gene_freq[line[1]]+=1
		else:
			gene_freq[line[1]]=1
		line = f.readline().split(",")

	sort_gene= sorted(gene_freq.items(), key=operator.itemgetter(1))
	top_genes = sort_gene[-1:-11:-1]
	# top_genes = sort_gene[0:10:1]
	# print top_genes
	gene=[]
	freq=[]
	for i in top_genes:
		gene.append(i[0]+"\n("+str(i[1])+")")
		freq.append(i[1])

	y_pos = np.arange(len(gene))
	 
	plt.bar(y_pos, freq, align='center', alpha=0.5)
	plt.xticks(y_pos, gene)
	plt.xlabel('Genes')
	plt.title("Top 10 Genes")
	plt.tight_layout()
	plt.show()	


def plot_3():#top 10 genes per class
	f = open("training_variants","r")
	line = f.readline()
	line = f.readline().split(",")

	gene_per_class=[{} for i in range(9)]

	while line!=['']:
		if line[1] in gene_per_class[int(line[-1].strip())-1]:
			gene_per_class[int(line[-1].strip())-1][line[1]]+=1
		else:
			gene_per_class[int(line[-1].strip())-1][line[1]]=1
		line = f.readline().split(",")

	for i in range(len(gene_per_class)):
		plt.figure(i+1)
		gene_freq = gene_per_class[i]

		sort_gene= sorted(gene_freq.items(), key=operator.itemgetter(1))
		top_genes = sort_gene[-1:-11:-1]
		# top_genes = sort_gene[0:10:1]
		# print top_genes
		gene=[]
		freq=[]
		for j in top_genes:
			gene.append(j[0]+"\n("+str(j[1])+")")
			freq.append(j[1])

		y_pos = np.arange(len(gene))
		 
		plt.bar(y_pos, freq, align='center', alpha=0.5)
		plt.xticks(y_pos, gene)
		plt.xlabel('Genes')
		plt.title("Top 10 Genes for Class "+str(i+1))
		plt.tight_layout()
		name = "plots/"+"top_gene_"+str(i+1)+".png"
		plt.savefig(name)
		# plt.show()


def plot_4():#top words in clinical text
	f = open("training_text","r")
	line = f.readline()
	line = f.readline().split("||")
	sw = stopwords.words("english")
	sw+=["fig", "figure", "et", "al", "table", "data", "analysis", "also","analyze", "study", "method", "result", "conclusion","use", "author", "find", "found", "show", "perform", "demonstrate", "evaluate", "discuss"]
	stemmer = SnowballStemmer("english")
	word_freq={}
	# f2=open("processes.txt","w")
	k=0
	while line!=['']:
		clin_text=""
		s = line[1].translate(string.maketrans("",""), string.punctuation)
		words = s.split()
		for i in words:
			temp=""
			try:
				temp = stemmer.stem(i)
			except:
				temp = i
			if i in sw or temp in sw:
				pass
			elif i.isdigit()==True:
				pass
			else:
				try:
					clin_text+=temp+" "
				except:
					pass
				if temp in word_freq:
					word_freq[temp]+=1
				else:
					word_freq[temp]=1
		# f2.write(str(k)+"||"+clin_text+"\n")
		line = f.readline().split("||")
		print k
		k+=1
		# print word_freq
	sort_words= sorted(word_freq.items(), key=operator.itemgetter(1))
	top_words = sort_words[-1:-21:-1]
	for i in top_words:
		print i

	word = []
	freq = []
	for j in top_words:
		word.append(j[0]+" ("+str(j[1])+")")
		freq.append(j[1])

	y_pos = np.arange(len(word))
	 
	plt.bar(y_pos, freq, align='center', alpha=0.5)
	plt.xticks(y_pos, word, rotation=90)
	plt.xlabel('Words')
	plt.title("Top Words in Clinical Text")
	plt.tight_layout()
	plt.show()


# plot_1()
# plot_2()
# plot_3()
# plot_4()