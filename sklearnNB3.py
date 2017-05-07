import pandas
import nltk
#nltk.download()
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import  accuracy_score
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
import re
import numpy

def main():

	training = pandas.read_csv("200ktrain.txt", header=0, delimiter="\t", quoting=3)
	testing = pandas.read_csv("200ktest.txt", header=0, delimiter="\t", quoting=3)

	for i in xrange( 0, len(training["stars"])):
		if numpy.isnan(training["stars"][i]):
			del training["stars"][i]
			del training["review"][i]
	for i in xrange( 0, len(testing["stars"])):
		if numpy.isnan(testing["stars"][i]):
			del testing["stars"][i]
			del testing["review"][i]




	pipeline = Pipeline([('count_vectorizer', \
		CountVectorizer(ngram_range=(1, 2), analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 8000)),\
		  ('classifier', BernoulliNB(binarize=0.0)) ])


	print "Training the model...\n"

	
	print "Training the naive bayes..."


	pipeline.fit(training['review'], training['stars'])

	predictions= pipeline.predict(testing['review'])

	acc = pipeline.score(testing['review'], testing['stars'])
	#print predictions

	#output = pandas.DataFrame( data={"review":testing["review"], "stars":result} )

	#auc = accuracy_score(testing['stars'], result)

	print "the acc is "
	print acc

	#output.to_csv("Bag_of_Words_model.csv", index=False, quoting=3)
	#with open('Bag_of_Words_model3.txt', 'w') as g:
	#	g.write(output.to_string().encode('utf-8'))
	#g.close() 

	#print "the macro f score is .."
	#print f1_score(testing['stars'], result, average='macro') 

	#print "size of test review %d ...\n" % (len(testing["review"]))
	#print "size of result stars %d... \n" % (len(result))


if __name__ == '__main__':
	main()