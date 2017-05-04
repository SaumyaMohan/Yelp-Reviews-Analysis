import pandas
import nltk
#nltk.download()
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import  accuracy_score
import re
import numpy

def main():

	training = pandas.read_csv("smalltrain.txt", header=0, delimiter="\t", quoting=3)
	testing = pandas.read_csv("smalltest.txt", header=0, delimiter="\t", quoting=3)
	testing_reviews = []
	training_reviews = []


	for i in xrange( 0, len(training["stars"])):
		if numpy.isnan(training["stars"][i]):
			del training["stars"][i]
			del training["review"][i]
	for i in xrange( 0, len(testing["stars"])):
		if numpy.isnan(testing["stars"][i]):
			del testing["stars"][i]
			del testing["review"][i]

	print "Cleaning and parsing the training set \n"
	#for i in xrange( 0, len(training["review"]))
	count = 0
	for i in training["review"].iteritems():
		if( (count+1) % 1000 == 0 ):
			print "Review %d of %d\n" % (count+1, len(training["review"]))

		training_reviews.append(clean_review(training["review"][i[0]]))
		count+=1

	print "Creating the bag of words...\n"
	vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 7000)

	print "Training the model...\n"
	training_features = vectorizer.fit_transform(training_reviews)
	numpy.asarray(training_features)

	
	print "Training the random forest..."
	forest = RandomForestClassifier(n_estimators = 100)
	forest = forest.fit( training_features, training["stars"] )

	print "Cleaning and parsing the testing set \n"
	count2 = 0
	for i in testing["review"].iteritems():
		if( (count2+1) % 1000 == 0 ):
			print "Review %d of %d\n" % (count2+1, len(testing["review"]))
		testing_reviews.append(clean_review(testing["review"][i[0]]))
		count2+=1

	testing_features = vectorizer.transform(testing_reviews)
	numpy.asarray(testing_features)

	#result = []

	result = forest.predict(testing_features)

	rs = map

	output = pandas.DataFrame( data={"review":testing["review"], "stars":result} )

	auc = accuracy_score(testing['stars'], result)

	print "the auc is "
	print auc

	#output.to_csv("Bag_of_Words_model.csv", index=False, quoting=3)
	with open('Bag_of_Words_model2.txt', 'w') as g:
		g.write(output.to_string().encode('utf-8'))
	g.close() 

	#print "size of test review %d ...\n" % (len(testing["review"]))
	#print "size of result stars %d... \n" % (len(result))


def clean_review(input):
	#if numpy.isnan(training['stars'][index]):
	#	return " "
	uni = input.decode('utf-8')
	words = uni.lower().split()
	stop_words = set(stopwords.words("english"))
	clean_words = [i for i in words if not i in stop_words]
	stemmer=PorterStemmer()
	stemmed = [stemmer.stem(w) for w in clean_words]
	return( " ".join( stemmed ))   


if __name__ == '__main__':
	main()