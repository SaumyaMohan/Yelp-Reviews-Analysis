from textblob.classifiers import NaiveBayesClassifier
import pandas
import numpy
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

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

	count = 0
	for i in training["review"].iteritems():
		if( (count+1) % 1000 == 0 ):
			print "Review %d of %d\n" % (count+1, len(training["review"]))

		#training_reviews.append((clean_review(training["review"][i[0]]), training["stars"][i[0]]))
		training_reviews.append((training["review"][i[0]].decode('utf-8'), training["stars"][i[0]]))
		if count == 1000:
			break
		count+=1

	print "Cleaning and parsing the testing set \n"
	count2 = 0
	for i in testing["review"].iteritems():
		if( (count2+1) % 1000 == 0 ):
			print "Review %d of %d\n" % (count2+1, len(testing["review"]))
		testing_reviews.append((testing["review"][i[0]].decode('utf-8'), testing["stars"][i[0]]))
		if count2 == 250:
			break
		count2+=1

	print "Initializing classifier \n"

	cl = NaiveBayesClassifier(training_reviews)

	print "Classifying test reviews \n"
	#result = []
	#for review in testing_reviews
	#	result.append(cl.classify(review))

	print cl.accuracy(testing_reviews)


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