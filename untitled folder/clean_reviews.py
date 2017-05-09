import pandas
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy

#this program was used to "clean" and store 200000 training reviews and 50000 
#testing reviews
def main():
	training = pandas.read_csv("train.txt", header=0, delimiter="\t", quoting=3)
	testing = pandas.read_csv("test.txt", header=0, delimiter="\t", quoting=3)
	testing_reviews = []
	training_reviews = []

	print "Cleaning and parsing the training set"
	for i in xrange( 0, len(training["review"])):
		if i > 200000:
			break
		if( (i+1) % 1000 == 0 ):
			print "Review %d of %d\n" % (i+1, len(training["review"]))
		if numpy.isnan(training["stars"][i]):
			continue
		cl = clean_review(training["review"][i]) + '\t %d' % (training["stars"][i])
		training_reviews.append(cl)

	thefile = open('200000_clean_train_reviews.txt', 'w')
	for item in training_reviews:
		print>>thefile, item.encode('utf-8')


	print "Cleaning and parsing the testing set"
	for i in xrange( 0, len(testing["review"])):
		if i > 50000:
			break
		if( (i+1) % 1000 == 0 ):
			print "Review %d of %d\n" % (i+1, len(testing["review"]))
		if numpy.isnan(testing["stars"][i]):
			continue
		cl = clean_review(testing["review"][i]) + '\t %d' % (testing["stars"][i])
		testing_reviews.append(cl)

	thefile = open('50000_clean_test_reviews.txt', 'w')
	for item in testing_reviews:
		print>>thefile, item.encode('utf-8')

#convert each review into a list of lowercase words
#for each word in the list, remove stop words and stem the word
#return a sentence made of joining the clean words
def clean_review(input):
	uni = input.decode('utf-8')
	words = uni.lower().split()
	stop_words = set(stopwords.words("english"))
	clean_words = [i for i in words if not i in stop_words]
	stemmer=PorterStemmer()
	stemmed = [stemmer.stem(w) for w in clean_words]
	return( " ".join( stemmed ))

if __name__ == '__main__':
	main()