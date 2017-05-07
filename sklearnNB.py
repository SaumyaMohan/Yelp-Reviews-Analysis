import pandas
import nltk
#nltk.download()
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import  accuracy_score
import re
import numpy

def main():

	training = pandas.read_csv("200000_clean_train_reviews.txt", header=0, delimiter="\t", quoting=3)
	testing = pandas.read_csv("50000_clean_test_reviews.txt", header=0, delimiter="\t", quoting=3)

	print "Creating the bag of words...\n"
	vectorizer = CountVectorizer(ngram_range=(1, 2),analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 5000)

	print "Training the model...\n"
	#training_features = vectorizer.fit_transform(training_reviews)
	training_features = vectorizer.fit_transform(training['review'])
	numpy.asarray(training_features)

	
	print "Training the random forest..."
	forest = MultinomialNB()
	forest = forest.fit( training_features, training["stars"] )

	testing_features = vectorizer.transform(testing['review'])
	numpy.asarray(testing_features)

	#result = []

	result = forest.predict(testing_features)

	acc = forest.score(testing_features, testing['stars'])

	output = pandas.DataFrame( data={"review":testing["review"], "stars":result} )

	#auc = accuracy_score(testing['stars'], result)

	print "the auc is "
	print acc

	#output.to_csv("Bag_of_Words_model.csv", index=False, quoting=3)
	with open('Bag_of_Words_model3.txt', 'w') as g:
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