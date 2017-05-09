import pandas
import nltk
#nltk.download()
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import  accuracy_score, classification_report, confusion_matrix
import numpy

def main():

	#using "uncleaned" reviews
	#these are the top 200000 training reviews and top 50000 test reviews
	training = pandas.read_csv("200ktrain.txt", header=0, delimiter="\t", quoting=3)
	testing = pandas.read_csv("200ktest.txt", header=0, delimiter="\t", quoting=3)

	# since these files are not cleaned, delete any erroneous/invalid lines
	for i in xrange( 0, len(training["stars"])):
		if numpy.isnan(training["stars"][i]):
			del training["stars"][i]
			del training["review"][i]
	for i in xrange( 0, len(testing["stars"])):
		if numpy.isnan(testing["stars"][i]):
			del testing["stars"][i]
			del testing["review"][i]

	#we used two different vectorizers
	#the tfidf vectorizer is the same as the count vectorizer 
	#followed by the tfidf transformer
	#the laplace smoothing attribute is true by default  	
	print "Creating the bag of words"
	vectorizer = CountVectorizer(ngram_range=(1,2), analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 8000)
	#vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features = 8000)


	#learn the vocabulary and fit the model
	#then convert to a numpy array
	print "Training the model"
	training_features = vectorizer.fit_transform(training["review"])
	numpy.asarray(training_features)

	#initialize the classifier and fit it with the features and stars
	#we tested two different classifiers: multinomial nb and bernoulli nb
	print "Training the naive bayes classifier..."
	NB = MultinomialNB()
	#NB = BernoulliNB()
	NB = NB.fit( training_features, training["stars"] )

	#transform the testing reviews into feature vectors and convert to a numoy array
	testing_features = vectorizer.transform(testing['review'])
	numpy.asarray(testing_features)

	#predict the classes for each testing review based on the feature vector
	result = NB.predict(testing_features)

	#output to be printed in the form of a DataFrame
	output = pandas.DataFrame( data={"review":testing["review"], "stars":result} )

	#calculate the accuracy score by comparing the predicted classes to the actual
	acc = NB.score(testing_features, testing['stars'])
	print "The accuracy is {}." .format(acc)

	#generate a classification report based on the classes (stars)
	#prints the precision, recall and f1 score for each class
	print "\nClassification Report:"
	print(classification_report(testing['stars'], result))

	#generate a print a confusion matrix for each class
	print "\nConfusion Matrix:"
	print(confusion_matrix(testing['stars'], result))

	#print the model to a file 
	with open('NB_Model.txt', 'w') as g:
		g.write(output.to_string().encode('utf-8'))
	g.close() 


if __name__ == '__main__':
	main()