import pandas
import nltk
#nltk.download()
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import  accuracy_score, confusion_matrix, classification_report
import re
import numpy

def main():

	#using previously "cleaned" reviews
	#the original reviews were stemmed and the stop words were removed
	#and stored for consistent use across different classifiers
	training = pandas.read_csv("200000_clean_train_reviews.txt", header=0, delimiter="\t", quoting=3)
	testing = pandas.read_csv("50000_clean_test_reviews.txt", header=0, delimiter="\t", quoting=3)

	#using "uncleaned" reviews
	#these are the top 200000 training reviews and top 50000 test reviews
	#training = pandas.read_csv("200ktrain.txt", header=0, delimiter="\t", quoting=3)
	#testing = pandas.read_csv("200ktest.txt", header=0, delimiter="\t", quoting=3)

	#we used two different vectorizers
	#the tfidf vectorizer is the same as the count vectorizer 
	#followed by the tfidf transformer
	#the laplace smoothing attribute is true by default  
	print "Creating the bag of words"
	vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 8000)
	#vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features = 8000)

	#learn the vocabulary and fit the model
	#then convert to a numpy array
	print "Training the model"
	training_features = vectorizer.fit_transform(training["review"])
	numpy.asarray(training_features)

	
	#initialize the classifier and fit it with the features and stars
	print "Training the random forest"
	forest = RandomForestClassifier(n_estimators = 100)
	forest = forest.fit( training_features, training["stars"] )

	#transform the testing reviews into feature vectors and convert to a numoy array
	testing_features = vectorizer.transform(testing['review'])
	numpy.asarray(testing_features)

	#predict the classes for each testing review based on the feature vector
	result = forest.predict(testing_features)

	#output to be printed in the form of a DataFrame
	output = pandas.DataFrame( data={"review":testing["review"], "stars":result} )

	#calculate the accuracy score by comparing the predicted classes to the actual
	acc = forest.score(testing_features, testing['stars'])
	print "The accuracy is {}." .format(acc)

	#generate a classification report based on the classes (stars)
	#prints the precision, recall and f1 score for each class
	print "\nClassification Report:"
	print(classification_report(testing['stars'], result))

	#generate a print a confusion matrix for each class
	print "\nConfusion Matrix:"
	print(confusion_matrix(testing['stars'], result))

	#print the model to a file 
	with open('Random_Forest_Model.txt', 'w') as g:
		g.write(output.to_string().encode('utf-8'))
	g.close() 


#this code was used to clean the reviews
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