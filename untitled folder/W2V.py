import pandas
import nltk
import os
import logging
#nltk.download()
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import  accuracy_score, classification_report
from nltk.corpus import stopwords
import re
import numpy

def main():

	#Read in the data
	training = pandas.read_csv("smalltrain.txt", header=0, delimiter="\t", quoting=3)
	testing = pandas.read_csv("smalltest.txt", header=0, delimiter="\t", quoting=3)

	#punkt tokenizer from nltk
	tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

	#Convert the review into a list of sentences after cleaning each sentence and using the punkt tokenizer
	sentences =[]
	print "Parsing training sentences"
	for review in training['review']:
		sentences += review_to_sentences(review, tokenizer)

	#Print logs
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

	#Parameters
	num_features = 300    # Word vector dimensionality
	min_word_count = 40   # Minimum word count
	num_workers = 4       # Number of threads to run in parallel
	context = 10          # Context window size
	downsampling = 1e-3   # Downsample setting for frequent words

	#Initialize the w2v model
	print "Training the Word2Vec model"
	model = Word2Vec(sentences, workers=num_workers, size=num_features, min_count = min_word_count, window = context, sample = downsampling, seed=1)

	model.init_sims(replace=True)

	#Save the model for future use
	model_name = "300features_40minwords_10context"
	model.save(model_name)

	print "Creating average feature vecs for training reviews"
	trainDataVecs = getAvgFeatureVecs( getCleanReviews(training), model, num_features )

	print "Creating average feature vecs for test reviews"
	testDataVecs = getAvgFeatureVecs( getCleanReviews(testing), model, num_features )

	#initialize the classifier and fit it with the features and stars
	forest = RandomForestClassifier( n_estimators = 100 )

	print "Fitting a random forest"
	forest = forest.fit( trainDataVecs, training["stars"] )

	print "Predicting test vecs"
	# predict the classes for the training data
	result = forest.predict( testDataVecs )


	# Write the test results
	output = pandas.DataFrame( data={"review":testing["review"], "sentiment":result} )

	with open('Word2Vec_AverageVectors.txt', 'w') as g:
		g.write(output.to_string().encode('utf-8'))
	g.close() 

	#calculate accuracy of the testing data
	acc = accuracy_score(testing['stars'], result)

	print "The accuracy is {}\n".format(acc)
	print "\nClassification Report:"
	print(classification_report(testing['stars'], result))


#returns a list of lowercase words for each input sentence. 
#Since word 2 vec looks at words in context of the whole sentence, the stop words are not removed and the words are not stemmed
def review_to_wordlist(review, remove_stopwords=False):
	review_text = re.sub('n\'t', ' not', review)
	words = review_text.lower().split()
	if remove_stopwords:
		negators = ['no', 'not', 'neither', 'never', 'none', 'nobody', 'nor', 'nothing', 'hardly', 'rarely', 'seldom', 'scarcely', 'cannot']
		stop_words = set(stopwords.words("english"))
		words = [i for i in words if not i in stop_words or i in negators]
	return(words)   

#returns a list of lists of words
def review_to_sentences(review, tokenizer, remove_stopwords=False):
	raw_sentences = tokenizer.tokenize(review.decode('utf8').strip())
	sentences = []
	for rs in raw_sentences:
		if len(rs) > 0:
			sentences.append(review_to_wordlist(rs, remove_stopwords))
	return sentences


def getCleanReviews(reviews):
	clean_reviews = []
	for review in reviews["review"]:
		clean_reviews.append(review_to_wordlist(review, remove_stopwords=True))
	return clean_reviews

#creates the average feature vectors for each review
def makeFeatureVec(words, model, num_features):
	featureVec = numpy.zeros((num_features,), dtype="float32")
	nwords=1.
	index2word_Set = set(model.wv.index2word)
	for word in words:
		if word in index2word_Set:
			nwords = nwords +1.
			featureVec = numpy.add(featureVec, model[word])
	featureVec = numpy.divide(featureVec, nwords)
	return featureVec

#returns the average feature vectors of all reviews
def getAvgFeatureVecs(reviews, model, num_features):
	counter = 0.
	reviewFeatureVecs = numpy.zeros((len(reviews), num_features), dtype='float32')
	for review in reviews:
		if counter%1000. == 0.:
			print "review %d of %d" %(counter, len(reviews))
		reviewFeatureVecs[int (counter)] = makeFeatureVec(review, model, num_features)
		counter = counter + 1.
	return reviewFeatureVecs

if __name__ == '__main__':
	main()