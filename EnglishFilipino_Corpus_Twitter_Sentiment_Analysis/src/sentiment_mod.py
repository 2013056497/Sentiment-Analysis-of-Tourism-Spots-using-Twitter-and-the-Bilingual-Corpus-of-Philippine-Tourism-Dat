import nltk
import random
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from nltk.classify import ClassifierI
from statistics import mode


class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

file = open('../picklefiles/documents.pickle', 'rb')
documents = pickle.load(file)
file.close()

file = open('../picklefiles/flenfiltered_word_features.pickle', 'rb')
flenfiltered_word_features = pickle.load(file)
file.close()

def find_features(document):
    words = set(document)
    features = {}
    for w in flenfiltered_word_features:
        features[w] = (w in words)

    return features

featuresets = [(find_features(rev), category) for (rev, category) in documents]
# file = open('../picklefiles/featuresets.pickle', 'rb')
# featuresets = pickle.load(file)
# file.close()

# FEATURE SET: 70% TRAINING, 30% TESTING
training_set = featuresets[:11552]
testing_set = featuresets[11552:]

# file = open("../picklefiles/naivebayes.pickle", "rb")
# classifier = pickle.load(file)
# file.close()
# print("Original Naive Bayes ACCURACY:", (nltk.classify.accuracy(classifier, testing_set)))
# classifier.show_most_informative_features(15)

file = open("../picklefiles/mnb.pickle", "rb")
MNB_classifier = pickle.load(file)
file.close()
# print("Multinomial Naive Bayes ACCURACY:", nltk.classify.accuracy(MNB_classifier, testing_set))

file = open("../picklefiles/lr.pickle", "rb")
LogisticRegression_classifier = pickle.load(file)
file.close()
# print("Logistic Regression ACCURACY:", nltk.classify.accuracy(LogisticRegression_classifier, testing_set))

file = open("../picklefiles/lsvc.pickle", "rb")
LinearSVC_classifier = pickle.load(file)
file.close()
# print("Linear Support Vector Clustering ACCURACY:", nltk.classify.accuracy(LinearSVC_classifier, testing_set))

voted_classifier = VoteClassifier(MNB_classifier, LogisticRegression_classifier, LinearSVC_classifier)
# print("Voted Classifier ACCURACY:", nltk.classify.accuracy(voted_classifier, testing_set))

def sentiment(text):
    feats = find_features(text)

    return voted_classifier.classify(feats), voted_classifier.confidence(feats)

