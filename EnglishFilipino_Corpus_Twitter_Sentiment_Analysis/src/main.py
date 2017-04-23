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


# VOTING ALGORITHM
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

# READ ENGLISH FILIPINO MERGED CORPUS
file = open("../corpus/enflcorpus.txt", "r", encoding="utf8")
cp = file.read().split("\n")
file.close()

# READ SENTIMENT CORPUS
file = open("../corpus/sentimentcorpus.txt", "r", encoding="utf8")
file_r = file.read()
st = []
neu = 0
pos = 0
neg = 0
for r in file_r.split("\n"):
    if r == ',':
        st.append("neu")
        neu += 1
    elif r == '1':
        st.append("pos")
        pos += 1
    else:
        st.append("neg")
        neg += 1
file.close()
st += st

# TUPLE OF (SENTENCE, SENTIMENT)
documents = []
for i in range(0, len(cp)):
    if st[i] != "neu":
        documents.append((list(word_tokenize(cp[i])), st[i]))

random.shuffle(documents)

all_words = []
for s in cp:
    for w in word_tokenize(s):
        all_words.append(w.lower())

# FREQUENCY DISTRIBUTION OF MOST COMMON WORDS IN THE CORPUS
all_words = nltk.FreqDist(all_words)


# GET TOP 5000 MOST COMMON WORDS FROM CORPUS
word_features = []
for w in all_words.most_common(5000):
    word_features.append(w[0])

# FILTER ENGLISH STOP WORDS
enfiltered_word_features = [word for word in word_features if word not in stopwords.words('english')]

# READ FILIPINO STOP WORDS FROM FILE
file = open("../corpus/flstopwords.txt", "r", encoding="utf8")
flstopwords = file.read().split("\n")
file.close()

# FILTER FILIPINO STOP WORDS
flenfiltered_word_features = [word for word in enfiltered_word_features if word not in flstopwords]

# FILTER NUMBERS
numlist = []
for i in range(0, 100):
    numlist.append(str(i))
flenfiltered_word_features = [word for word in flenfiltered_word_features if word not in numlist]


def find_features(document):
    words = set(document)
    features = {}
    for w in flenfiltered_word_features:
        features[w] = (w in words)

    return features

featuresets = [(find_features(rev), category) for (rev, category) in documents]

# file = open('../picklefiles/featuresets.pickle', 'rb')
# # pickle.dump(featuresets, file)
# featuresets = pickle.load(file)
# file.close()

# FEATURE SET: 70% TRAINING, 30% TESTING
training_set = featuresets[:11552]
testing_set = featuresets[11552:]

#CLASSIFIERS
classifier = nltk.NaiveBayesClassifier.train(training_set)
# file = open("../picklefiles/naivebayes.pickle", "rb")
# # pickle.dump(classifier, file)
# classifier = pickle.load(file)
# file.close()
print("Original Naive Bayes ACCURACY:", (nltk.classify.accuracy(classifier, testing_set)))
classifier.show_most_informative_features(15)

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
# file = open("../picklefiles/mnb.pickle", "rb")
# # pickle.dump(MNB_classifier, file)
# MNB_classifier = pickle.load(file)
# file.close()
print("Multinomial Naive Bayes ACCURACY:", nltk.classify.accuracy(MNB_classifier, testing_set))

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
# file = open("../picklefiles/lr.pickle", "rb")
# # pickle.dump(LogisticRegression_classifier, file)
# LogisticRegression_classifier = pickle.load(file)
# file.close()
print("Logistic Regression ACCURACY:", nltk.classify.accuracy(LogisticRegression_classifier, testing_set))

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
# file = open("../picklefiles/lsvc.pickle", "rb")
# # pickle.dump(LinearSVC_classifier, file)
# LinearSVC_classifier = pickle.load(file)
# file.close()
print("Linear Support Vector Clustering ACCURACY:", nltk.classify.accuracy(LinearSVC_classifier, testing_set))

voted_classifier = VoteClassifier(MNB_classifier, LogisticRegression_classifier, LinearSVC_classifier)
print("Voted Classifier ACCURACY:", nltk.classify.accuracy(voted_classifier, testing_set))

