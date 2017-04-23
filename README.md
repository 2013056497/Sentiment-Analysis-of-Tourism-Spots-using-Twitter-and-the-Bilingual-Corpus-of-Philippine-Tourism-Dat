# Sentiment-Analysis-of-Tourism-Spots-using-Twitter-and-the-Bilingual-Corpus-of-Philippine-Tourism-Dat

File Directories
corpus
*stored within this directory is the files containing the corpus and the stop words used in the study

   >enflcorpus.txt
   *English-Filipino_Tourism_Corpus_and_Lexicon_for_an_ASEAN_Language_Translation_System
   *The parrallel corpus is merged into one file

   >flstopwords
   *list of filipino stop words from: https://github.com/stopwords-iso/stopwords-tl

   >sentimentcorpus
   *this file contains the sentiment labels of the sentences parallel to the enflcorpus.txt
   *LEGEND: '1'-pos, ','-neu, '0'-neg

picklefiles
*pickling is the way for python to save large objects into files to save run time from training classifiers and instantiating large variables

   >lr, lsvc, mnb, and naivebayes .pickle
   *are classifiers in pickle files

   >documents
   *array of senteces in enflcorpus along with its label (pos/neg)

   >all_words.pickle
   *variable object that contains all the words in enflcorpus

   >flenfiltered_word_features
   *are word features with english and filipino stop words removed

src
*python scripts

   >main.py
   *used to get the accuracy of the classfiers
   *WARNING LONG RUNTIME
   *DEPENDENCIES(PYTHON LIBRARIES NEEDED):
       -nltk
       -sklearn *(many dependencies in itself, I suggest you download anaconda: https://www.continuum.io/downloads)

   >sentiment_mod.py
   *is used by tweer_classification.py to classify tweets
   *use pickled variables to increase run time

   >tweet_classfication.py
   *run to get the tweet sentiment and the most common words from tourist spot tweets

>tweets
*tweets filtered from PH tourist spots
