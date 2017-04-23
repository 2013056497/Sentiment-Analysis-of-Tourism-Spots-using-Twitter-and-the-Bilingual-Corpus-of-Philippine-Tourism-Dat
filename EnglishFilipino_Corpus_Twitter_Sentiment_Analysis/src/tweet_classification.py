import nltk
from src import sentiment_mod as s


# method for getting sentiment from tweets
def get_sentiments(tweets):
    sentiments = []
    sent_tuple = []
    for t in tweets:
        sentiments.append(s.sentiment(t)[0])
        sent_tuple.append((s.sentiment(t)[0], t))

    return sentiments.count("pos") / len(sentiments), sent_tuple


# method for getting the most common words
def get_all_words(tweetlist):
    all_words = []
    for s in tweetlist:
        for w in nltk.word_tokenize(s):
            all_words.append(w.lower())
    return all_words


print("POSITIVE SENTIMENT PERCENTAGE:\nnumber of positive sentiments/ total number of tweets(100)")
# MANILA
file = open('../tweets/manila.txt', 'r', encoding='utf8')
manila_tweets = file.read().split('\ntweet: ')
file.close()
manila_sentperc, manila_sent = get_sentiments(manila_tweets)
# print(manila_sent[0])
print('MANILA: ', manila_sentperc)

# TAGAYTAY
file = open('../tweets/tagaytay.txt', 'r', encoding='utf8')
tagaytay_tweets = file.read().split('\ntweet: ')
file.close()
tagaytay_sentperc, tagaytay_sent = get_sentiments(tagaytay_tweets)
# print(tagaytay_sent[0])
print('TAGAYTAY: ',tagaytay_sentperc)

# VIGAN
file = open('../tweets/vigan.txt', 'r', encoding='utf8')
vigan_tweets = file.read().split('\ntweet: ')
file.close()
vigan_sentperc, vigan_sent = get_sentiments(vigan_tweets)
# print(vigan_sent[0])
print('VIGAN: ',vigan_sentperc)

# PALAWAN
file = open('../tweets/palawan.txt', 'r', encoding='utf8')
palawan_tweets = file.read().split('\ntweet: ')
file.close()
palawan_sentperc, palawan_sent = get_sentiments(palawan_tweets)
# print(palawan_sent[0])
print('PALAWAN: ',palawan_sentperc)

# MAKATI
file = open('../tweets/makati.txt', 'r', encoding='utf8')
makati_tweets = file.read().split('\ntweet: ')
file.close()
makati_sentperc, makati_sent = get_sentiments(makati_tweets)
# print(palawan_sent[0])
print('MAKATI: ',makati_sentperc)

# DAVAO
file = open('../tweets/davao.txt', 'r', encoding='utf8')
davao_tweets = file.read().split('\ntweet: ')
file.close()
davao_sentperc, davao_sent = get_sentiments(davao_tweets)
# print(palawan_sent[0])
print('DAVAO: ',davao_sentperc)

# CEBU
file = open('../tweets/cebu.txt', 'r', encoding='utf8')
cebu_tweets = file.read().split('\ntweet: ')
file.close()
cebu_sentperc, cebu_sent = get_sentiments(cebu_tweets)
# print(palawan_sent[0])
print('CEBU: ', cebu_sentperc)

# BORACAY
file = open('../tweets/boracay.txt', 'r', encoding='utf8')
boracay_tweets = file.read().split('\ntweet: ')
file.close()
boracay_sentperc, boracay_sent = get_sentiments(boracay_tweets)
# print(palawan_sent[0])
print('BORACAY: ', boracay_sentperc)

# BOHOL
file = open('../tweets/bohol.txt', 'r', encoding='utf8')
bohol_tweets = file.read().split('\ntweet: ')
file.close()
bohol_sentperc, bohol_sent = get_sentiments(bohol_tweets)
# print(palawan_sent[0])
print('BOHOL: ', bohol_sentperc)

# SUBIC
file = open('../tweets/subic.txt', 'r', encoding='utf8')
subic_tweets = file.read().split('\ntweet: ')
file.close()
subic_sentperc, subic_sent = get_sentiments(subic_tweets)
print('SUBIC: ', subic_sentperc)

# set to top 50, adjust in most_common() method
print("\nMOST COMMON WORDS\n('word', number of times appeared in tweets)")
bohol_tweets = nltk.FreqDist(get_all_words(bohol_tweets))
print('BOHOL:',bohol_tweets.most_common(50))
boracay_tweets = nltk.FreqDist(get_all_words(boracay_tweets))
print('BORACAY:',boracay_tweets.most_common(50))
cebu_tweets = nltk.FreqDist(get_all_words(cebu_tweets))
print('CEBU:',cebu_tweets.most_common(50))
davao_tweets = nltk.FreqDist(get_all_words(davao_tweets))
print('DAVAO:',davao_tweets.most_common(50))
makati_tweets = nltk.FreqDist(get_all_words(makati_tweets))
print('MAKATI:',makati_tweets.most_common(50))
manila_tweets = nltk.FreqDist(get_all_words(manila_tweets))
print('MANILA:',manila_tweets.most_common(50))
palawan_tweets = nltk.FreqDist(get_all_words(palawan_tweets))
print('PALAWAN:',palawan_tweets.most_common(50))
tagaytay_tweets = nltk.FreqDist(get_all_words(tagaytay_tweets))
print('TAGAYTAY:',tagaytay_tweets.most_common(50))
vigan_tweets = nltk.FreqDist(get_all_words(vigan_tweets))
print('VIGAN:',vigan_tweets.most_common(50))
subic_tweets = nltk.FreqDist(get_all_words(subic_tweets))
print('SUBIC:',subic_tweets.most_common(50))
