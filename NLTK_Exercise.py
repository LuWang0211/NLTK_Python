# Python3
# Copyright 2019 Lulu Wang
# The Gutenberg corpus
## 
# import nltk
# nltk.download()
# print(nltk.__file__)

from nltk.tokenize import sent_tokenize, word_tokenize, PunktSentenceTokenizer
from nltk.corpus import stopwords, gutenberg, wordnet
# sample text
# EXAMPLE_TEXT_1 = "Hello Mr. Smith, how are you doing today? The weather is great, and Python is awesome. The sky is pinkish-blue! You shouldn't eat cardboard."

# print(sent_tokenize(EXAMPLE_TEXT_1))

# print(word_tokenize(EXAMPLE_TEXT_1))
# for i in word_tokenize(EXAMPLE_TEXT_1):
#     print(i)


# sample text
# example_sent = "This is a sample sentence, showing off the stop words filtration."

# stop_words = set(stopwords.words('english'))
# print(stop_words)

# word_tokens = word_tokenize(example_sent)
# filtered_sentence = [w for w in word_tokens if not w in stop_words]

# print(word_tokens)
# print(filtered_sentence)

# sample text
# sample = gutenberg.raw("bible-kjv.txt")

# tok = sent_tokenize(sample)

# for x in range(5):
#     print(tok[x])


from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

# print(lemmatizer.lemmatize("cats"))
# print(lemmatizer.lemmatize("cacti"))
# print(lemmatizer.lemmatize("geese"))
# print(lemmatizer.lemmatize("rocks"))
# print(lemmatizer.lemmatize("python"))
# print(lemmatizer.lemmatize("better", pos="a"))
# print(lemmatizer.lemmatize("best", pos="a"))
# print(lemmatizer.lemmatize("run"))
# print(lemmatizer.lemmatize("run",'v'))


# worldnet sample text
# syns = wordnet.synsets("program")
syns = wordnet.synsets("good")
# print(syns)
# synset
# print(syns)
# word
# print(syns[0].name())
# lemmas
# print(syns[0].lemmas()[0].name())
# definition
# print(syns[0].definition())
# print(syns[0].examples())

# synonyms = []
# antonyms = []

# for syn in wordnet.synsets("good"):
#     for l in syn.lemmas():
#         synonyms.append(l.name())
#         if l.antonyms():
#             antonyms.append(l.antonyms()[0].name())

# print(set(synonyms))
# print(set(antonyms))

#  compare the noun of "ship" and "boat:"
# w1 = wordnet.synset('ship.n.01')
# w2 = wordnet.synset('boat.n.01')
# print(w1.wup_similarity(w2))
#  compare the noun of "ship" and "car:"
# w1 = wordnet.synset('ship.n.01')
# w2 = wordnet.synset('car.n.01')
# print(w1.wup_similarity(w2))

# --------------------------------------------------------
from nltk.corpus import stopwords, gutenberg, wordnet
# print(text1)
# text1.concordance("monstrous")
# print(len(set(text5)) / len(text5))
# print(text5.count('lol'))
# print(100*text5.count('lol')/len(text5))

def lexical_diversity(text):
    return len(set(text)) / len(text)

def percentage(word, text): 
    str_word = str(word)
    count = text.count(str_word)
    total = len(text)
    return 100 * count / total

# print(lexical_diversity(text5))
# print(percentage('lol', text5))

# for fileid in gutenberg.fileids():
#     # title = gutenberg.raw(fileid)
#     title = nltk.Text(gutenberg.words(fileid))
#     num_chars = len(gutenberg.raw(fileid))
#     num_words = len(gutenberg.words(fileid))
#     num_sents = len(gutenberg.sents(fileid))
#     num_vocab = len(set([w.lower() for w in gutenberg.words(fileid)]))
#     # print(fileid, title, num_chars/num_words, num_chars/num_sents,num_words/num_vocab)
    
#     print(title, 'num_chars:', num_chars, 'num_words: ', num_words, 'num_sents: ', num_sents, 'num_vocab: ', num_vocab)




#-------------------------------------------------------------
import nltk
import random
from nltk.corpus import movie_reviews
import pickle

#  importing the data set: 
#  In each category (pos or neg), take all of the file IDs (each review has its own ID)
#  Then store the word_tokenized version (a list of words) for the file ID, followed by the positive or negative label in one big list.
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# shuffle documents in order to train and test correctly
# If we left them in order, chances are we'd train on all of the negatives, some positives, and then test only against positives.
random.shuffle(documents)

# print(documents[1])

all_words = []

for w in movie_reviews.words():
    all_words.append(w.lower())

# frequency distribution- can find the most common words
all_words = nltk.FreqDist(all_words)
# print(all_words.most_common(15))
# print(all_words["stupid"])

word_features = list(all_words.keys())[:3000]  # contains the top 3,000 most common words

# --------------word_features --------------------
# funcation: 
# find word_features(these top 3,000 words) in our positive and negative documents(documents)
def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

# saving the feature existence booleans and their respective positive or negative categories
featuresets = [(find_features(rev), category) for (rev, category) in documents]

# --------------Train and Test --------------------
# set that we'll train our classifier with
training_set = featuresets[:1900]

# set that we'll test against.
testing_set = featuresets[1900:]

# train classifier
classifier = nltk.NaiveBayesClassifier.train(training_set)

# then test
print("Classifier accuracy percent:",(nltk.classify.accuracy(classifier, testing_set))*100)

# the most valuable words- when it comes to positive or negative reviews
classifier.show_most_informative_features(15)

# --------------Pickle module --------------------
# when classifier too big
# Pickle module to go ahead and serialize our classifier object
# import pickle at the top of your script, then, after you have trained with .train() the classifier
# step 1 : save the object
save_classifier = open("naivebayes.pickle","wb")  # opens up a pickle file, preparing to write in bytes some data
pickle.dump(classifier, save_classifier)  #  pickle.dump() to dump the data
# The first parameter-"classifier" to pickle.dump() is what are you dumping, the second parameter is where are you dumping it
save_classifier.close()  # close the file

# step 2 :  read it into memory
classifier_f = open("naivebayes.pickle", "rb") # open the file to read as bytes. 
classifier = pickle.load(classifier_f) # pickle.load() to load the file, and we save the data to the classifier variable.
classifier_f.close() #  close the file. We now have the same classifier object as before!
# we no longer need to train our classifier every time we wanted to use it to classify
