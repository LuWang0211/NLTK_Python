# /home/lududu/anaconda3/bin/activate
# lu 2019 copyright
# Stylometric Test: John Burrows’ Delta Method 
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import gutenberg
import math
import pickle

def DeltaMethod(subcorpora_by_author_string, Authors):
  # Simplifiy Samples
  subcorpora_by_author_tokens = {}
  subcorpora_by_author_length_distributions = {}
  for author in Authors:
      tokens = nltk.word_tokenize(subcorpora_by_author_string[author])
      subcorpora_by_author_tokens[author] = ([token for token in tokens if any(c.isalpha() for c in token)])
    
      token_lengths = [len(token) for token in subcorpora_by_author_tokens[author]]
      subcorpora_by_author_length_distributions[author] = nltk.FreqDist(token_lengths)

  whole_corpus = []
  for author in Authors:
      whole_corpus += subcorpora_by_author_tokens[author]

  whole_corpus_freq_dist = list(nltk.FreqDist(whole_corpus).most_common(30))

  # Calculate each feature's presence in the subcorpus, for each candidate's features
  features = [word for word,freq in whole_corpus_freq_dist]

  # Calculate feature freqs
  feature_freqs = {}
  
  for author in Authors:
    feature_freqs[author] = {} 
    
    sum_tokens_author = len(subcorpora_by_author_tokens[author])
    
    for feature in features:
        presence = subcorpora_by_author_tokens[author].count(feature)
        feature_freqs[author][feature] = presence / sum_tokens_author

  # Find a “mean of means” and a standard deviation for each feature
  corpus_features = {}

  # Calculate the mean
  for feature in features:
    corpus_features[feature] = {}

    feature_average = 0
    for author in Authors:
        feature_average += feature_freqs[author][feature]
    feature_average /= len(Authors)
    corpus_features[feature]['Mean'] = feature_average

    # Calculate the standard deviation
    feature_stdev = 0
    for author in Authors:
      diff = feature_freqs[author][feature] - corpus_features[feature]['Mean']
      feature_stdev += diff*diff
      if len(Authors) != 1:
        feature_stdev /= (len(Authors) - 1)
      feature_stdev = math.sqrt(feature_stdev)
      corpus_features[feature]['StdDev'] = feature_stdev

  # Calculating z-scores, its definition = (value - mean) / stddev
  feature_zscores = {}
  for author in Authors:
    feature_zscores[author] = {}
    for feature in features:
      feature_val = feature_freqs[author][feature]
      feature_mean = corpus_features[feature]['Mean']
      feature_stdev = corpus_features[feature]['StdDev']
      feature_zscores[author][feature] = ((feature_val-feature_mean) / feature_stdev)

  return features, feature_zscores, corpus_features

#--------------------------------------------------------------------------------
#----------------Samples----------------
subGutenberg = {
  'Bible': ['bible-kjv.txt'],
  'Austen': ['austen-emma.txt', 'austen-persuasion.txt', 'austen-sense.txt'],
  'Chesterton': ['chesterton-ball.txt', 'chesterton-brown.txt', 'chesterton-thursday.txt'],
}

Authors = ['Bible', 'Austen', 'Chesterton']

def read_files_into_string(filenames):
  strings = []
  for filename in filenames:
    with open(f'gutenberg/{filename}', encoding = "ISO-8859-1") as f:
      strings.append(f.read())
  return ''.join(strings)

subcorpora_by_author = {}  
for author, files in subGutenberg.items():
  subcorpora_by_author[author] = read_files_into_string(files)

subGutenberg_feature_zscores = DeltaMethod(subcorpora_by_author, Authors)


# --------------Pickle module --------------------
save_classifier = open("DeltaMethod.pickle","wb")
pickle.dump(subGutenberg_feature_zscores, save_classifier)
save_classifier.close()  # close the file