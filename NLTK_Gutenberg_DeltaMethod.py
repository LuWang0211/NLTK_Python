# /home/lududu/anaconda3/bin/activate
# lu 2019 copyright
# Stylometric Test: John Burrows’ Delta Method 
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import gutenberg
import math

#--------------------------------------------------------------------------------
#----------------Samples----------------
subGutenberg = {
  'Bible': ['bible-kjv.txt'],
  'Austen': ['austen-emma.txt', 'austen-persuasion.txt', 'austen-sense.txt'],
  'Chesterton': ['chesterton-ball.txt', 'chesterton-brown.txt', 'chesterton-thursday.txt'],
  'TestCase':['austen-persuasion.txt'] #TestCase
}

Authors = ['Bible', 'Austen', 'Chesterton']

# Create unit-texts by authors
def read_files_into_string(filenames):
  strings = []
  for filename in filenames:
    with open(f'gutenberg/{filename}', encoding = "ISO-8859-1") as f:
      strings.append(f.read())
  return ''.join(strings)

subcorpora_by_author = {}  
# i = 1
for author, files in subGutenberg.items():
  # print(f'before read files{author,files}', f'loop NO.{i} start' )
  subcorpora_by_author[author] = read_files_into_string(files)
  # print(f'after read files{author,files}', f'loop NO.{i} end' )
  # i += 1

# print(subcorpora_by_author['Bible'][:300])

# Simplifiy Samples
subcorpora_by_author_tokens = {}
subcorpora_by_author_length_distributions = {}
for author in Authors:
    tokens = nltk.word_tokenize(subcorpora_by_author[author])
    subcorpora_by_author_tokens[author] = ([token for token in tokens if any(c.isalpha() for c in token)])
   
    token_lengths = [len(token) for token in subcorpora_by_author_tokens[author]]
    subcorpora_by_author_length_distributions[author] = nltk.FreqDist(token_lengths)


whole_corpus = []
for author in Authors:
    whole_corpus += subcorpora_by_author_tokens[author]

whole_corpus_freq_dist = list(nltk.FreqDist(whole_corpus).most_common(30))
# print(whole_corpus_freq_dist[ :10 ])

# Calculate each feature's presence in the subcorpus, for each candidate's features
features = [word for word,freq in whole_corpus_freq_dist]
feature_freqs = {}

for author in Authors:
  feature_freqs[author] = {} 
  
  sum_tokens_author = len(subcorpora_by_author_tokens[author])
  
  for feature in features:
      presence = subcorpora_by_author_tokens[author].count(feature)
      feature_freqs[author][feature] = presence / sum_tokens_author
# print(feature_freqs)

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
    feature_stdev /= (len(Authors) - 1)
    feature_stdev = math.sqrt(feature_stdev)
    corpus_features[feature]['StdDev'] = feature_stdev
# print(corpus_features)

# Calculating z-scores, its definition = (value - mean) / stddev
feature_zscores = {}
for author in Authors:
  feature_zscores[author] = {}
  for feature in features:
    feature_val = feature_freqs[author][feature]
    feature_mean = corpus_features[feature]['Mean']
    feature_stdev = corpus_features[feature]['StdDev']
    feature_zscores[author][feature] = ((feature_val-feature_mean) / feature_stdev)
# print(feature_zscores)

#----------------Test files----------------
# Calculating features and z-scores for test
# Tokenize the test case
testcase_tokens = nltk.word_tokenize(subcorpora_by_author['TestCase'])
    
# Filter out punctuation and lowercase the tokens
testcase_tokens = [token.lower() for token in testcase_tokens if any(c.isalpha() for c in token)]

# Calculate the test case's features
overall = len(testcase_tokens)
testcase_freqs = {}
for feature in features:
    presence = testcase_tokens.count(feature)
    testcase_freqs[feature] = presence / overall
    
# Calculate the test case's feature z-scores
testcase_zscores = {}
for feature in features:
    feature_val = testcase_freqs[feature]
    feature_mean = corpus_features[feature]['Mean']
    feature_stdev = corpus_features[feature]['StdDev']
    testcase_zscores[feature] = (feature_val - feature_mean) / feature_stdev
print("Test case z-score for feature", feature, "is", testcase_zscores[feature])

#----------------Delta Method----------------
# Calculating Delta-  the formula for Delta defined by Burrows
author_by_delta = {}
for author in Authors:
    delta = 0
    for feature in features:
        delta += math.fabs((testcase_zscores[feature] - 
                            feature_zscores[author][feature]))
    delta /= len(features)
    print( "Delta score for candidate", author, "is", delta )
    author_by_delta[delta] = author
# print(author_by_delta)

min_delta = min(author_by_delta, key=author_by_delta.get)
most_likely_author = author_by_delta[min_delta]

print(f'Delta identifition Result: "{most_likely_author}" as TestCase most likely author.')
