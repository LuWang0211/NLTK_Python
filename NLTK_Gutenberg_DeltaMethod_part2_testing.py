# /home/lududu/anaconda3/bin/activate
# lu 2019 copyright
# Stylometric Test: John Burrows’ Delta Method 
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import gutenberg
import math
import NLTK_Gutenberg_DeltaMethod_part1_training
import pickle
from NLTK_Gutenberg_DeltaMethod_part1_training import DeltaMethod, read_files_into_string

#--------------------------------------------------------------------------------
#----------------Samples----------------
TestCase = {
  'UnknownAuthors':['austen-persuasion.txt'] #TestCase
}

Authors = ['Bible', 'Austen', 'Chesterton']

classifier_f = open("DeltaMethod.pickle", "rb") 
feature_zscores = pickle.load(classifier_f) 

TestCase_by_author = {}  
for author, files in TestCase.items():
  TestCase_by_author[author] = read_files_into_string(files)

testcase_tokens = nltk.word_tokenize(TestCase_by_author['UnknownAuthors'])
    
# Filter out punctuation and lowercase the tokens
testcase_tokens = [token.lower() for token in testcase_tokens if any(c.isalpha() for c in token)]

# Calculate the test case's features
overall = len(testcase_tokens)
testcase_freqs = {}
for feature in feature_zscores[0]:
    presence = testcase_tokens.count(feature)
    testcase_freqs[feature] = presence / overall

corpus_features = feature_zscores[2]

# Calculate the test case's feature z-scores
testcase_zscores = {}
for feature in feature_zscores[0]:
    feature_val = testcase_freqs[feature]
    feature_mean = corpus_features[feature]['Mean']
    feature_stdev = corpus_features[feature]['StdDev']
    testcase_zscores[feature] = (feature_val - feature_mean) / feature_stdev
# print("Test case z-score for feature", feature, "is", testcase_zscores[feature])

#----------------Delta Method----------------
# Calculating Delta-  the formula for Delta defined by Burrows
author_by_delta = {}
for author in Authors:
    delta = 0
    for feature in feature_zscores[0]:
        delta += math.fabs((testcase_zscores[feature] - 
                            feature_zscores[1][author][feature]))
    delta /= len(feature_zscores[0])
    print( "Delta score for candidate", author, "is", delta )
    author_by_delta[delta] = author
# print(author_by_delta)

min_delta = min(author_by_delta, key=author_by_delta.get)
most_likely_author = author_by_delta[min_delta]

print(f'Delta identifition Result: "{most_likely_author}" as TestCase most likely author.')
