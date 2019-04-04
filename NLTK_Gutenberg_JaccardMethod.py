# /home/lududu/anaconda3/bin/activate
# lu 2019 copyright
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import gutenberg, stopwords, wordnet
import os
# import pickle

#---------------- Tokenization, Cleaning and Frequency Distribution ----------------
# # Gutenberg Samples
def Clean_Freq_Gfile(files):

  word_tokens = word_tokenize(gutenberg.raw(files[i] for i in range(len(files))))
  stop_words = set(stopwords.words('english'))  # delete the most common words in English
  filtered_sentence = [w.lower() for w in word_tokens if not w in stop_words] 
  filtered_sentence = [w for w in filtered_sentence if w.isalpha()] # delete punctuation
  filtered_sentence = nltk.FreqDist(filtered_sentence) # frequency distribution

  return filtered_sentence

# # test files - not in Gutenberg path files
def Clean_Freq_Testfile(files):

  test_f = open(files, "r")
  word_tokens = word_tokenize(test_f.read())
  stop_words = set(stopwords.words('english'))
  filtered_sentence = [w.lower() for w in word_tokens if not w in stop_words]
  filtered_sentence = [w for w in filtered_sentence if w.isalpha()]
  filtered_sentence = nltk.FreqDist(filtered_sentence)

  return filtered_sentence

# # test text, not files
def Clean_Freq_text(texts):
  
  word_tokens = word_tokenize(texts)
  stop_words = set(stopwords.words('english'))
  filtered_sentence = [w.lower() for w in word_tokens if not w in stop_words]
  filtered_sentence = [w for w in filtered_sentence if w.isalpha()]
  filtered_sentence = nltk.FreqDist(filtered_sentence)

  return filtered_sentence

#---------------- Similarity method ----------------
# Jaccard similarity method
def jaccard_similarity(x,y):
 
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    return intersection_cardinality/float(union_cardinality)

#--------------------------------------------------------------------------------
#----------------Samples----------------
Authors = ['Bible', 'Austen', 'Chesterton']
Bible_Example_fileids = ['bible-kjv.txt']
Austen_Example_fileids = ['austen-emma.txt', 'austen-persuasion.txt', 'austen-sense.txt']
Chesterton_Example_fileids = ['chesterton-ball.txt', 'chesterton-brown.txt', 'chesterton-thursday.txt']

# Simplifiy Samples
# 1. Use Clean_Freq_Gfile() Simplify Samples
Bible_example_words = Clean_Freq_Gfile([Bible_Example_fileids[i] for i in range(len(Bible_Example_fileids))])
Austen_example_words = Clean_Freq_Gfile([Austen_Example_fileids[i] for i in range(len(Austen_Example_fileids))])
Chesterton_example_words = Clean_Freq_Gfile([Chesterton_Example_fileids[i] for i in range(len(Chesterton_Example_fileids))])

# 2. Further Simlify- Delete common High frequency in three authors samples
# 2.1: Find common words in three authors samples
comom_dict = dict()
for keys in Bible_example_words.keys():
  if keys in Austen_example_words.keys() and keys in Chesterton_example_words.keys():
    comom_dict[keys] = Bible_example_words[keys]

# 2.2: delete common words
Bible_example_unique_words = dict()
Austen_example_unique_words = dict()
Chesterton_example_unique_words = dict()

for keys in Bible_example_words.keys():
  if keys not in comom_dict:
    Bible_example_unique_words[keys] = Bible_example_words[keys]

for keys in Austen_example_words.keys():
  if keys not in comom_dict:
    Austen_example_unique_words[keys] = Austen_example_words[keys]

for keys in Chesterton_example_words.keys():
  if keys not in comom_dict:
    Chesterton_example_unique_words[keys] = Chesterton_example_words[keys]

# Creat Simplified Samples
Bible_example_unique_words = nltk.FreqDist(Bible_example_unique_words)
Austen_example_unique_words = nltk.FreqDist(Austen_example_unique_words)
Chesterton_example_unique_words = nltk.FreqDist(Chesterton_example_unique_words)
# print(Bible_example_unique_words.most_common(50))
# print(Austen_example_unique_words.most_common(50))
# print(Chesterton_example_unique_words.most_common(50))

#----------------Test files----------------
## Input: accept input from text or file
# Simplifiy Test files
my_input = input("Enter a txt file name or a text content: ")
if '.txt' in my_input: 
  # check result- find title of files, which includes author's name
  with open(f'gutenberg/{my_input}') as f:
    title = nltk.word_tokenize((f.read()))
  # print(title[:20])
  my_input = f'gutenberg/{my_input}'
  test_files_words = Clean_Freq_Testfile(my_input)
else:
  title = word_tokenize(my_input)
  test_files_words = Clean_Freq_text(my_input)

# delete common words in Test files
test_files_unique_words = dict()
for keys in test_files_words.keys():
  if keys not in comom_dict:
    test_files_unique_words[keys] = test_files_words[keys]

# Creat Simplified Test files
test_files_unique_words = nltk.FreqDist(test_files_unique_words)
# print(test_files_unique_words.most_common(50))


#----------------Compare files, Find similarity----------------
# Use Jaccard similarity method 
Similarity_Bible_flag = jaccard_similarity(Bible_example_unique_words.keys(),test_files_unique_words.keys())
Similarity_Austen_flag = jaccard_similarity(Austen_example_unique_words.keys(),test_files_unique_words.keys())
Similarity_Chesterton_flag = jaccard_similarity(Chesterton_example_unique_words.keys(),test_files_unique_words.keys())
Similarity_max = max(Similarity_Bible_flag, Similarity_Austen_flag, Similarity_Chesterton_flag)

# print(Similarity_Bible_flag, Similarity_Austen_flag, Similarity_Chesterton_flag)


# ----------------Prediction Result----------------
if Similarity_max < 0.20: # Depreciation 0.20 ?
  Predictive_author = 'None of them'
  print('The author may be not Bible, Austen, or Chesterton')
  
  for i in Authors:
    if i in title[:20]:
      print('The prediction result is worry.')
      break
  print('The prediction result is correct.')
else:
  if Similarity_max == Similarity_Bible_flag:
    Predictive_author = Authors[0]
  elif Similarity_max == Similarity_Austen_flag:
    Predictive_author = Authors[1]
  elif Similarity_max == Similarity_Chesterton_flag:
    Predictive_author = Authors[2]
  print('The author may be:', Predictive_author)

  if Predictive_author in title[:20]:
    print('The prediction result is correct.')
  else:
    print('The prediction result is worry.')
