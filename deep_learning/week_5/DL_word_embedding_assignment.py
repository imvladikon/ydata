#!/usr/bin/env python
# coding: utf-8

# # Word Embedding - Home Assigment
# ## Dr. Omri Allouche 2021. YData Deep Learning Course
# 
# [Open in Google Colab](https://colab.research.google.com/github/omriallouche/ydata_deep_learning_2021/blob/master/assignments/word_vectors_text_classification/DL_word_embedding_assignment.ipynb)
# 
#     
# In this exercise, you'll use word vectors trained on a corpus of lyrics of songs from MetroLyrics ([http://github.com/omriallouche/ydata_deep_learning_2021/blob/master/data/metrolyrics.parquet](/data/metrolyrics.parquet)).
# The dataset contains these fields for each song, in CSV format:
# 1. index
# 1. song
# 1. year
# 1. artist
# 1. genre
# 1. lyrics
# 
# Before doing this exercise, we recommend that you go over the "Bag of words meets bag of popcorn" tutorial (https://www.kaggle.com/c/word2vec-nlp-tutorial)
# 
# Other recommended resources:
# - https://rare-technologies.com/word2vec-tutorial/
# - https://www.kaggle.com/pierremegret/gensim-word2vec-tutorial

# ### Train word vectors
# Train word vectors using the Skipgram Word2vec algorithm and the gensim package.
# Make sure you perform the following:
# - Tokenize words
# - Lowercase all words
# - Remove punctuation marks
# - Remove rare words
# - Remove stopwords
# 
# Use 300 as the dimension of the word vectors. Try different context sizes.

# In[ ]:





# ### Review most similar words
# Get initial evaluation of the word vectors by analyzing the most similar words for a few interesting words in the text. 
# 
# Choose words yourself, and find the most similar words to them.

# In[ ]:





# ### Word Vectors Algebra
# We've seen in class examples of algebraic games on the word vectors (e.g. man - woman + king = queen ). 
# 
# Try a few vector algebra terms, and evaluate how well they work. Try to use the Cosine distance and compare it to the Euclidean distance.

# In[ ]:





# ## Sentiment Analysis
# Estimate sentiment of words using word vectors.  
# In this section, we'll use the SemEval-2015 English Twitter Sentiment Lexicon.  
# The lexicon was used as an official test set in the SemEval-2015 shared Task #10: Subtask E, and contains a polarity score for words in range -1 (negative) to 1 (positive) - http://saifmohammad.com/WebPages/SCL.html#OPP

# Build a classifier for the sentiment of a word given its word vector. Split the data to a train and test sets, and report the model performance on both sets.

# In[ ]:





# Use your trained model from the previous question to predict the sentiment score of words in the lyrics corpus that are not part of the original sentiment dataset. Review the words with the highest positive and negative sentiment. Do the results make sense?

# In[ ]:





# ### Visualize Word Vectors
# In this section, you'll plot words on a 2D grid based on their inner similarity. We'll use the tSNE transformation to reduce dimensions from 300 to 2. You can get sample code from https://www.kaggle.com/pierremegret/gensim-word2vec-tutorial or other tutorials online.
# 
# Perform the following:
# - Keep only the 3,000 most frequent words (after removing stopwords)
# - For this list, compute for each word its relative abundance in each of the genres
# - Compute the ratio between the proportion of each word in each genre and the proportion of the word in the entire corpus (the background distribution)
# - Pick the top 50 words for each genre. These words give good indication for that genre. Join the words from all genres into a single list of top significant words. 
# - Compute tSNE transformation to 2D for all words, based on their word vectors
# - Plot the list of the top significant words in 2D. Next to each word output its text. The color of each point should indicate the genre for which it is most significant.
# 
# You might prefer to use a different number of points or a slightly different methodology for improved results.  
# Analyze the results.

# In[ ]:





# ## Text Classification
# In this section, you'll build a text classifier, determining the genre of a song based on its lyrics.

# ### Text classification using Bag-of-Words
# Build a Naive Bayes classifier based on the bag of Words.  
# You will need to divide your dataset into a train and test sets.

# In[ ]:





# Show the confusion matrix.

# In[ ]:





# Show the classification report - precision, recall, f1 for each class.

# In[ ]:





# ### Text classification using Word Vectors
# #### Average word vectors
# Do the same, using a classifier that averages the word vectors of words in the document.

# In[ ]:





# #### TfIdf Weighting
# Do the same, using a classifier that averages the word vectors of words in the document, weighting each word by its TfIdf.
# 

# In[ ]:





# ### Text classification using ConvNet
# Do the same, using a ConvNet.  
# The ConvNet should get as input a 2D matrix where each column is an embedding vector of a single word, and words are in order. Use zero padding so that all matrices have a similar length.  
# Some songs might be very long. Trim them so you keep a maximum of 128 words (after cleaning stop words and rare words).  
# Initialize the embedding layer using the word vectors that you've trained before, but allow them to change during training.  
# 
# Extra: Try training the ConvNet with 2 slight modifications:
# 1. freezing the the weights trained using Word2vec (preventing it from updating)
# 1. random initialization of the embedding layer

# You are encouraged to try this question on your own.  
# 
# You might prefer to get ideas from the paper "Convolutional Neural Networks for Sentence Classification" (Kim 2014, [link](https://arxiv.org/abs/1408.5882)).
# 
# There are several implementations of the paper code in PyTorch online (see for example [this repo](https://github.com/prakashpandey9/Text-Classification-Pytorch) for a PyTorch implementation of CNN and other architectures for text classification). If you get stuck, they might provide you with a reference for your own code.

# In[ ]:




