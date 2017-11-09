# Quora_Question_Pairs_another_approach_Capstone

This repository describes an approach to the Quora Question pairs ML challenge, that was hosted in kaggle.

The solution uses Python 2.7, but can also be used with python 3.x

Python packages Required to run the Notebooks: 
* Numpy
* Pandas
* Scikit-learn
* Matplotlib
* Keras
* Tensorflow
* tqdm
* re
* pickle
* gc

## Required Files
  Duplicate Questions Dataset: http://qim.ec.quoracdn.net/quora_duplicate_questions.tsv
  
  GloVe Embeddings: http://nlp.stanford.edu/data/glove.840B.300d.zip


## Steps

* Run Preprocess.py on the root folder containing the dataset and the embedding, to create the **question_pair.pickle** file and **embeddings.txt** file
* Final Model notebook can be run now to replicate the results.
* Base Models Notebook is a standalone Notebook and doesnot require running preprocess.py first. Though it is to be noted that if preprocessing.py has already been run before then only import of **question_pair.pickle** file and **embeddings.txt** file is required in this notebook.
