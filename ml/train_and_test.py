"""
Train and test various classifiers on the data
"""

from glob import glob
from sklearn.feature_extraction.text import CountVectorizer

DATA_FOLDER = '../data/'

cv = CountVectorizer(ngram_range=(1,4), analyzer='char')
def extract_ngrams(roadname_list):
    return cv.fit_transform(roadname_list)

def extract_training_data():
    """Read training data in from files, generate X matrix and y vector
    where X is a matrix of feature vectors and y is the (supervised)
    set of results
    """
    # first get list of files
    training_files = glob(DATA_FOLDER + 'data.*.gold.csv')

    # list of roadnames - need a global list to deploy CountVectorizer on
    roadname_list = list()
    # also build up an array
    
    # go through each file and extract tab-separated data
    for filename in training_files:
        with open(filename, 'r') as f:
            for line in f:
                (roadname, malay_road_tag, average_word_length,
                 all_words_in_dictionary, classification) = line.split("\t")
                roadname_list.append(roadname.strip())

    return extract_ngrams(roadname_list)
        
extract_training_data()
