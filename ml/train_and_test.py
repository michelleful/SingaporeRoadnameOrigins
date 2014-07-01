"""
Train and test various classifiers on the data
"""
from numpy import array
from sklearn.feature_extraction.text import CountVectorizer

DATA_FOLDER = '../data/'
TRAIN_FROM = 0 # don't change unless necessary
TRAIN_TO   = 0 # CHANGE THIS
TEST_FROM  = TRAIN_TO + 1 # don't change unless necessary
TEST_TO    = 9 # by default test all, change if necessary

cv = CountVectorizer(ngram_range=(1,4), analyzer='char')
def extract_ngrams(roadname_list, type='test'):
    if type == 'train':
        return cv.fit_transform(roadname_list).toarray()
    else: # type == 'test'
        return cv.transform(roadname_list).toarray()

def extract_training_data():
    """Read training data in from files, generate X matrix and y vector
    where X is a matrix of feature vectors and y is the (supervised)
    set of results
    """
    # global values so we can build up a single X and y matrix/feature vector
    roadname_list = list()
    X_other_features = list()    
    y = list()
    
    # go through each file and extract tab-separated data
    for i in range(TRAIN_FROM, TRAIN_TO + 1):
        filename = DATA_FOLDER + 'data.%s.gold.csv' % i
        with open(filename, 'r') as f:
            for line in f:
                data = [item.strip() for item in line.split("\t")]
                [roadname, malay_road_tag, average_word_length,
                 all_words_in_dictionary, classification] = data
                roadname_list.append(roadname.strip())

                # glue together the rest of the data
                X_other_features.append([malay_road_tag, average_word_length,
                                         all_words_in_dictionary])

                # build up the gold standard vector
                y.append(classification)

    # finally, get the output of the n-gram vectorizer
    # and "glue" it to the other features
    X_ngram_features = extract_ngrams(roadname_list, type='train')
    
def extract_testing_data():
    """Read testing data in from files, generate X matrix of feature vectors
    """
    # global values so we can build up a single X matrix
    roadname_list = list()
    X_other_features = list()    
    
    # go through each file and extract tab-separated data
    for i in range(TEST_FROM, TEST_TO + 1):
        filename = DATA_FOLDER + 'data.%s.conv.csv' % i
        with open(filename, 'r') as f:
            for line in f:
                data = [item.strip() for item in line.split("\t")]
                [roadname, malay_road_tag, average_word_length,
                 all_words_in_dictionary] = data
                roadname_list.append(roadname.strip())

                # glue together the rest of the data
                X_other_features.append([malay_road_tag, average_word_length,
                                         all_words_in_dictionary])

    # finally, get the output of the n-gram vectorizer
    # and "glue" it to the other features
    X_ngram_features = extract_ngrams(roadname_list, type='test')
    
        
extract_training_data()
extract_testing_data()