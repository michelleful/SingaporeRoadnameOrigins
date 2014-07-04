"""
Train and test various classifiers on the data
"""
from numpy import array, hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import cross_validation, grid_search
from sklearn import svm, naive_bayes, neighbors, ensemble  # classifiers

DATA_FOLDER = '../data/'
TRAIN_FROM = 0  # don't change unless necessary
TRAIN_TO   = 0  # CHANGE THIS
TEST_FROM  = TRAIN_TO + 1  # don't change unless necessary
TEST_TO    = 9  # by default test all, change if necessary

# ---------------------
#  FEATURE EXTRACTION
# ---------------------

NGRAM = CountVectorizer(ngram_range=(1, 4), analyzer='char')


def extract_ngrams(roadname_list, type='test'):
    if type == 'train':
        return NGRAM.fit_transform(roadname_list).toarray()
    else:  # type == 'test'
        return NGRAM.transform(roadname_list).toarray()


# ---------------------------
#  DATA PROCESSING FUNCTIONS
# ---------------------------

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
                X_other_features.append([int(malay_road_tag),
                                         float(average_word_length),
                                         int(all_words_in_dictionary)])

                # build up the gold standard vector
                y.append(int(classification))

    # finally, get the output of the n-gram vectorizer
    # and "glue" it to the other features
    X_ngram_features = extract_ngrams(roadname_list, type='train')

    X = hstack((X_ngram_features, array(X_other_features)))

    return roadname_list, X, y


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
                X_other_features.append([int(malay_road_tag),
                                         float(average_word_length),
                                         int(all_words_in_dictionary)])

    # finally, get the output of the n-gram vectorizer
    # and "glue" it to the other features
    X_ngram_features = extract_ngrams(roadname_list, type='test')
    X = hstack((X_ngram_features, array(X_other_features)))

    return roadname_list, X

# ------------------
#    CLASSIFIERS
# ------------------

# note: why multiple skeletons? Because we'll need to tune different
#       sets of parameters for each one (presumably)

# how to find parameters: classifier.get_params

# ------------------
#    Naive Bayes
# ------------------

def trained_multinomial_naive_bayes(X_train, y_train, X_dev, y_dev):
    """Trains a multinomial Naive Bayes classifier using X_train, y_train
       and tunes parameters based on X_dev, y_dev.
       Returns the classifier
    """
    classifier = naive_bayes.MultinomialNB()
    
    # parameters:
    # {'alpha': 1.0, 'fit_prior': True, 'class_prior': None}
    
    return classifier


def trained_bernoulli_naive_bayes(X_train, y_train, X_dev, y_dev):
    """Trains a Bernoulli Naive Bayes classifier using X_train, y_train
       and tunes parameters based on X_dev, y_dev.
       Returns the classifier
    """
    classifier = naive_bayes.BernoulliNB()

    # parameters:
    # {'binarize': 0.0, 'alpha': 1.0, 'fit_prior': True, 'class_prior': None}

    return classifier

# ------------------
#    Linear SVC
# ------------------

def trained_linear_svc(X_train, y_train, X_dev, y_dev):
    """Trains a Linear SVC classifier using X_train, y_train
       and tunes parameters based on X_dev, y_dev.
       Returns the classifier
    """
    # svm.LinearSVC
    pass

# ------------------
#        SVC
# ------------------

def trained_svc(X_train, y_train, X_dev, y_dev):
    """Trains a SVC classifier using X_train, y_train
       and tunes parameters based on X_dev, y_dev.
       Returns the classifier
    """
    # svm.SVC
    pass


# ------------------
#        KNN
# ------------------

def trained_knn(X_train, y_train, X_dev, y_dev):
    """Trains a k-Nearest Neighbours classifier using X_train, y_train
       and tunes parameters based on X_dev, y_dev.
       Returns the classifier
    """
    # neighbors.KNeighborsClassifier
    pass


# --------------------
#   Ensemble methods
# --------------------

# Ensemble methods: Extra Trees Classifier,
# Ada Boost Classifier, Gradient Boosting Classifier

def trained_random_forest(X_train, y_train, X_dev, y_dev):
    """Trains a Random Forest classifier using X_train, y_train
       and tunes parameters based on X_dev, y_dev.
       Returns the classifier
    """
    # ensemble.RandomForestClassifier
    pass


def trained_ada_boost(X_train, y_train, X_dev, y_dev):
    """Trains an Ada Boost classifier using X_train, y_train
       and tunes parameters based on X_dev, y_dev.
       Returns the classifier
    """
    # ensemble.AdaBoostClassifier
    pass


def trained_gradient_boost(X_train, y_train, X_dev, y_dev):
    """Trains a Gradient Boost classifier using X_train, y_train
       and tunes parameters based on X_dev, y_dev.
       Returns the classifier
    """
    # ensemble.GradientBoostClassifier
    pass


# --------------------
#       MAIN
# --------------------

# get the training data plus the data to be classified
train_roadnames, train_X, train_y = extract_training_data()
test_roadnames,  test_X           = extract_testing_data()

# further split the training data into a training set and a development set
X_train, y_train, X_dev, y_dev = cross_validation.train_test_split(train_X, 
                                    train_y, test_size=0.2, random_state=42)






# what about ranges for parameter search? 
# what ranges should be defined for each?
# see: GridSearchCV http://scikit-learn.org/stable/modules/grid_search.html


# what scoring function to use?
# http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
# there is also a random search function but you still need to define the
# parameter distributions.



#clf = svm.SVC(gamma=0.001, C=100.)

#clf.fit(train_X, train_y)
#print clf.predict(test_X)

#from sklearn import cross_validation
#X_train, X_test, y_train, y_test = cross_validation.train_test_split(train_X,
#                                        train_y, test_size=0.2, random_state=0)

#clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
#print clf.score(X_test, y_test)
