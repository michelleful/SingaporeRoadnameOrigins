"""
Train and test various classifiers on the data
"""
from numpy import array, hstack
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import cross_validation, grid_search
from sklearn import svm, naive_bayes, neighbors, ensemble  # classifiers

DATA_FOLDER = '../data/'
TRAIN_FROM = 0  # don't change unless necessary
TRAIN_TO   = 3  # CHANGE THIS
TEST_FROM  = TRAIN_TO + 1  # don't change unless necessary
TEST_TO    = TEST_FROM + 1 # by default test next one only, change if necessary

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
                 all_words_in_dictionary, classification] = data[:5]
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

# how to find parameters: classifier.get_params()

# what about ranges for parameter search? 
# what ranges should be defined for each?
# see: GridSearchCV http://scikit-learn.org/stable/modules/grid_search.html


# what scoring function to use?
# http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
# there is also a random search function but you still need to define the
# parameter distributions.

# ------------------
#    Naive Bayes
# ------------------

def trained_multinomial_naive_bayes(X_train, y_train, X_dev, y_dev, tune=False):
    """Trains a multinomial Naive Bayes classifier using X_train, y_train
       and tunes parameters based on X_dev, y_dev.
       Returns the classifier
    """
    classifier = naive_bayes.MultinomialNB()

    classifier.fit(X_train, y_train)
    
    # parameters:
    # {'alpha': 1.0, 'fit_prior': True, 'class_prior': None}

    return classifier


def trained_bernoulli_naive_bayes(X_train, y_train, X_dev, y_dev, tune=False):
    """Trains a Bernoulli Naive Bayes classifier using X_train, y_train
       and tunes parameters based on X_dev, y_dev.
       Returns the classifier
    """
    classifier = naive_bayes.BernoulliNB()

    classifier.fit(X_train, y_train)

    # parameters:
    # {'binarize': 0.0, 'alpha': 1.0, 'fit_prior': True, 'class_prior': None}

    return classifier

# ------------------
#    Linear SVC
# ------------------

def trained_linear_svc(X_train, y_train, X_dev, y_dev, tune=False):
    """Trains a Linear SVC classifier using X_train, y_train
       and tunes parameters based on X_dev, y_dev.
       Returns the classifier
    """
    classifier = svm.LinearSVC()

    classifier.fit(X_train, y_train)
    
    # parameters:
    # {'loss': 'l2', 'C': 1.0, 'verbose': 0, 'intercept_scaling': 1, 
    #  'fit_intercept': True, 'penalty': 'l2', 'multi_class': 'ovr', 
    #  'random_state': None, 'dual': True, 'tol': 0.0001, 'class_weight': None}

    return classifier


# ------------------
#        SVC
# ------------------

def trained_svc(X_train, y_train, X_dev, y_dev, tune=False):
    """Trains a SVC classifier using X_train, y_train
       and tunes parameters based on X_dev, y_dev.
       Returns the classifier
    """
    classifier = svm.SVC()

    classifier.fit(X_train, y_train)

    # parameters:
    # {'kernel': 'rbf', 'C': 1.0, 'verbose': False, 'probability': False, 
    #  'degree': 3, 'shrinking': True, 'max_iter': -1, 'random_state': None, 
    #  'tol': 0.001, 'cache_size': 200, 'coef0': 0.0, 'gamma': 0.0, 
    #  'class_weight': None}

    return classifier


# ------------------
#        KNN
# ------------------

def trained_knn(X_train, y_train, X_dev, y_dev, tune=False):
    """Trains a k-Nearest Neighbours classifier using X_train, y_train
       and tunes parameters based on X_dev, y_dev.
       Returns the classifier
    """
    classifier = neighbors.KNeighborsClassifier()

    classifier.fit(X_train, y_train)
    
    # parameters: 
    # {'n_neighbors': 5, 'algorithm': 'auto', 'metric': 'minkowski', 
    # 'p': 2, 'weights': 'uniform', 'leaf_size': 30}
    
    return classifier


# --------------------
#   Ensemble methods
# --------------------

# Ensemble methods: Extra Trees Classifier,
# Ada Boost Classifier, Gradient Boosting Classifier

def trained_random_forest(X_train, y_train, X_dev, y_dev, tune=False):
    """Trains a Random Forest classifier using X_train, y_train
       and tunes parameters based on X_dev, y_dev.
       Returns the classifier
    """
    classifier = ensemble.RandomForestClassifier()

    classifier.fit(X_train, y_train)

    # parameters: 
    # {'oob_score': False, 'n_jobs': 1, 'verbose': 0, 'min_density': None, 
    #  'compute_importances': None, 'bootstrap': True, 'min_samples_leaf': 1, 
    #  'n_estimators': 10, 'min_samples_split': 2, 'random_state': None, 
    #  'criterion': 'gini', 'max_features': 'auto', 'max_depth': None}

    return classifier


def trained_ada_boost(X_train, y_train, X_dev, y_dev, tune=False):
    """Trains an Ada Boost classifier using X_train, y_train
       and tunes parameters based on X_dev, y_dev.
       Returns the classifier
    """
    classifier = ensemble.AdaBoostClassifier()

    classifier.fit(X_train, y_train)

    # parameters:
    # {'base_estimator__min_samples_split': 2, 'base_estimator__max_depth': 1, 
    #  'algorithm': 'SAMME.R', 'base_estimator__compute_importances': None, 
    #  'learning_rate': 1.0, 
    #  'base_estimator': DecisionTreeClassifier(compute_importances=None, 
    #                                           criterion='gini',
    #                                           max_depth=1, max_features=None, 
    #                                           min_density=None,
    #                                           min_samples_leaf=1, 
    #                                           min_samples_split=2, 
    #                                           random_state=None,
    #                                           splitter='best'), 
    #  'base_estimator__criterion': 'gini', 
    #  'base_estimator__max_features': None, 
    #  'base_estimator__random_state': None, 
    #  'n_estimators': 50, 'random_state': None, 
    #  'base_estimator__min_density': None, 'base_estimator__splitter': 'best', 
    #  'base_estimator__min_samples_leaf': 1}

    return classifier


def trained_gradient_boost(X_train, y_train, X_dev, y_dev, tune=False):
    """Trains a Gradient Boost classifier using X_train, y_train
       and tunes parameters based on X_dev, y_dev.
       Returns the classifier
    """
    classifier = ensemble.GradientBoostingClassifier()

    classifier.fit(X_train, y_train)

    # parameters:
    # {'loss': 'deviance', 'verbose': 0, 'subsample': 1.0, 'learning_rate': 0.1,
    #  'min_samples_leaf': 1, 'n_estimators': 100, 'min_samples_split': 2, 
    #  'init': None, 'random_state': None, 'max_features': None, 'max_depth': 3}
    
    return classifier


# --------------------
#    Majority vote
# --------------------

def majority_vote(votes):
    """Takes a list of classifier outcomes (N) for some data
    and returns the majority vote for each datapoint
    (note: more precisely this is probably the plurality vote
     because the classification is not binary.)
    """
    return array([Counter(x).most_common(1)[0][0] for x in zip(*votes)])


# --------------------
#       MAIN
# --------------------

# get the training data plus the data to be classified
train_roadnames, train_X, train_y = extract_training_data()
test_roadnames,  test_X           = extract_testing_data()

# further split the training data into a training set and a development set
X_train, X_dev, y_train, y_dev = cross_validation.train_test_split(train_X, 
                                    train_y, test_size=0.2, random_state=42)

# make classifiers
#mnb = trained_multinomial_naive_bayes(X_train, y_train, X_dev, y_dev, tune=False)
#bnb = trained_bernoulli_naive_bayes(X_train, y_train, X_dev, y_dev, tune=False)
lsvc = trained_linear_svc(X_train, y_train, X_dev, y_dev, tune=False)
#svc = trained_svc(X_train, y_train, X_dev, y_dev, tune=False)
#knn = trained_knn(X_train, y_train, X_dev, y_dev, tune=False)
#rfe = trained_random_forest(X_train, y_train, X_dev, y_dev, tune=False)
#abe = trained_ada_boost(X_train, y_train, X_dev, y_dev, tune=False)
#gbe = trained_gradient_boost(X_train, y_train, X_dev, y_dev, tune=False)

results = [classifier.predict(test_X) for classifier in [lsvc]]
#           [mnb, bnb, lsvc, svc, knn, rfe, abe, gbe]] # DEBUG
#results.append(majority_vote(results))

#print "\t".join(["", "MNB", "BNB", "SVClin", "SVC", "KNN", "RF", "ADA", "GRAD"])
for i, clf in enumerate(zip(*results)):
    print test_roadnames[i], "\t", "\t".join([str(vote) for vote in clf])

