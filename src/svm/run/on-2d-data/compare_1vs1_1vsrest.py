import numpy as np
from succinctly.multi_class import load_X, load_y, load_X_test, load_y_test
from svm.algorithms import kernels
from svm.algorithms.multi_classifier import OneVsOneClassifier, OneVsRestClassifier
from ulti.log import create_logger
from ulti.accuracy import calculate_accuracy

if __name__ == '__main__':
    # 1. get data
    X = load_X()
    y = load_y()
    X_test = load_X_test()
    y_test = load_y_test()
    dataset_name = 'my dataset'

    # init
    args_c      = np.array([1, 10, 50, 100, 200, 400, 600, 800, 1000, 2000])      # get_args_c('args_c.txt')
    kernels     = np.array([kernels.linear])                         # kernels.get_all_kernel()

    _1vs1       = OneVsOneClassifier()
    _1vsrest    = OneVsRestClassifier()

    classifiers = np.array([_1vs1, _1vsrest])
    classifer_names       = np.array(['_1vs1', '_1vsrest'])

    for c in args_c:
        for kernel in kernels:
            for i, classifier in enumerate(classifiers):
                ## create a new logger
                file_name = 'log/{}_{}{}.log'.format(c, kernel.func_name, classifer_names[i])
                logger = create_logger(file_name)
                logger.info('*'*100)
                logger.info('dataset name        {}'.format(dataset_name))
                logger.info('classifer_names     {}'.format(classifer_names[i]))
                logger.info('c                   {}'.format(c))
                logger.info('kernel              {}'.format(kernel.func_name))
                logger.info('#'*50)

                # fit model
                classifier.fit(X, y, kernel, c)
                logger.info('time_fit            {}'.format(classifier.time_fit))
                logger.info('#'*50)

                # predit on training_set
                y_predit_train                      = classifier.predit(X)
                logger.info('y_predit_train      \n {}'.format(y_predit_train))
                mis_indices_train, accuracy_train   = calculate_accuracy(y, y_predit_train)
                logger.info('accuracy_train      {}'.format(accuracy_train))
                logger.info('n_missamples_train  {}'.format(len(mis_indices_train)))
                logger.info('mis_indices_train  \n {}'.format(mis_indices_train))
                logger.info('y_mis              \n {}'.format(y[mis_indices_train]))
                logger.info('y_predit_false     \n {}'.format(y_predit_train[mis_indices_train]))
                logger.info('#'*50)

                # predit on testing_set
                y_predit_test   = classifier.predit(X_test)
                logger.info('y_predit_test      \n {}'.format(y_predit_test))
                mis_indices_test, accuracy_test   = calculate_accuracy(y_test, y_predit_test)
                logger.info('accuracy_test       {}'.format(accuracy_test))
                logger.info('n_missamples_test   {}'.format(len(mis_indices_test)))
                logger.info('mis_indices_test   \n {}'.format(mis_indices_test))
                logger.info('y_mis              \n {}'.format(y_test[mis_indices_test]))
                logger.info('y_predit_false     \n {}'.format(y_predit_test[mis_indices_test]))
