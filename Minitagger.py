import os
import sys
import math
import time
import pickle
import random
import datetime
import subprocess
import collections
import numpy as np
import warnings
import copy

from SequenceData import SequenceData
from sklearn.metrics import confusion_matrix
from model_evaluation import report_fscore

from visualize_utils import plot_confusion_matrix

LIBLINEAR_PATH = os.path.join(os.path.dirname(__file__), "liblinear-1.96/python")
sys.path.append(os.path.abspath(LIBLINEAR_PATH))
import liblinearutil


class Minitagger(object):
    """
	Represents the Minitagger model and can be used to train a classifier and make predictions.
	Also it includes the active learning feature
	"""

    def __init__(self):
        # feature extractor that is used (it is a SequenceDataFeatureExtractor object)
        self.__feature_extractor = None
        # it stores a trained liblinearutil
        self.__liblinear_model = None
        # flag in order to print more/less log messages
        self.quiet = False
        # path to output directory for active learning
        self.active_output_path = ""
        # store predictions
        self.debug = False
        # path to output the predictions
        self.prediction_path = ""
        # path to output the predictions
        self.model_path = ""
        # number of seed examples for active learning
        self.active_seed_size = 0
        # number of examples for labeling at each iteration in active learning
        self.active_step_size = 0
        # output actively selected examples every time this value divides their number
        self.active_output_interval = 0
        # path of the project
        self.project_dir = "."
        # language of the model
        self.language = ""
        # wikiner dataset
        self.wikiner = ""

    def equip_feature_extractor(self, feature_extractor):
        """
		Equips the Minitagger with a feature extractor

		@type feature_extractor: SequenceDataFeatureExtractor
		@param feature_extractor: contains the feature extraction object
		"""

        self.__feature_extractor = feature_extractor

    def set_prediction_path(self, prediction_path):
        assert self.language != "", "you must set the language before using this function"
        self.prediction_path = os.path.join(self.project_dir, "predictions", self.language, prediction_path)
        if not os.path.isdir(self.prediction_path):
            os.mkdir(self.prediction_path)

    def set_model_path(self, model_path):
        assert self.language != "", "you must set the language before using this function"
        self.model_path = os.path.join(self.project_dir, "models_path", self.language, model_path)
        if not os.path.isdir(self.model_path):
            os.mkdir(self.model_path)

    def train(self, data_train, data_test):
        """
		Trains Minitagger on the given train data. If test data is given, it reports the accuracy of the trained model
		and the F1_score (macro average of f1_score of each label)

		@type data_train: SequenceData
		@param data_train: the training data set
		@type data_test: SequenceData
		@param data_test: the test data set
		"""

        # keep the training start timestamp
        start_time = time.time()
        assert (self.__feature_extractor.is_training), "In order to train, is_training flag should be True"

        # Extract features only for labeled instances from data_train
        [label_list, features_list, _] = self.__feature_extractor.extract_features(data_train, False, [])
        # print some useful information about the data
        if not self.quiet:
            print("{0} labeled words (out of {1})".format(len(label_list), data_train.num_of_words))
            print("{0} label types".format(len(data_train.label_count)))
            print("{0} word types".format(len(data_train.word_count)))
            print("\"{0}\" feature template".format(self.__feature_extractor.feature_template))
            print("{0} feature types".format(self.__feature_extractor.num_feature_types()))
        # define problem to be trained using the parameters received from the feature_extractor
        problem = liblinearutil.problem(label_list, features_list)
        # train the model (-q stands for quiet = True in the liblinearutil)
        self.__liblinear_model = liblinearutil.train(problem, liblinearutil.parameter("-q"))
        # training is done, set is_training to False, so that prediction can be done
        self.__feature_extractor.is_training = False

        # print some useful information
        if not self.quiet:
            num_seconds = int(math.ceil(time.time() - start_time))
            # how much did the training last
            print("Training time: {0}".format(str(datetime.timedelta(seconds=num_seconds))))
            # perform prediction on the data_test and report accuracy
        if data_test is not None:
            quiet_value = self.quiet
            self.quiet = True
            pred_labels, acc = self.predict(data_test)
            self.quiet = quiet_value

            self.__save_prediction_to_file(data_test, pred_labels)
            f1score, precision, recall = report_fscore(self.prediction_path + "/predictions.txt", wikiner=self.wikiner)

            # create some files useful for debugging
            if self.debug:
                self.__debug(data_test, pred_labels)
        return f1score, precision, recall

    def cross_validation(self, data_train, data_test=None, n_fold=2):
        """
        compute the cross validation on the data_train.
        It report the f1_score of each fold and the average with the standard deviation
        If data_test is given, it reports the accuracy on the test set and
        the F1 score (macro average f1_score of each label)

        :type data_train: SequenceData
        :param data_train: the training data_set
        :type data_test: SequenceData
        :param data_test: the test data_set
        :param n_fold: int
        :type n_fold: number of fold for the CV
        """
        start_time = time.time()
        print("======== Cross Validation =========")
        training_size = len(data_train.sequence_pairs)
        assert (n_fold >= 2), "n_fold must be at least 2"
        if n_fold > training_size:
            n_fold = training_size
            warnings.warn("n_fold can not be bigger than the size of the training set. n-fold is replace by the size "
                          "of the training set")

        # for cross validation we want to train in quiet mode to not have x times the same information
        # but we need to restore the quiet status at the end of the cross validation
        quiet_status_backup = self.quiet
        self.quiet = True

        # data_train.sequence_pairs: [[[tokens sentence1]. [label senteces1]], [[tokens sentence2]. [label senteces12]],...]
        # 1)permute randomly all sentences
        np.random.seed(123456)
        data_train.sequence_pairs = np.random.permutation(data_train.sequence_pairs)
        # 2) do cross validation
        f_score_list = []
        precision_list = []
        recall_list = []
        test_size = int(training_size / n_fold)

        if data_test is not None:
            data_train.sequence_pairs = np.append(data_train.sequence_pairs, data_test.sequence_pairs, axis=0)
        for k in range(n_fold):
            print("--- Fold: {} ---".format(k))
            test_set = copy.deepcopy(data_train)
            test_set.sequence_pairs = test_set.sequence_pairs[k * test_size: (k + 1) * test_size]
            train_set = copy.deepcopy(data_train)
            train_set.sequence_pairs = np.append(train_set.sequence_pairs[0:k * test_size],
                                                 train_set.sequence_pairs[(k + 1) * test_size:], axis=0)
            # automatilcy set to false after one training session
            self.__feature_extractor.is_training = True

            print('Total:', len(data_train.sequence_pairs))
            print('total test: ', len(test_set.sequence_pairs))
            print('total train: ', len(train_set.sequence_pairs))
            fscore, precision, recall = self.train(train_set, test_set)
            f_score_list.append(float("{:.3f}".format(fscore)))
            precision_list.append(float("{:.3f}".format(precision)))
            recall_list.append(float("{:.3f}".format(recall)))

            report_fscore(self.prediction_path + "/predictions.txt", wikiner=self.wikiner)

        print("F1-Score: ", f_score_list)
        print("F1-score Mean: {:.3f}".format(np.mean(f_score_list)))
        print("F1-score Standard deviation: {:.3f}".format(np.std(f_score_list)))
        print("---")
        print("Precision: ", precision_list)
        print("Precision Mean: {:.3f}".format(np.mean(precision_list)))
        print("Precision Standard deviation: {:.3f}".format(np.std(precision_list)))
        print("---")
        print("Recall: ", recall_list)
        print("Recall Mean: {:.3f}".format(np.mean(recall_list)))
        print("Recall Standard deviation: {:.3f}".format(np.std(recall_list)))

        print("=================================")
        self.quiet = quiet_status_backup

    def __debug(self, data_test, pred_labels):
        """
		Creates log files useful for debugging and prints a confusion matrix

		@type data_test: SequenceData object
		@param data_test: contains the testing data set
		@type pred_labels: list
		@param pred_labels: contains the prediction labels as they result from the classifier
		"""

        true_labels = self.__save_prediction_to_file(data_test, pred_labels)
        print()
        # find number of each label in the test set
        max_count = 0
        labels_list = list(data_test.label_count.keys())
        for label in labels_list:  # ["B", "I", "O"]:
            count = (np.array(true_labels) == label).sum()
            print("Number of " + label + " in the test set:", count)
            # find most frequent class in the test data set
            if count > max_count:
                max_count = count
                l = label
        print()
        # print accuracy of a naive baseline mode
        baseline_accuracy = "{0:.3f}".format(float(max_count) / len(true_labels) * 100)
        print("A naive model could predict always \'" + l + "\' with an accuracy of " + baseline_accuracy + "%")
        print()
        # create confusion matrix
        cm = confusion_matrix(true_labels, pred_labels, labels_list)
        print(labels_list)
        for row in cm:
            print(row)
            # plot_confusion_matrix(cm, labels_list)

    def save(self, model_path):
        """
		Saves the model as a directory at the given path

		@type model_path: str
		@param model_path: path to save the trained model
		"""
        # model_path = os.path.join("models_path", model_path)
        # remove model_path if it already exists
        # if os.path.exists(model_path):
        #    subprocess.check_output(["rm", "-rf", model_path])

        # make model_path directory
        # os.makedirs(model_path)
        self.set_model_path(model_path)
        # save feature extractor in the model_path directory
        ## if-else statement added on 06.02.2017
        if (self.__feature_extractor.feature_template == "relational") and (
                    self.__feature_extractor.parser_type == "spacy"):
            print("Relational model with spaCy parser cannot be saved")
        else:
            pickle.dump(self.__feature_extractor, open(os.path.join(self.model_path, "feature_extractor"), "wb"),
                        protocol=pickle.HIGHEST_PROTOCOL)
            # save trained model in the model_path directory
            liblinearutil.save_model(os.path.join(self.model_path, "liblinear_model"), self.__liblinear_model)

    def __save_prediction_to_file(self, data_test, pred_labels):
        # file to print all predictions
        file_name = os.path.join(self.prediction_path, "predictions.txt")
        f1 = open(file_name, "w")
        # file to print only sentences that contain at least one wrong label after classification
        file_name = os.path.join(self.prediction_path, "predictions_wrong.txt")
        f2 = open(file_name, "w")
        # file to print only sentences whose labels are predicted 100% correctly
        file_name = os.path.join(self.prediction_path, "predictions_correct.txt")
        f3 = open(file_name, "w")
        # index for prediction label
        pred_idx = 0
        # list to store all true labels
        true_labels = []
        true_pos_tags = []
        # iterate through the test set
        # labels_pos: [[labels]. [pos tag]] => labels = labels_pos[0] / pos_tag = labels_pos[1]
        for words, *labels_pos in data_test.sequence_pairs:
            # prediction sequence for each sentence
            pred_sequence = []
            for i in range(len(words)):
                # append label to the prediction sequence
                pred_sequence.append(pred_labels[pred_idx])
                # append label to the list of true labels
                true_labels.append(labels_pos[0][i])
                # append pos tag (if exist) to the list of true pos tag
                if len(labels_pos) == 2: true_pos_tags.append(labels_pos[1][i])
                # create line to print in the file
                line = words[i] + " " + labels_pos[0][i] + " " + pred_labels[pred_idx] + "\n"
                # write to file
                f1.write(line)
                pred_idx += 1
            # separate sentences with empty lines
            f1.write("\n")
            # check if classification error occurred
            if labels_pos[0] != pred_sequence:
                for i in range(len(labels_pos[0])):
                    # create line to print to file
                    line = words[i] + " " + labels_pos[0][i] + " " + pred_sequence[i] + "\n"
                    f2.write(line)
                # separate sentences with empty lines
                f2.write("\n")
            else:
                for i in range(len(labels_pos[0])):
                    # create line to print to file
                    line = words[i] + " " + labels_pos[0][i] + " " + pred_sequence[i] + "\n"
                    f3.write(line)
                # separate sentences with empty lines
                f3.write("\n")
        # close files
        f1.close()
        f2.close()
        f3.close()
        return true_labels

    def load(self, model_path):
        """
		Loads the model from the directory at the given path

		@type model_path: str
		@param model_path: path to load the trained model
		"""

        # load feature_extractor object (used to extract features for the test set)
        ## if-else statement added on 06.02.2017

        # if (self.__feature_extractor.feature_template == "relational") and (self.__feature_extractor.parser_type == "spacy"):
        # 	print("Relational model with spaCy parser cannot be loaded")
        # else:
        # 	self.__feature_extractor = pickle.load(open(os.path.join(model_path, "feature_extractor"), "rb"))
        # 	# load trained model
        # 	self.__liblinear_model = liblinearutil.load_model(os.path.join(model_path, "liblinear_model"))
        self.set_model_path(model_path)
        try:
            print(self.model_path)
            self.__feature_extractor = pickle.load(open(os.path.join(self.model_path, "feature_extractor"), "rb"))
            # load trained model
            self.__liblinear_model = liblinearutil.load_model(os.path.join(self.model_path, "liblinear_model"))
        except:
            raise Exception("No files found in the model path")

    def predict(self, data_test):
        """
		Predicts tags in the given data
		It reports the accuracy if the data is fully labeled

		@type data_test: SequenceData
		@param data_test: the test data set
		@return: the predicted labels, the accuracy, the f1_score
		"""

        # keep starting timestamp
        start_time = time.time()
        assert (not self.__feature_extractor.is_training), "In order to predict, is_training should be False"

        # Extract features on all instances (labeled or unlabeled) of the test set
        [label_list, features_list, _] = self.__feature_extractor.extract_features(data_test, True, [])
        # pass them to liblinearutil for prediction
        pred_labels, (acc, _, _), _ = liblinearutil.predict(label_list, features_list, self.__liblinear_model, "-q")

        # print some useful information
        if not self.quiet:
            num_seconds = int(math.ceil(time.time() - start_time))
            # estimate prediction time
            print("Prediction time: {0}".format(str(datetime.timedelta(seconds=num_seconds))))
            # report accuracy if the data is fully labeled
            if not data_test.is_partially_labeled:
                print("Per-instance accuracy: {0:.3f}%".format(acc))
            else:
                print("Not reporting accuracy: test data is not fully labeled")

        # convert predicted labels from integer IDs to strings.
        for i, label in enumerate(pred_labels):
            pred_labels[i] = self.__feature_extractor.get_label_string(label)

        return pred_labels, acc

    def __find_most_frequent_words(self, data_train):
        """
		Computes the (active_seed_size) most frequent word types in data_train

		@type data_train: SequenceData
		@param data_train: t# Train for the first time.he training data set
		@return: the X most frequent words (X = active_seed_size) in train set
		"""

        # data_train.word_count is a dictionary with key = word , value = frequency of word in the train set
        # sort dictionary in descending order
        sorted_wordcount_pairs = sorted(data_train.word_count.items(),
                                        key=lambda type_count: type_count[1], reverse=True)
        # take the X most frequent words (X = active_seed_size)
        seed_wordtypes = [wordtype for wordtype, _ in sorted_wordcount_pairs[:self.active_seed_size]]
        return seed_wordtypes

    def __find_frequent_word_locations(self, data_train, seed_wordtypes):
        """
		Finds random locations of the most frequent words

		@type data_train: SequenceData
		@param data_train: train data set
		@type seed_wordtypes: list
		@param seed_wordtypes: list of the X most frequent words (X = active_seed_size) in train set
		@return: a random location for each word in the seed_wordtypes
		each location is a tuple of the form (sequence number, position of the word in the sequence)
		"""

        occurring_locations = collections.defaultdict(list)
        # iterate through all sequences and words in the train set
        for sequence_num, (word_sequence, _) in enumerate(data_train.sequence_pairs):
            for position, word in enumerate(word_sequence):
                # append location is the current word is one of the seed_wordtypes
                if word in seed_wordtypes:
                    occurring_locations[word].append((sequence_num, position))
        # take one random position for each word
        locations = [random.sample(occurring_locations[wordtype], 1)[0] for wordtype in seed_wordtypes]
        return locations

    def __make_data_from_locations(self, data_train, locations, skip_extraction):
        """
		Makes SequenceData out of a subset of data_train from given location=(sequence_num, position) pairs

		@type data_train: SequenceData
		@param data_train: the train data set
		@type locations: list
		@param locations: list of tuples of locations for each one of the  X most frequent words
		(X = active_seed_size) in train set
		@return: a SequenceData object. It contains all words in the sequences. For words corresponding to locations,
		the true labels are used. Otherwise, the label is None
		"""

        # find all selected positions for each sequence
        selected_positions = collections.defaultdict(list)
        for (sequence_num, position) in locations:
            selected_positions[sequence_num].append(position)

        sequence_list = []
        for sequence_num in selected_positions:
            word_sequence, label_sequence = data_train.sequence_pairs[sequence_num]
            # initialize all labels to None
            selected_labels = [None for _ in range(len(word_sequence))]
            # take the right label for the words in the selected positions
            for position in selected_positions[sequence_num]:
                selected_labels[position] = label_sequence[position]
                # skip each word that corresponds to a position in the locations list
                # this example will not be selected again
                skip_extraction[sequence_num][position] = True
            # the sequence_list contains all words in a sequence. For words corresponding to locations, the true labels
            # are used. Otherwise, the label is None
            sequence_list.append((word_sequence, selected_labels))

        # make a SequenceData object using the sequence_list
        selected_data = SequenceData(sequence_list)
        return selected_data

    def __train_silently(self, data_selected):
        """
		Trains on the selected data in silent mode

		@type data_selected: SequenceData
		@param data_selected: the selected subset of the training data set. Some words have correct labels
		and other words have None labels
		"""

        # reset for training
        self.__feature_extractor.is_training = True
        quiet_value = self.quiet
        self.quiet = True
        # no need for test set here.
        self.train(data_selected, None)
        self.quiet = quiet_value

    def __interval_report(self, data_selected, data_test, logfile):
        """
		Reports accuracy in the specified interval

		@type data_selected: SequenceData
		@param data_selected: selected data based on locations
		@type data_test: SequenceData
		@param data_test: test data set
		@type logfile: file
		@param logfile: file used for log messages
		@return: None if it is not time to report output yet
		"""

        # report only at each interval
        if data_selected.num_labeled_words % self.active_output_interval != 0:
            return

        # test on the development data if any available
        if data_test is not None:
            quiet_value = self.quiet
            self.quiet = True
            # make prediction and return accuracy
            _, acc = self.predict(data_test)
            self.quiet = quiet_value
            message = "{0} labels: {1:.3f}%".format(data_selected.num_labeled_words, acc)
            print(message)
            logfile.write(message + "\n")
            logfile.flush()

        # Output the selected labeled examples so far.
        file_name = os.path.join(self.active_output_path, "example" + str(data_selected.num_labeled_words))
        with open(file_name, "w") as outfile:
            outfile.write(data_selected.__str__())

    def __find_confidence_index_pairs(self, confidence_index_pairs, scores_list):
        """
		Estimates the confidence index pairs

		@type confidence_index_pairs:list
		@param confidence_index_pairs: list of tuples like (confidence, index)
		@type: list
		@param scores_list: list of scores for each word
		"""

        for index, scores in enumerate(scores_list):
            sorted_scores = sorted(scores, reverse=True)

            # handle the binary case
            # liblinear gives only 1 score whose sign indicates the class (+ versus -)
            confidence = sorted_scores[0] - sorted_scores[1] if len(scores) > 1 else abs(scores[0])
            confidence_index_pairs.append((confidence, index))

    def train_actively(self, data_train, data_test):
        """
		Does margin-based active learning on the given data

		@type data_train: list
		@param data_train: list of training data set
		@type data_test: list
		@param data_test: list of test data set
		"""

        # for active learning, every data point (word) in the data_train should be labeled
        assert (not data_train.is_partially_labeled), "for active learning, every data point (word) " \
                                                      "in the data_train should be labeled"

        # keep track of which examples can be still selected for labeling.
        __skip_extraction = []
        # initialize __skip_extraction to False for every label in every sequence,
        # i.e. nothing is skipped at the beginning
        for _, label_sequence in data_train.sequence_pairs:
            __skip_extraction.append([False for _ in label_sequence])

        # create an output directory
        if os.path.exists(self.active_output_path):
            subprocess.check_output(["rm", "-rf", self.active_output_path])
        os.makedirs(self.active_output_path)
        logfile = open(os.path.join(self.active_output_path, "log"), "w")

        # take the X most frequent words (X = active_seed_size)
        seed_wordtypes = self.__find_most_frequent_words(data_train)
        # select a random location (sequence number, position in the sequence) of each selected type for a seed example
        locations = self.__find_frequent_word_locations(data_train, seed_wordtypes)
        # build a SequenceData object from the selected locations
        data_selected = self.__make_data_from_locations(data_train, locations, __skip_extraction)
        # train for the first time
        self.__train_silently(data_selected)
        self.__interval_report(data_selected, data_test, logfile)

        while len(locations) < data_train.num_labeled_words:
            # extract features for the remaining (i.e. not in the skip list) labeled examples
            [label_list, features_list, location_list] = self.__feature_extractor.extract_features(
                data_train, False, __skip_extraction)
            # make predictions on the remaining (i.e. not in the skip list) labeled examples
            _, _, scores_list = liblinearutil.predict(label_list, features_list, self.__liblinear_model, "-q")

            # Compute "confidence" of each prediction:
            #   max_{y} score(x,y) - max_{y'!=argmax_{y} score(x,y)} score(x,y')
            confidence_index_pairs = []
            self.__find_confidence_index_pairs(confidence_index_pairs, scores_list)

            # Select least confident examples for next labeling.
            confidence_index_pairs.sort()
            for _, index in confidence_index_pairs[:self.active_step_size]:
                locations.append(location_list[index])
            data_selected = self.__make_data_from_locations(data_train, locations, __skip_extraction)
            self.__train_silently(data_selected)  # Train from scratch.
            self.__interval_report(data_selected, data_test, logfile)

        logfile.close()
