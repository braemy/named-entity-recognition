import re
import sys
#import spacy
import pickle
# from nltk.tag import StanfordNERTagger
# from nltk.parse.stanford import StanfordDependencyParser


## created on 07.02.2017
class RelationalFeatureAnalyzer(object):
	"""
	Performs relational feature analysis for a given sequence of words
	"""

	def __init__(self, parser_type, language):
		# define parser type, i.e. which parser to use
		self.parser_type = parser_type
		# define language to be used (language of the data set)
		self.language = language
		if self.parser_type == "spacy":
			print("Loading spacy...")
			self.nlp = spacy.load(self.language)
			print("Spacy loaded")
		elif self.parser_type == "stanford":
			if self.language != "en":
				print("Only English is supported at the moment")
				sys.exit()
			self.stanford_relational_information = pickle.load(open("../coreNLP/relational_information_stanford.pickle", "rb"))
		elif self.parser_type == "syntaxnet":
			if self.language != "en":
				print("Only English is supported at the moment")
				sys.exit()
			# load all sentences of the data set
			self.sentences = self.__load_sentences()
			# build a dictionary:
			# key = sentence
			# value = relational information extracted from syntaxnet for a particular sentence
			self.relational_analysis_dict = self.__build_relational_analysis_mapping()
			# check if all sentences are contained as keys in the dictionary
			assert len(self.relational_analysis_dict.keys()) == len(self.sentences), "sentences and relational_analysis_dict should have the same length (data set may contain duplicate sentences)"
		else:
			raise Exception("Unsupported parser type {0}".format(self.parser_type))

	## added on 09.02.2017
	def __load_sentences(self):
		"""
		Loads and returns a list of all the sentences of the data set

		@return: a list of all the sentences in the train and test set
		"""

		# file name with all the sentences
		file_name = "../syntaxnet/dataset.txt"
		# list to store the sentences
		sentences = []
		f = open(file_name, "r")
		# read file and store sentences
		for line in f:
			sentences.append(line.strip())
		f.close()
		return sentences

	## added on 09.02.2017
	def __build_relational_analysis_mapping(self):
		"""
		Creates a dictionary
		key = sentence
		value = relational analysis of the sentence

		@ return: dictionary that maps each sentence to its relational analysis
		"""

		# file that contains the full syntaxnet parser output
		file_name = "../syntaxnet/parser_output.txt"
		f = open(file_name, "r")
		# list to contain relational information about each sentence
		sentence = []
		relational_analysis_dict = {}
		sentence_idx = 0
		# iterate through all lines in the file
		for line in f:
			# split each line with tab delimiter
			tokens = line.split("\t")
			if tokens[0] != "\n":
				# non-empty line indicates the there is more information about the same sentence
				sentence.append(tokens)
			else:
				# blank new line means the end of a sentence
				# the information collected so far contains the relational analysis of a specific sentence
				# put this information into a dictionary with the sentence as a key
				relational_analysis_dict[self.sentences[sentence_idx]] = sentence
				# increment sentence index
				sentence_idx += 1
				# flush the sentence list to store information about next sentence
				sentence = []
		f.close()
		# parse relational_analysis_dict and store only useful information
		self.__parse_relational_analysis_mapping(relational_analysis_dict)
		return relational_analysis_dict

	## added on 09.02.2017
	def __parse_relational_analysis_mapping(self, relational_analysis_dict):
		"""
		Keeps only the necessary information from the syntaxnet output. This information should be in the same format
		as the the one given from the spacy and Stanford parser

		@type relational_analysis_dict: dict
		@param relational_analysis_dict: dictionary with sentence as key and a list containing the relational analysis of each sentence as a value
		"""

		# iterate though the dict
		for key, value in relational_analysis_dict.items():
			# list to store the useful relational analysis
			relational_analysis = []
			for idx in range(len(value)):
				# get the arc label
				arc_label = value[idx][7]
				# get the arc head (when the arc head is 0 (ROOT), replace the arc head with the word itself)
				if value[idx][6] == '0':
					arc_head = value[idx][1]
				else:
					arc_head = value[int(value[idx][6])-1][1]
				# aggregate relational analysis for each sentence
				relational_analysis.append([arc_label, arc_head])
			# store new information to the dictionary
			relational_analysis_dict[key] = relational_analysis

	## added on 08.02.2017
	def __build_parser_mapping(self, stanford_dependencies):
		"""
		Performs a mapping between word position and word id
		Example of word_sequence = [I, saw, the, cat]
		position 0 = None and declares the root of the sentence
		position 1 = maps to the first word in a word sequence, i.e. I

		Performs a mapping between arcs and arc labels
		Example:
		0 -> 2: ROOT represents an arc from position 0 to the second word of the sentence and the arc label is ROOT
		2 -> 1: nsubj represents an arc from position 2 (saw) to position 1 (I), i.e. from the second to the first word, with
		arc label nsubj

		@type stanford_dependencies: str
		@param stanford_dependencies: string containing the result of the dependency parser
		@return: two dictionaries with word and arc mappings
		"""

		# dictionary for arc mapping: key = arc e.g. 2 -> 1, value = arc label e.g. nsubj
		arcs_dict = {}
		# dictionary for word mapping: key = word position, value = word, e.g. 2: saw
		words_dict = {}
		# parse the string
		for line in stanford_dependencies.split('\n'):
			# line should not be empty
			if len(line) != 0:
				# check if line starts with digit (this means that it holds dependency information)
				if line[0].isdigit():
					# if line contains information about an arc (declared with ->)
					if "->" in line:
						# parse line and extract necessary information
						# get the arc e.g. 2 -> 1
						arc = line.split('[')[0].strip()
						# get the arc label
						arc_label = re.findall(r'\"(.+?)\"',line.split('[')[1])[0]
						# put in the dictionary
						arcs_dict[arc] = arc_label
					# line contains mapping of word and position
					else:
						# extract position
						position = int(line.split()[0])
						# extract word
						word = re.findall(r'\((.+?)\)',line.split('[')[1])[0]
						# put in dictionary
						words_dict[position] = word
		return arcs_dict, words_dict

	## added on 06.02.2017
	def __relational_features_analysis_spacy(self, word_sequence):
		"""
		Takes a list of words that represents a sentence and performs relational feature analysis using the spacy parser.
		Information about the arc label, the arc head, the entity id and the iob id of each word is extracted

		@type word_sequence: list
		@param word_sequence: sequence of words
		@return: a list of lists with relational features. The position in the list corresponds to the position
		of the word in the sentence. The inner list contains the relational information about each word
		"""

		# join words to re-create sentence
		sentence = " ".join(word_sequence)
		# pass sentence to nlp object
		doc = self.nlp(sentence)
		# create a list to store relational information
		relational_analysis = []
		# append relational information for each word in the sentence
		for token in doc:
			# punctuation should be skipped because Stanford parser does not
			# handle it (the parsers should treat punctuation in the same way)
			if token.is_punct:
				continue
			# the position in the list indicates the position of the word in the sentence
			# relational information is given in the following way
			# ['label of incoming arc' | 'token at the source of the arc' | 'entity type' | 'entity iob']
			# relational_analysis.append([token.dep_, token.head.text, token.ent_type, token.ent_iob_])
			relational_analysis.append([token.dep_, token.head.text, token.pos_])
		return relational_analysis

	## added on 08.02.2017
	def __relational_features_analysis_stanford(self, word_sequence):
		"""
		Takes a list of words that represents a sentence and performs relational feature analysis using the Stanford parser.
		Information about the arc label and the arc head of each word is extracted

		@type word_sequence: list
		@param word_sequence: sequence of words
		@return: a list of lists with relational features. The position in the list corresponds to the position
		of the word in the sentence. The inner list contains the relational information about each word
		"""

		sentence = " ".join(word_sequence)
		relational_analysis = self.stanford_relational_information[sentence]
		return relational_analysis

	def __relational_features_analysis_syntaxnet(self, word_sequence):
		"""
		Returns the relational information about a given sequence of words, i.e. sentence from the data set

		@type word_sequence: list
		@param word_sequence: list of words in a sentence
		@return: the relational analysis of the given sentece
		"""

		# build string of sentence from the list of words
		sentence = " ".join(word_sequence)
		# extract and return relational analysis of the given sentence
		return self.relational_analysis_dict[sentence]

	def analyze_relational_features(self, word_sequence):
		"""
		Analyses a sequence of words and extracts relational information

		@type word_sequence: list
		@param word_sequence: list of words
		"""

		# call the appropriate function to perform relational analysis based
		# on the type of parser
		if self.parser_type == "spacy":
			relational_analysis = self.__relational_features_analysis_spacy(word_sequence)
		if self.parser_type == "stanford":
			relational_analysis = self.__relational_features_analysis_stanford(word_sequence)
		if (self.parser_type == "syntaxnet"):
			relational_analysis = self.__relational_features_analysis_syntaxnet(word_sequence)
		# the length of the relational_analysis list and the word_sequence
		# must be the same (relational_analysis list should have one entry
		# for each word in the word_sequence list)
		assert len(relational_analysis) == len(word_sequence), "Length mismatch"
		return relational_analysis
