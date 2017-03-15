import os
import argparse
from utils import analyze_data
from SequenceData import SequenceData
from SequenceDataFeatureExtractor import SequenceDataFeatureExtractor
from Minitagger import Minitagger

# Used for instances without gold labels
ABSENT_GOLD_LABEL = "<NO_GOLD_LABEL>"


def main(args):
    # if specified, just analyze the given data and return.
    # this data can be a prediction output file.
    if args.analyze:
        analyze_data(args.data_path)
        return



    # train or use a tagger model on the given data.
    minitagger = Minitagger()
    minitagger.quiet = args.quiet
    sequence_data = SequenceData(args.data_path,args.pos_tag)

    if args.project_dir:
        minitagger.project_dir = args.project_dir

    minitagger.language = args.language

    minitagger.set_prediction_path(args.prediction_path, args.embedding_size if args.embedding_size else None)
    minitagger.set_model_path(args.model_path, args.embedding_size if args.embedding_size else None)

    if args.wikiner:
        minitagger.wikiner = True

    if args.enable_embeddings:
        assert args.embedding_path, "Embeddings path should be specified when embeddings are enabled"

    if args.train:
        # initialize feature extractor with the right feature template
        feature_extractor = SequenceDataFeatureExtractor(args.feature_template, args.language, args.parser_type,
                                                         args.enable_embeddings)
        # load bitstring or embeddings data
        if args.embedding_path:
            feature_extractor.load_word_embeddings(args.embedding_path, args.embedding_size)
        if args.bitstring_path:
            feature_extractor.load_word_bitstrings(args.bitstring_path)
        # equip Minitagger with the appropriate feature extractor
        minitagger.equip_feature_extractor(feature_extractor)
        test_data = SequenceData(args.test_data_path, args.pos_tag) if args.test_data_path else None
        if test_data is not None:
            # Test data should be fully labeled
            assert (not test_data.is_partially_labeled), "Test data should be fully labeled"
        minitagger.debug = args.debug
        if minitagger.debug:
            assert args.prediction_path, "Path for prediction should be specified"
        # normal training, no active learning used

        if not args.active:
            assert args.model_path
            minitagger.train(sequence_data, test_data)
            minitagger.save(args.model_path)

            # minitagger.cross_validation(sequence_data, test_data, 5)

        # do active learning on the training data
        else:
            assert (args.active_output_path), "Active output path should not be empty"
            # assign the right parameters to minitagger
            minitagger.active_output_path = args.active_output_path
            minitagger.active_seed_size = args.active_seed_size
            minitagger.active_step_size = args.active_step_size
            minitagger.active_output_interval = args.active_output_interval
            minitagger.train_actively(sequence_data, test_data)
    # predict labels in the given data.
    else:
        assert args.model_path
        minitagger.load(args.model_path)
        pred_labels, _ = minitagger.predict(sequence_data)

        # optional prediction output
        # write predictions to file
        if args.prediction_path:
            file_name = os.path.join(args.project_dir, args.prediction_path, "predictions.txt")
            with open(file_name, "w") as outfile:
                label_index = 0
                for sequence_num, (word_sequence, label_sequence) in enumerate(sequence_data.sequence_pairs):
                    for position, word in enumerate(word_sequence):
                        if not label_sequence[position] is None:
                            gold_label = label_sequence[position]
                        else:
                            gold_label = ABSENT_GOLD_LABEL
                        outfile.write(word + "\t" + gold_label + "\t" + pred_labels[label_index] + "\n")
                        label_index += 1
                    if sequence_num < len(sequence_data.sequence_pairs) - 1:
                        outfile.write("\n")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data_path", type=str, help="path to data (used for training/testing)", required=True)

    argparser.add_argument("--model_path", type=str, help="path to model directory", required=True)
    argparser.add_argument("--prediction_path", type=str, help="path to output data for prediction")
    argparser.add_argument("--train", action="store_true", help="train the tagger on the given data")
    argparser.add_argument("--feature_template", type=str, default="baseline",
                           help="feature template (default: %(default)s)")
    argparser.add_argument("--embedding_path", type=str, help="path the folder containing word embeddings")
    argparser.add_argument("--embedding_size", type=int, choices=[50, 100, 200,300],
                           help="the size of the word embedding vectors")

    argparser.add_argument("--quiet", action="store_true", help="no messages")
    argparser.add_argument("--test_data_path", type=str, help="path to test data set (used for training)")
    argparser.add_argument("--active", action="store_true", help="perform active learning on the given data")
    argparser.add_argument("--active_output_path", type=str, help="path to output directory for active learning")
    argparser.add_argument("--active_seed_size", type=int, default=1,
                           help="number of seed examples for active learning (default: %(default)d)")
    argparser.add_argument("--active_step_size", type=int, default=1,
                           help="number of examples for labeling at each iteration in active learning (default: %("
                                "default)d)")
    argparser.add_argument("--active_output_interval", type=int, default=100,
                           help="output actively selected examples every time this value divides their number"
                                "(default: %(default)d)")
    argparser.add_argument("--language", type=str, choices=["en", "de", "fr"],
                           help="language of the data set [en, de,fr]", required=True)
    argparser.add_argument("--parser_type", type=str, choices=["spacy", "stanford", "syntaxnet"],
                           help="type of parser to be used for relational feature extraction [default = spacy]",
                           default="spacy")
    argparser.add_argument("--enable_embeddings", action="store_true",
                           help="enriches the relational feature space with word embeddings")
    argparser.add_argument("--debug", action="store_true", help="produce some files for debugging")
    argparser.add_argument("--pos_tag", action="store_true",
                           help="indicate if the part-of-speech tag is present or not")
    argparser.add_argument("--project_dir", type=str, help="directory of the path")
    argparser.add_argument("--wikiner", action="store_true",
                           help="if we are using wikiner dataset, use this arg to use appropriate scoring function")

    parsed_args = argparser.parse_args()

    main(parsed_args)

    #from model_evaluation import report_fscore

    #report_fscore("test_predictions.txt", wikiner=True)
    #report_fscore("../old_model_and_prediction/predictions/en/predictions_wikiner/predictions_wrong.txt", wikiner =True)
