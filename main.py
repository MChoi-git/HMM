import sys
import learnhmm as hmm
import forwardbackward as fb
import numpy as np
from itertools import chain


def main():
    """ Main script for running learnhmm and forwardbackward

    Argv:
        train_input: Path to training dataset
        index_to_word: Path to index2word .txt file
        index_to_tag: Path to index2tag .txt file
        hmm_prior: Path to output file for the prior matrix (initialization)
        hmm_emit: Path to output file for the emission matrix
        hmm_trans: Path to output file for the transition matrix

    :return: Returns 1 on success, -1 on error
    """
    # Check for correct # args
    if len(sys.argv) != 7:
        print("Error: Incorrect program syntax.")
        return -1
    # Assign args
    train_input, index_to_word, index_to_tag, hmm_prior, hmm_emit, hmm_trans = sys.argv[1:]
    # Get the parsed data from the input files
    sentences, tags, word_dict, tag_dict = hmm.parse_tags_words_to_index(index_to_word, index_to_tag, train_input)
    # Create the emission, transition, and initialization matrices
    emit_matrix = hmm.create_emit_matrix(sentences, tags, word_dict, tag_dict)
    trans_matrix = hmm.create_trans_matrix(sentences, tags, word_dict, tag_dict)
    prior_matrix = hmm.create_init_matrix(sentences, tags, word_dict, tag_dict)
    # Send the matrices to their files
    hmm.matrix_to_txt(hmm_emit, emit_matrix)
    hmm.matrix_to_txt(hmm_trans, trans_matrix)
    hmm.matrix_to_txt(hmm_prior, prior_matrix)
    # Do the forward computation for alpha
    alpha = fb.forward(emit_matrix, trans_matrix, prior_matrix, sentences, tags, word_dict, tag_dict)
    beta = fb.backward(emit_matrix, trans_matrix, prior_matrix, sentences, tags, word_dict, tag_dict)
    conditional_probabilities = fb.get_conditional_probabilities(alpha, beta)
    # Lazy prediction section
    # Should be refactored ASAP
    tags_index = []
    preds = []
    # Get the label indexes according to prob dist.
    for example in conditional_probabilities:
        tags_index_example = np.argmax(example, axis=0)
        tags_index.append(tags_index_example)
    # Get the predictions and put them back in format
    for i, example in enumerate(tags_index):
        for j in range(len(example)):
            preds += [f"{sentences[i][j]}_{list(tag_dict.keys())[list(tag_dict.values()).index(example[j])]}"]
    # Calculate the accuracy
    wrong = 0
    flat_sentences = list(chain.from_iterable(sentences))
    flat_tags = list(chain.from_iterable(tags))
    for i in range(len(flat_sentences)):
        if preds[i] != f"{flat_sentences[i]}_{flat_tags[i]}":
            wrong += 1
    accuracy = 1 - wrong / len(flat_sentences)
    print(f"Accuracy: {accuracy}")


if __name__ == "__main__":
    main()