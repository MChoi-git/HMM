import numpy as np


def parse_tags_words_to_index(index_to_word_filename, index_to_tag_filename, data_filename):
    """ Extracts and separates the words from their tags. Also creates translation dictionaries for indexing.

    :param index_to_word_filename: Path to the index2word .txt file
    :param index_to_tag_filename: Path to the index2tag .txt file
    :param data_filename: Path to the train/test/validation dataset
    :return: Returns an array of sentences with words, an array of sentences with tags, and translation dictionaries for the words and tags
    """
    # Create enums out of translation files and format the sentences into numpy arrays
    with open(index_to_tag_filename, "r") as f_index_to_tag, \
            open(index_to_word_filename, "r") as f_index_to_word, \
            open(data_filename, "r") as f_data:
        raw_index_to_word = enumerate(f_index_to_word.readlines())
        raw_index_to_tag = enumerate(f_index_to_tag.readlines())
        raw_data = map(str.strip, f_data.readlines())
        data = [sentence.replace("_", " ").split(" ") for sentence in raw_data]
    # Split up the words and their tags
    sentences = [sentence[0::2] for sentence in data]
    tags = [sentence[1::2] for sentence in data]
    # Create translation dictionaries
    word_dict = {word.strip(): index for index, word in raw_index_to_word}
    tag_dict = {tag.strip(): index for index, tag in raw_index_to_tag}
    return sentences, tags, word_dict, tag_dict


def create_emit_matrix(sentences, tags, word_dict, tag_dict):
    """ Creates the emission matrix A, which represents the conditional P(X_t = k|Y_t = j). The calculation of the conditional
    is equivalent to: #(x_m == tag_j)/#(x_m to x_M == tag_j) over the entire dataset.

    :param sentences: Array of all sentences from dataset
    :param tags: Array of all tags from dataset
    :param word_dict: Dictionary where keys=index, values=words (unique)
    :param tag_dict: Dictionary where keys=index, values=tags (unique)
    :return: Emission matrix
    """
    emit_matrix = np.zeros((len(tag_dict), len(word_dict)))
    # Count the occurrences of x_m:tag pairs, and place them in emission matrix
    for i, sentence in enumerate(sentences):
        for j, word in enumerate(sentence):
            word_index = word_dict.get(word)
            tag_index = tag_dict.get(tags[i][j])
            emit_matrix[tag_index][word_index] += 1
    # Pseudo-count
    emit_matrix += 1
    # Normalize counts, rows sum to 1
    emit_matrix /= emit_matrix.sum(axis=1, keepdims=True)
    return emit_matrix


def create_trans_matrix(sentences, tags, word_dict, tag_dict):
    """ Creates the transition matrix B, which represents the conditional P(Y_t = k|Y_t-1 = j). The calculation of the conditional
    is equivalent to: #(tag => tag_t+1)/#(tag => any tag) over the entire dataset.

    :param sentences: Array of all sentences from dataset
    :param tags: Array of all tags from dataset
    :param word_dict: Dictionary where keys=index, values=words (unique)
    :param tag_dict: Dictionary where keys=index, values=tags (unique)
    :return: Transition matrix
    """
    trans_matrix = np.zeros((len(tag_dict), len(tag_dict)))
    # Count the occurrences of tag:tag_t+1 pairs, and place them in emission matrix
    for i, sentence in enumerate(sentences):
        for j, word in enumerate(sentence):
            if j < len(sentence) - 1:
                current_tag = tags[i][j]
                next_tag = tags[i][j + 1]
                current_tag_index = tag_dict[current_tag]
                next_tag_index = tag_dict[next_tag]
                trans_matrix[current_tag_index][next_tag_index] += 1
    # Pseudo-count
    trans_matrix += 1
    # Normalize counts, rows sum to 1
    trans_matrix /= trans_matrix.sum(axis=1, keepdims=True)
    return trans_matrix


def create_init_matrix(sentences, tags, word_dict, tag_dict):
    """ Creates the initialization matrix Pi, which represents P(Y_1 == j). The calculation of the conditional
    is equivalent to: #(Y_1 == j)/#(tags in each Y1) over the entire dataset.

    :param sentences: Array of all sentences from dataset
    :param tags: Array of all tags from dataset
    :param word_dict: Dictionary where keys=index, values=words (unique)
    :param tag_dict: Dictionary where keys=index, values=tags (unique)
    :return: Initialization matrix
    """
    init_matrix = np.zeros(len(tag_dict))
    for i, tag_row in enumerate(tags):
        tag_index = tag_dict[tag_row[0]]
        init_matrix[tag_index] += 1
    # Pseudo-count
    init_matrix += 1
    # Normalize counts, row sums to 1
    init_matrix /= init_matrix.sum(keepdims=True)
    return init_matrix


def matrix_to_txt(filename, matrix):
    np.savetxt(filename, matrix)