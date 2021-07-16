import numpy as np


def forward(emit_matrix, trans_matrix, prior_matrix, sentences):
    """ Does the forward computation, which creates the alpha matrix

    :param sentences:
    :param emit_matrix:
    :param trans_matrix:
    :param prior_matrix:
    :return:
    """
    longest_sentence = []
    for sentence in sentences:
        if len(sentence) > len(longest_sentence):
            longest_sentence = sentence
    longest_sentence = len(longest_sentence)
    alpha_matrix = np.zeros((longest_sentence, prior_matrix.size))
    print(alpha_matrix.shape)