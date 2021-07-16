import numpy as np


# IMPORTANT: The alpha is not a matix, but a 3D TENSOR, JxMxD
def forward(emit_matrix, trans_matrix, prior_matrix, sentences):
    """ Does the forward computation, which creates the alpha matrix

    :param sentences:
    :param emit_matrix:
    :param trans_matrix:
    :param prior_matrix:
    :return:
    """
    # longest_sentence = []
    # for sentence in sentences:
    #     if len(sentence) > len(longest_sentence):
    #         longest_sentence = sentence
    # longest_sentence = len(longest_sentence)
    # alpha_matrix = np.zeros((longest_sentence, prior_matrix.size))
    # Calculate all alpha_1 for each tag
    alpha_prior = emit_matrix[:, 0] * prior_matrix
    alpha_prior /= alpha_prior.sum(axis=0, keepdims=True)
    alpha_previous = alpha_prior
    alpha_matrix = np.zeros((len(emit_matrix[0]) - 1, len(emit_matrix)))
    # Calculate the summation term in alpha for t > 1
    for x_n in range(len(emit_matrix[0]) - 1):
        for tag_n in range(len(trans_matrix)):
            alpha_matrix[tag_n][x_n] = emit_matrix[tag_n][x_n + 1] * (alpha_previous * trans_matrix[tag_n]).sum()
        alpha_matrix[:, x_n] /= alpha_matrix[:, x_n].sum()
        alpha_previous = alpha_matrix[:, x_n]
    print(alpha_matrix)
    return