import numpy as np


# IMPORTANT: The alpha is not a matrix, but a 3D TENSOR, JxMxD
def forward(emit_matrix, trans_matrix, prior_matrix, sentences, tags, word_dict, tag_dict):
    """ Does the forward computation, which creates the alpha matrix

    :param emit_matrix:
    :param trans_matrix:
    :param prior_matrix:
    :param sentences:
    :param tags:
    :return:
    """
    # longest_sentence = []
    # for sentence in sentences:
    #     if len(sentence) > len(longest_sentence):
    #         longest_sentence = sentence
    # longest_sentence = len(longest_sentence)
    # alpha_matrix = np.zeros((longest_sentence, prior_matrix.size))
    alpha = []
    for sentence, tag_line in zip(sentences, tags):
        alpha_sentence = np.zeros((len(trans_matrix), len(sentence)))   # Alpha is JxT
        for t_n, tag_n in enumerate(tag_line):
            x_n = sentence[t_n]   # Keep track of the word
            # Calculate prior
            if t_n == 0:
                alpha_sentence[:, 0] = emit_matrix[:, 0] * prior_matrix
                alpha_sentence[:, 0] /= alpha_sentence[:, 0].sum()
            # Calculate one column @ t_n in alpha
            else:
                # Get summation term
                for k in tag_dict.values():
                    summation_term = (alpha_sentence[:, t_n - 1] * trans_matrix[k]).sum()
                    result = emit_matrix[k][word_dict[x_n]] * summation_term
                    if alpha_sentence[:, t_n][k] != 0:  # Check that element assignment indexing tiles properly
                        print("Alpha values overlapping!")
                        break;
                    alpha_sentence[:, t_n][k] = result
                alpha_sentence[:, t_n] /= alpha_sentence[:, t_n].sum()
        alpha.append(alpha_sentence)
    print(alpha)
    return alpha




    # # Calculate all alpha_1 for each tag
    # alpha_prior = emit_matrix[:, 0] * prior_matrix
    # alpha_prior /= alpha_prior.sum(axis=0, keepdims=True)
    # alpha_previous = alpha_prior
    # alpha_matrix = np.zeros((len(emit_matrix[0]) - 1, len(emit_matrix)))
    # # Calculate the summation term in alpha for t > 1
    # for x_n in range(len(emit_matrix[0]) - 1):
    #     for tag_n in range(len(trans_matrix)):
    #         alpha_matrix[tag_n][x_n] = emit_matrix[tag_n][x_n + 1] * (alpha_previous * trans_matrix[tag_n]).sum()
    #     alpha_matrix[:, x_n] /= alpha_matrix[:, x_n].sum()
    #     alpha_previous = alpha_matrix[:, x_n]
    # print(alpha_matrix)
    return