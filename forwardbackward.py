import numpy as np


# IMPORTANT: The alpha is not a matrix, but a 3D TENSOR, JxMxD
def forward(emit_matrix, trans_matrix, prior_matrix, sentences, tags, word_dict, tag_dict):
    """ Does the forward computation, which creates the alpha matrix

    :param emit_matrix:
    :param trans_matrix:
    :param prior_matrix:
    :param sentences:
    :param tags:
    :param word_dict:
    :param tag_dict:
    :return:
    """
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
                        break
                    alpha_sentence[:, t_n][k] = result
                alpha_sentence[:, t_n] /= alpha_sentence[:, t_n].sum()
        alpha.append(alpha_sentence)
    print(alpha)
    return alpha


def backward(emit_matrix, trans_matrix, prior_matrix, sentences, tags, word_dict, tag_dict):
    """ Does the forward computation, which creates the alpha matrix

    :param emit_matrix:
    :param trans_matrix:
    :param prior_matrix:
    :param sentences:
    :param tags:
    :param word_dict:
    :param tag_dict:
    :return:
    """
    beta = []
    for sentence, tag_line in zip(sentences, tags):
        beta_sentence = np.zeros((len(trans_matrix), len(sentence)))    # Beta is JxT
        for t_n, tag in zip(reversed(range(len(tag_line))), tag_line):
            if t_n == len(tag_line) - 1:    # Every state can be an ending state, beta@T = 1 for all states
                beta_sentence[:, len(tag_line) - 1] = 1
            else:   # Summation term
                x_n = sentence[t_n + 1]  # Keep track of word
                beta_sentence[:, t_n] = (trans_matrix * beta_sentence[:, t_n + 1] * emit_matrix[:,  word_dict[x_n]]).sum(axis=-1, keepdims=True).flatten()
            beta_sentence[:, t_n] /= beta_sentence[:, t_n].sum(axis=0, keepdims=True)
        beta.append(beta_sentence)
    print(beta)
    return
