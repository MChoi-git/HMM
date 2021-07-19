import numpy as np


# IMPORTANT: The alpha is not a matrix, but a 3D TENSOR, JxMxD
def forward(emit_matrix, trans_matrix, prior_matrix, sentences, tags, word_dict, tag_dict):
    """ Does the forward computation, which creates a collection of alpha matrices

    :param emit_matrix: Emission probability matrix
    :param trans_matrix: Transition probability matrix
    :param prior_matrix: Prior matrix
    :param sentences: Array of all sentences from dataset
    :param tags: Array of all tags from dataset
    :param word_dict: Dictionary where keys=index, values=words (unique)
    :param tag_dict: Dictionary where keys=index, values=tags (unique)
    :return: Collection of alpha matrices
    """
    alpha = []
    for sentence, tag_line in zip(sentences, tags):
        alpha_sentence = np.zeros((len(trans_matrix), len(sentence)))   # Alpha is JxT
        # Calculate one alpha matrix per sentence
        for t_n, tag_n in enumerate(tag_line):
            x_n = sentence[t_n]
            # Calculate prior
            if t_n == 0:
                alpha_sentence[:, 0] = emit_matrix[:, word_dict[x_n]] * prior_matrix
                alpha_sentence[:, 0] /= alpha_sentence[:, 0].sum()
            # Calculate rest of alpha
            else:
                x1 = emit_matrix[:, word_dict[x_n]]
                x2 = trans_matrix.T
                x5 = trans_matrix
                x3 = alpha_sentence[:, t_n - 1]
                x4 = trans_matrix.T * alpha_sentence[:, t_n - 1]
                alpha_sentence[:, t_n] = emit_matrix[:, word_dict[x_n]] * (trans_matrix.T * alpha_sentence[:, t_n - 1]).sum(axis=-1, keepdims=True).flatten()
            if t_n != len(tag_line) - 1:
                alpha_sentence[:, t_n] /= alpha_sentence[:, t_n].sum()
        alpha.append(alpha_sentence)
    return alpha


def backward(emit_matrix, trans_matrix, prior_matrix, sentences, tags, word_dict, tag_dict):
    """ Does the forward computation, which creates a collection of beta matrices

    :param emit_matrix: Emission probability matrix
    :param trans_matrix: Transition probability matrix
    :param prior_matrix: Prior matrix
    :param sentences: Array of all sentences from dataset
    :param tags: Array of all tags from dataset
    :param word_dict: Dictionary where keys=index, values=words (unique)
    :param tag_dict: Dictionary where keys=index, values=tags (unique)
    :return: Collection of beta matrices
    """
    beta = []
    for sentence, tag_line in zip(sentences, tags):
        beta_sentence = np.zeros((len(trans_matrix), len(sentence)))    # Beta is JxT
        # Calculate one beta matrix per sentence
        for t_n, tag in zip(reversed(range(len(tag_line))), tag_line):
            # Set ending state
            if t_n == len(tag_line) - 1:
                beta_sentence[:, len(tag_line) - 1] = 1
            # Calculate rest of beta
            else:
                x_n = sentence[t_n + 1]
                beta_sentence[:, t_n] = (trans_matrix * beta_sentence[:, t_n + 1] * emit_matrix[:,  word_dict[x_n]]).sum(axis=-1, keepdims=True).flatten()
            beta_sentence[:, t_n] /= beta_sentence[:, t_n].sum(axis=0, keepdims=True)
        beta.append(beta_sentence)
    return beta


def get_conditional_probabilities(alpha, beta):
    cond_probs = [ex_a * ex_b for ex_a, ex_b in zip(alpha, beta)]
    # for example in cond_probs:
    #     example /= example.sum(axis=0, keepdims=True)

    return cond_probs