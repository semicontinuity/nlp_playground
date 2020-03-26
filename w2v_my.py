from typing import List, Set, Dict, Tuple, Optional
import random
import numpy as np

sentences = [
    ["this", "is", "red", "ball", "falling" ],
    ["this", "is", "green", "ball", "falling"],
    ["this", "is", "blue", "ball", "falling"]]
all_tokens : List[str] = []
word_to_index = {}
tokens = []
MAX_SENTENCES = 20000


def populate_data_structs():
    i = 0
    n = 0
    for s in sentences:
        if n >= MAX_SENTENCES:
            break

        for w in s:
            all_tokens.append(w)
            if w not in word_to_index:
                tokens.append(w)
                word_to_index[w] = i
                i += 1

        n += 1


populate_data_structs()
n_sentences = min(MAX_SENTENCES, len(sentences))

ITERATIONS = 20000
PRINT_LOSS_EVERY = 1000

K = 5
ALPHA = 1
BETA = 0.2
GAMMA = 2   # Bigger repellence coefficient works fine (similar words become more similar), but the system diverges => need to normalize.
LAMBDA = 0.001


BATCH_SIZE = 10
WIDTH = 300
WINDOW_SIZE = 2
HEIGHT = len(word_to_index)

matrix_v = (np.random.rand(HEIGHT, WIDTH) - 0.5)

print(f'Sentences: {n_sentences}')
print(f'Tokens: {len(all_tokens)}')
print(f'Unique words: {HEIGHT}')


def get_random_sentence() -> List[str]:
    return sentences[random.randint(0, n_sentences - 1)]


def random_word_and_its_outside_words() -> Tuple[str, List[str]]:
    random_sentence : List[str] = get_random_sentence()
    random_offset : int = random.randint(0, len(random_sentence) - 1)
    window_offsets = range(
        max(0, random_offset - WINDOW_SIZE),
        min(len(random_sentence), random_offset + WINDOW_SIZE + 1)
    )
    return random_sentence[random_offset], [random_sentence[i] for i in window_offsets if i != random_offset]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def normalize(v):
    norm=np.linalg.norm(v, ord=2)
    return v/norm


def debug(*args):
    # print(*args)
    pass


def gradient(v):
    grad_v = np.zeros_like(v)

    for _ in range(BATCH_SIZE):
        debug("-----------------------------------------------------------------------------------------")
        a_word, outside_words = random_word_and_its_outside_words()

        a_word_index = word_to_index[a_word]
        outside_words_indices = set((word_to_index[w] for w in outside_words))
        outside_words_indices_list = list(outside_words_indices)

        debug(a_word)
        word_vector = v[a_word_index]
        debug(word_vector)

        outside_word_vectors = v[outside_words_indices_list]
        context_vector = np.mean(outside_word_vectors, axis=0)
        debug(outside_words)
        debug(context_vector)

        corr = np.dot(word_vector, context_vector)
        debug("CORR", corr)
        context_sigma_correlation = sigmoid(corr)
        debug("SIGM", context_sigma_correlation)

        # word_vector is attracted to context_vector
        # (perhaps, can be attracted to each of outside_words with the same success?)
        #############################################################################
        grad_v[a_word_index] += (1 - context_sigma_correlation) * (context_vector - word_vector)
        #############################################################################

        # not along diff, but at least in the direction of more correlation with context_vector
        # grad_v[a_word_index] += (1 - context_sigma_correlation) * context_vector

        debug("GRAD", grad_v[a_word_index])

        neg_sample_indices = []
        while True:
            random_word : str = all_tokens[random.randint(0, len(all_tokens) - 1)]
            random_word_index = word_to_index[random_word]
            if random_word_index in outside_words_indices_list:
                continue
            debug(random_word)
            neg_sample_indices.append(random_word_index)
            if len(neg_sample_indices) == K:
                break

        for k in neg_sample_indices:
            negative_word_vector = v[k]
            negative_sigma_correlation = sigmoid(np.dot(negative_word_vector, context_vector))
            debug(k, all_tokens[k], negative_sigma_correlation)

            # negative sample vector is repelled from context_vector
            #############################################################################
            # not strictly along diff of vectors, but at least in the direction of less correlation with context_vector
            # this works better, but, perhaps, only cause of missing system-wide normalization.
            grad_v[k] += GAMMA * negative_sigma_correlation * (-context_vector)
            #############################################################################

            # let's also repel word vector form negative sample vector (at least in the direction of less correlation with negative_word_vector)
            grad_v[a_word_index] += BETA * negative_sigma_correlation * (-negative_word_vector)


            # repel_direction = normalize(negative_word_vector - context_vector)
            # grad_v[k] += GAMMA * diff / l2norm

            # diff = negative_word_vector - context_vector
            # l2norm = np.linalg.norm(diff, ord=2)
            # grad_v[k] += GAMMA * negative_sigma_correlation * diff / l2norm

            debug(grad_v[k])

    return grad_v/BATCH_SIZE/(1+K)
    # return loss, grad_v, grad_u


def evaluate_dot_product(v):
    red_ = v[word_to_index["red"]]
    # print(red_)
    green_ = v[word_to_index["green"]]
    # print(green_)
    blue_ = v[word_to_index["blue"]]
    # print(blue_)
    print(np.dot(red_, green_))
    print(np.dot(red_, blue_))
    print(np.dot(green_, blue_))


def similarity(word1, word2):
    return np.dot(matrix_v[word_to_index[word1]], matrix_v[word_to_index[word2]])


def most_similar(word):
    embedding = matrix_v[word_to_index[word]]
    # tokens_and_similarities = [(tokens[i], np.dot(embedding, matrix_v[i])) for i in range(len(tokens))]
    tokens_and_similarities = [(tokens[i], np.dot(matrix_v[i]-embedding, matrix_v[i]-embedding)) for i in range(len(tokens))]
    tokens_and_similarities.sort(key=lambda t: t[1])
    return tokens_and_similarities[:10]


def evaluate_similar():
    print('most_similar("red")=', most_similar("red"))
    print('most_similar("green")=', most_similar("green"))
    print('most_similar("blue")=', most_similar("blue"))

    print('most_similar("this")=', most_similar("this"))
    print('most_similar("ball")=', most_similar("ball"))
    print('most_similar("is")=', most_similar("is"))


def train(v):
    for i in range(ITERATIONS):
        grad_v = gradient(v)
        if i % PRINT_LOSS_EVERY == 0:
            print(i)
        # evaluate(v, u)
        v -= LAMBDA * grad_v


train(matrix_v)
evaluate_similar()
