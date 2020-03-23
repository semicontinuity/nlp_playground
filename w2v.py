import random
import numpy as np
from nltk.corpus import brown


sentences = brown.sents()
all_tokens = []
word_to_index = {}


def populate_data_structs():
    i = 0
    for s in sentences:
        for w in s:
            all_tokens.append(w)
            if w not in word_to_index:
                word_to_index[w] = i
                i += 1


populate_data_structs()

ITERATIONS = 1000000
PRINT_LOSS_EVERY = 1000
LAMBDA = 0.05
K = 30
BATCH_SIZE = 50
WIDTH = 32
WINDOW_SIZE = 5
HEIGHT = len(word_to_index)

matrix_v = (np.random.rand(HEIGHT, WIDTH) - 0.5)
matrix_u = (np.random.rand(HEIGHT, WIDTH) - 0.5)

print(f'Sentences: {len(sentences)}')
print(f'Tokens: {len(all_tokens)}')
print(f'Unique words: {HEIGHT}')


def random_word_and_its_outside_words():
    random_sentence = sentences[random.randint(0, len(sentences) - 1)]
    random_offset = random.randint(0, len(random_sentence) - 1)
    window_offsets = range(
        max(0, random_offset - WINDOW_SIZE),
        min(len(random_sentence), random_offset + WINDOW_SIZE + 1)
    )
    return random_sentence[random_offset], [random_sentence[i] for i in window_offsets if i != random_offset]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def loss_and_gradients(v, u):
    loss = 0.0
    grad_v = np.zeros_like(v)
    grad_u = np.zeros_like(u)

    for _ in range(BATCH_SIZE):
        a_word, outside_words = random_word_and_its_outside_words()
        a_word_index = word_to_index[a_word]
        outside_words_indices = set((word_to_index[w] for w in outside_words))
        v_c = v[a_word_index]

        test_words = ["red", "green", "blue"]
        for an_outside_word in outside_words:
            an_outside_word_index = word_to_index[an_outside_word]
            u_o = u[an_outside_word_index]
            u_o_dot_v_c = u_o.dot(v_c)
            sigma_u_o_dot_v_c = sigmoid(u_o_dot_v_c)
            loss += - (np.log(sigma_u_o_dot_v_c))
            grad_v[a_word_index] += (sigma_u_o_dot_v_c - 1) * u_o
            grad_u[an_outside_word_index] += (sigma_u_o_dot_v_c - 1) * v_c
            if an_outside_word in test_words:
                print("OUT", an_outside_word, "for", a_word)
                print(" sigma ", sigma_u_o_dot_v_c)
                # print(" grad_v ", grad_v[a_word_index])
                # print(" grad_u ", grad_u[an_outside_word_index])
                # print(" v_c ", v_c)

        neg_sample_indices = []
        while True:
            random_word = all_tokens[random.randint(0, len(all_tokens) - 1)]
            if random_word in test_words:
                print("NEG", random_word, "for", a_word)
            random_word_index = word_to_index[random_word]
            if random_word_index in outside_words_indices:
                continue
            neg_sample_indices.append(random_word_index)
            if len(neg_sample_indices) == K:
                break

        u_k = u[neg_sample_indices]
        u_k_dot_v_c = u_k.dot(v_c)
        sigma_minus_u_k_dot_v_c = sigmoid(- u_k_dot_v_c)
        loss += - (np.sum(np.log(sigma_minus_u_k_dot_v_c)))
        grad_v[a_word_index] += np.sum((1 - sigma_minus_u_k_dot_v_c)[:, np.newaxis] * u_k, axis=0)

        if a_word in test_words:
            # print(a_word, outside_words)
            print(a_word, v_c)

        for index, k in enumerate(neg_sample_indices):
            grad_u[k] += (1 - sigma_minus_u_k_dot_v_c[index]) * v_c
            # if a_word in test_words:
            #     print("grad_u[k]", grad_u[k])

    return loss/BATCH_SIZE, grad_v/BATCH_SIZE, grad_u/BATCH_SIZE
    # return loss, grad_v, grad_u


def evaluate(v, u):
    red_ = v[word_to_index["red"]]
    # print(red_)
    green_ = v[word_to_index["green"]]
    # print(green_)
    blue_ = v[word_to_index["blue"]]
    # print(blue_)
    print(np.dot(red_, green_))
    print(np.dot(red_, blue_))
    print(np.dot(green_, blue_))


def train(v, u):
    for i in range(ITERATIONS):
        loss, grad_v, grad_u = loss_and_gradients(v, u)
        if i % PRINT_LOSS_EVERY == 0:
            print(i, loss)
            # evaluate(v, u)
        v -= grad_v*LAMBDA
        u -= grad_u*LAMBDA


train(matrix_v, matrix_u)
evaluate(matrix_v, matrix_u)
