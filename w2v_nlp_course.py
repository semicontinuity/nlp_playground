import random
import numpy as np

def train(text):
    sentences = [text.split()]
    stops = {"the", "a", "in", "of", "at"}
    all_tokens = []
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

    ITERATIONS = 1000
    PRINT_LOSS_EVERY = 1
    LAMBDA = 0.01
    K = 15
    BATCH_SIZE = 32
    WIDTH = 64
    WINDOW_SIZE = 6
    HEIGHT = len(word_to_index)

    matrix_v = (np.random.rand(HEIGHT, WIDTH) - 0.5)
    matrix_u = (np.random.rand(HEIGHT, WIDTH) - 0.5)

    def debug(*args):
        # print(*args)
        pass

    # debug(f'Sentences: {n_sentences}')
    # debug(f'Tokens: {len(all_tokens)}')
    # debug(f'Unique words: {HEIGHT}')

    def get_random_sentence():
        return sentences[random.randint(0, n_sentences - 1)]

    def random_word_and_its_outside_words():
        random_sentence = get_random_sentence()
        random_offset = random.randint(0, len(random_sentence) - 1)
        window_size = random.randint(0, WINDOW_SIZE)
        window_offsets = range(
            max(0, random_offset - window_size),
            min(len(random_sentence), random_offset + window_size + 1)
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

            if a_word in stops: continue
            outside_words = [w for w in outside_words if w not in stops]

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

            neg_sample_indices = []
            while True:
                random_word = all_tokens[random.randint(0, len(all_tokens) - 1)]
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

            for index, k in enumerate(neg_sample_indices):
                grad_u[k] += (1 - sigma_minus_u_k_dot_v_c[index]) * v_c

        return loss / BATCH_SIZE, grad_v / BATCH_SIZE, grad_u / BATCH_SIZE
        # return loss, grad_v, grad_u

    def train_model(v, u):
        for i in range(ITERATIONS):
            loss, grad_v, grad_u = loss_and_gradients(v, u)
            if i % PRINT_LOSS_EVERY == 0:
                debug(i, loss)
                # evaluate(v, u)
            v -= grad_v * LAMBDA
            u -= grad_u * LAMBDA

    train_model(matrix_v, matrix_u)
    return {w: matrix_v[index] for w, index in word_to_index.items()}


d = train("this is red ball falling "
          "this is green ball falling "
          "this is blue ball falling "
          "this is orange ball falling "
          "this is cyan ball falling ")
print(d)
