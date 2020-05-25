import random
import numpy as np
#from nltk.corpus import brown
#from nltk.corpus import stopwords


# sentences = [
#     ["this", "is", "red", "ball", "falling"],
#     ["this", "is", "green", "ball", "falling"],
#     ["this", "is", "blue", "ball", "falling"]]

#sentences = brown.sents()
#stops = set(stopwords.words('english'))

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

ITERATIONS = 200000
PRINT_LOSS_EVERY = 1000
LAMBDA = 0.005
K = 30
BATCH_SIZE = 100
WIDTH = 64
WINDOW_SIZE = 10
HEIGHT = len(word_to_index)

matrix_v = (np.random.rand(HEIGHT, WIDTH) - 0.5)
matrix_u = (np.random.rand(HEIGHT, WIDTH) - 0.5)

def debug(*args):
    # print(*args)
    pass

debug(f'Sentences: {n_sentences}')
debug(f'Tokens: {len(all_tokens)}')
debug(f'Unique words: {HEIGHT}')




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
            if an_outside_word in test_words:
                debug("OUT", an_outside_word, "for", a_word)
                debug(" sigma ", sigma_u_o_dot_v_c)
                # print(" grad_v ", grad_v[a_word_index])
                # print(" grad_u ", grad_u[an_outside_word_index])
                # print(" v_c ", v_c)

        neg_sample_indices = []
        while True:
            random_word = all_tokens[random.randint(0, len(all_tokens) - 1)]
            if random_word in test_words:
                debug("NEG", random_word, "for", a_word)
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
            debug(a_word, v_c)

        for index, k in enumerate(neg_sample_indices):
            grad_u[k] += (1 - sigma_minus_u_k_dot_v_c[index]) * v_c
            # if a_word in test_words:
            #     print("grad_u[k]", grad_u[k])

    return loss/BATCH_SIZE, grad_v/BATCH_SIZE, grad_u/BATCH_SIZE
    # return loss, grad_v, grad_u



def e(v, w):
    return v[word_to_index[w]]


def evaluate(v, u):
    red = e(v, "red")
    # print(red_)
    green = e(v, "green")
    # print(green_)
    blue = e(v, "blue")
    # print(blue_)
    print(np.dot(red, green))
    print(np.dot(red, blue))
    print(np.dot(green, blue))



def most_similar(word):
    embedding = matrix_v[word_to_index[word]]
    # tokens_and_similarities = [(tokens[i], np.dot(matrix_v[i]-embedding, matrix_v[i]-embedding)) for i in range(len(tokens))]
    tokens_and_similarities = [(tokens[i], np.dot(matrix_v[i], embedding)) for i in range(len(tokens))]
    # tokens_and_similarities.sort(key=lambda t: t[1])
    tokens_and_similarities.sort(key=lambda t: -t[1])
    return tokens_and_similarities[:10]


def evaluate_similar():
    print(most_similar("red"))
    print(most_similar("green"))
    print(most_similar("blue"))
    print(most_similar("go"))
    print(most_similar("walk"))
    print(most_similar("run"))


def train_model(v, u):
    for i in range(ITERATIONS):
        loss, grad_v, grad_u = loss_and_gradients(v, u)
        if i % PRINT_LOSS_EVERY == 0:
            print(i, loss)
            # evaluate(v, u)
        v -= grad_v*LAMBDA
        u -= grad_u*LAMBDA


train_model(matrix_v, matrix_u)
evaluate_similar()
evaluate(matrix_v, matrix_u)


from sklearn.manifold import TSNE
from matplotlib import pylab
pylab.figure(figsize=(15, 15))

def plot():
    print("Running T-SNE")
    tsne = TSNE(perplexity=5, n_components=2, init='pca', n_iter=5000)
    # get the T-SNE manifold
    two_d_embeddings = tsne.fit_transform(matrix_v)
    print(two_d_embeddings)
    selected_words = ['red', 'green', 'blue']
    print("Plotting")
    # plot all the embeddings and their corresponding words
    for label, i in word_to_index.items():
        x, y = two_d_embeddings[i, :]
        pylab.scatter(x, y, c='darkgray')
        if label in selected_words:
            print(x, y)
            pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
                           ha='right', va='bottom', fontsize=10)
    print("Showing")
    pylab.show()
    print("Done")


#plot()
