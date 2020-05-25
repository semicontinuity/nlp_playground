import gensim
import numpy as np

def t():
  s = "Natural language processing is cool"
  words = s.split()
  test(words)

def test(words):
  for i in words:
    for j in words:
      if i != j:
        print (i + '_' + j, model.similarity(i, j), sep='\t')


model = gensim.models.KeyedVectors.load_word2vec_format('.model/GoogleNews-vectors-negative300.bin', binary=True)

#colors = ['red', 'green', 'blue', 'white', 'black', 'orange', 'yellow']
#vectors = {c: model.wv.word_vec(c) for c in colors}
#a = np.average(list(vectors.values()), axis=0)



#for c in colors:
#  print(c, vectors[c])
#  print(c, np.linalg.norm(vectors[c]),  np.linalg.norm(vectors[c] - a))

#test(colors)

print(model.wv.word_vec('dfgsdgsdfgdsfgsetrge'))
#z = (model.wv.word_vec('man') - model.wv.word_vec('woman')) - (model.wv.word_vec('king') - model.wv.word_vec('queen'))
#print(z)
#print(np.linalg.norm(z))