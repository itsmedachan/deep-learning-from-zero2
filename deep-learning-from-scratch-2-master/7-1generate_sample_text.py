# coding: utf-8
import sys
sys.path.append('..')
from common.np import *
from ch07.rnnlm_gen import BetterRnnlmGen
from dataset import ptb


corpus, word_to_id, id_to_word = ptb.load_data('train')
vocab_size = len(word_to_id)
corpus_size = len(corpus)


model = BetterRnnlmGen()
model.load_params('/Users/yuri/sophia/deep-learning-from-zero2/deep-learning-from-scratch-2-master/ch06/BetterRnnlm.pkl')

# # start文字とskip文字の設定
# start_word = 'you'
# start_id = word_to_id[start_word]
skip_words = ['N', '<unk>', '$']
skip_ids = [word_to_id[w] for w in skip_words]
# # 文章生成
# word_ids = model.generate(start_id, skip_ids)
# txt = ' '.join([id_to_word[i] for i in word_ids])
# txt = txt.replace(' <eos>', '.\n')

# print(txt)


# model.reset_state()

start_words = 'beauty is'
start_ids = [word_to_id[w] for w in start_words.split(' ')]

for x in start_ids[:-1]:
    x = np.array(x).reshape(1, 1)
    model.predict(x)

word_ids = model.generate(start_ids[-1], skip_ids)
word_ids = start_ids[:-1] + word_ids
txt = ' '.join([id_to_word[i] for i in word_ids])
txt = txt.replace(' <eos>', '.\n')
print('-' * 50)
print(txt)
# beauty is an adjustment to any investment.
#  the s&l 30-share index averaged two cents.
#  friday 's volume of october was solid by a year ago.
#  some california oil & gas prices fell higher in the fourth quarter.
#  insurance prices rose away from a barometer of u.s. sellers and declining buying in singapore.
#  mark adams chief financial economist at the trade group said platinum again were sellers in the del bridge which served big blue-chip orders yesterday.
#  when the announcement seemed down to the u.s. push the purchased dollar came from a nervous report success as