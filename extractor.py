# from nltk.stem.wordnet import WordNetLemmatizer
import collections
from nltk.corpus import semcor
import numpy as np

from misc import *

class Context:
    ''' x_i: vector
        y_i: sense_id
        word_list: list of word as string, for debugging
    '''
    def __init__(self, vector, sense_id, word_list):
        self.vector = vector
        self.sense_id = int(sense_id)
        self.word_list = word_list
    def dump(self):
        print 'words: %s\n vector: %s, sense id: %d' % (str(self.word_list) ,self.vector, self.sense_id)
    def vec(self):
        return np.append(self.vector, [self.sense_id])

class ContextContainer:
    def __init__(self, text):
        self.text = text
        self.context_list = np.array([])
    def update(self, vector, sense_id, word_list):
        if ';' in sense_id:
            ids = sense_id.split(';')
        else:
            ids = [sense_id]
        for sid in ids:
            self.context_list = np.append(self.context_list,Context(vector,
                sid, word_list))

    def dump(self,n=1):
        # sample some instance and print to screen
        for i,c in enumerate(self.context_list):
            if i > n: break
            c.dump()

    def len(self):
        return len(self.context_list)

    def dump2file(self, filename):
        vector_list = map(lambda x: x.vec(), self.context_list)
        matrix = reduce(lambda x,y: np.vstack((x,y)), vector_list)
        _dir = os.path.dirname(filename)
        if not os.path.exists(_dir):
            os.makedirs(_dir)
        np.savetxt(filename,matrix,delimiter=',')

class ContextExtractor:
    def __init__(self, word_map, word2vec_model, pos2vec_model, dir_name):
        self.word_map = word_map
        self.word2vec_model = word2vec_model
        self.pos2vec_model = pos2vec_model

        self.context_map = {}
        self.WORD_VECTOR_LEN = 100
        self.dir = dir_name

    def go(self, index_file):
        tag_files = load_tag_fies(index_file)
        for t in tag_files:
            print 'scannig %s' % t
            self.__scan_corpus(t)

    def __scan_corpus(self, filename):
        # get sentence from semcor file
        sentences = semcor.xml(filename).findall('context/p/s')

        for sent in sentences:
            for wordform in sent.getchildren():
                lemma = wordform.get('lemma')
                sense_id = wordform.get('wnsn')
                text = wordform.text

                if lemma in self.word_map.keys():
                    self.__parse(text, lemma, sent, sense_id)

    def __parse(self, word, lemma, sent, sense_id):
        # from word to a matrix representation
        if lemma not in self.context_map.keys():
            self.context_map[lemma] = ContextContainer(word)
        vector,word_list = self.__sent2vec(word, sent)
        self.context_map[lemma].update(vector, sense_id, word_list)

    @staticmethod
    def __combine_bufs(last_words,next_words):
        return reduce(lambda x,y: np.append(x,y), np.append(last_words,
            next_words))

    def __word2vec(self, word, pos_tag):
        # TODO: incooperate POS tag informatoin
        return np.append(self.word2vec_model[word],self.pos2vec_model[pos_tag])

    def __sent2vec(self, word, sent):
        WINDOW_SIZE = 2
        last_words = collections.deque(maxlen=WINDOW_SIZE)
        next_words = collections.deque(maxlen=WINDOW_SIZE)
        word_list = []

        # set default to 0 vectors: the length of each word include word vector
        # and pos tag
        padding = np.zeros(self.WORD_VECTOR_LEN * 2)
        for i in range(WINDOW_SIZE):
            last_words.append(padding)
            next_words.append(padding)

        is_seen = False
        look_ahead = 0
        # pass the whole sentence again
        for wf in sent.getchildren():
            pos_tag = wf.get('pos')
            lemma = wf.get('lemma')
            word_list.append(wf.text)

            if wf.text == word:
                is_seen = True

            if wf.text in self.word2vec_model.wv.vocab and pos_tag in self.pos2vec_model.wv.vocab:
                # the key used in word2vec model is the original word, not the
                # lemma
                _vec = self.__word2vec(wf.text,pos_tag)
                if is_seen and look_ahead < WINDOW_SIZE:
                    next_words.append(_vec)
                    look_ahead += 1
                else:
                    last_words.append(_vec)
        assert is_seen == True
        return self.__combine_bufs(last_words,next_words), word_list

    def dump(self):
        for i,w in enumerate(self.context_map.keys()):
            if i>3: break
            print w
            self.context_map[w].dump()

    def dump2file(self):
        for w,c in self.context_map.items():
            filename = './dataset/%s/%s.txt' %(self.dir, w)
            print 'dumpping %s to %s' % (w,filename)
            c.dump2file(filename)
