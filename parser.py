#!/usr/bin/python

import nltk
import os
nltk.data.path.append(os.getcwd() + '/dataset/')
from nltk.corpus import semcor

from gensim.models import Word2Vec

from extractor import *
from misc import *

from collections import defaultdict

def scan_corpus(index_file, f_parse):
    tag_files = load_tag_fies(index_file)
    for filename in tag_files:
        print 'parsing %s...' % filename
        sentences = semcor.xml(filename).findall('context/p/s')
        f_parse(sentences)

class ModelTrainer:
    def __init__(self, model_type):
        self._model = Word2Vec(size=100, window=5, min_count=5, workers=2)
        self.model_type = model_type
        self._trained = False
    def parse(self, sentences):
        sent_str = map(lambda x:self.__semcor_sent2str(x), sentences)
        assert len(sent_str) > 0

        if not self._trained:
            self.model.build_vocab(sent_str)
            self._trained = True
        else:
            self.model.build_vocab(sent_str,update=True)
        self.model.train(sent_str,total_examples=self.model.corpus_count,epochs=self.model.iter)

    def __semcor_sent2str(self, sent):
        '''  sent is the wordform structure in semcor corpus
        ''' 
        # key is original word, not lemma
        if self.model_type == 'word':
            return map(lambda x: x.text, sent)
        if self.model_type == 'pos':
            pos_list = map(lambda x: x.get('pos'), sent.getchildren())
            return filter(lambda x: x !=None, pos_list)

    @property
    def model(self):
        return self._model

class CorpusParser:
    def __init__(self, index_file, force_update=False):
        # try to load from pickle file first
        pkl_dir = './pickle'
        pkl_file = '%s/%s_word_map.pkl' % (pkl_dir,
                os.path.basename(index_file).split('.')[0])
        loaded = load(pkl_file)

        if loaded != None and force_update==False:
            self.word_map = loaded
        else:
            self.word_map = {}
            # have to parse corpose
            tag_files = load_tag_fies(index_file)
            for f in tag_files:
                self.parse(f)

            if not os.path.exists(pkl_dir):
                os.makedirs(pkl_dir)

            save(self.word_map, pkl_file)

    def parse(self,filename):
        # get sentence from semcor file
        print 'parsing %s...' % filename
        sentences = semcor.xml(filename).findall('context/p/s')

        for sent in sentences:
            for wordform in sent.getchildren():
                lemma = wordform.get('lemma')
                sense_id = wordform.get('wnsn')

                if sense_id == None or not sense_id.isdigit():
                    continue

                if lemma not in self.word_map.keys():
                    self.word_map[lemma] = defaultdict(int)
                self.word_map[lemma][sense_id] += 1

def filter_word_map(word_map, min_sense_appr, min_size):
    filtered_map = {}
    for word, senses in word_map.items():
        if len(senses.keys()) <= 1:
            continue
        if sum(senses.values()) < min_size:
            continue
        if min(senses.values()) < min_sense_appr:
            continue
        filtered_map[word] = senses
    return filtered_map

def load_model(index_file, model_type, force_update=False):
    model_dir = 'model'
    if model_type == 'word':
        model_file = './%s/%s_w2v.embedding' % (model_dir, 
                    os.path.basename(index_file).split('.')[0])
    else:
        model_file = './%s/%s_%s.embedding' % (model_dir, 
                    os.path.basename(index_file).split('.')[0], model_type)

    if os.path.isfile(model_file) and not force_update:
        print 'Load from %s' % model_file
        model = Word2Vec.load(model_file)
    else:
        trainer = ModelTrainer(model_type)
        scan_corpus(index_file, trainer.parse)
        model = trainer.model
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model.save(model_file)
        print 'Saved model to %s' % model_file
    return model

if __name__ == '__main__':
    index_file = './dataset/semcor_tagfiles_full.txt'
    # index_file = './dataset/brown1_tagfiles.txt'
    
    word2vec_model = load_model(index_file, 'word', force_update=False)
    pos2vec_model = load_model(index_file, 'pos', force_update=False)

    word_map = CorpusParser(index_file,force_update=False).word_map

    MIN_SENSE_APPR = 20
    MIN_TOTAL_SIZE = 200
    ambiguous_words = filter_word_map(word_map, MIN_SENSE_APPR, MIN_TOTAL_SIZE)
    print '%d ambiguous words' % len(ambiguous_words.keys())
    for i,w in enumerate(ambiguous_words.keys()):
        # if i > 3: break
        print w,ambiguous_words[w]

    ce = ContextExtractor(ambiguous_words, word2vec_model, pos2vec_model)
    ce.go(index_file)
    ce.dump2file()
    # ce.dump()
