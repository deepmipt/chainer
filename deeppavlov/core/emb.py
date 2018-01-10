from deeppavlov.core.components import Component
from deeppavlov.core.registrable import Registrable
from overrides import overrides
import logging
from gensim.models import word2vec
import numpy as np
import os

logger = logging.getLogger(__name__)


@Registrable.register("w2v")
class W2VEmbComponent(Component):
    def __init__(self, config):
        super().__init__(config)
        self.local_input_names = ['utterance']
        self.local_output_names = ['emb']
        self.corpus = self.config["corpus"] if "corpus" in self.config else None
        self.dim = self.config["dim"] if "dim" in self.config else 300
        self.emb = UtteranceEmbed(self.dim)

    @overrides
    def forward(self, smem, add_local_mem=False):
        if len(self.inputs) > 0 and len(self.outputs) > 0:
            utterance = self.get_input("utterance", smem)
            result = self.emb.infer(utterance)
            self.set_output("emb", result, smem)

    @overrides
    def train(self, smem, add_local_mem=False):
        self.emb.train(self.corpus)

    @overrides
    def save(self):
        if "save_to" in self.config:
            path = self.config["save_to"]
            import os
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.emb.save(path)

    @overrides
    def load(self):
        if "load" in self.config:
            path = self.config["load"]
            self.emb.load(path)

    @overrides
    def setup(self, components={}):
        super().setup(components)
        self.load()


class UtteranceEmbed():
    def __init__(self, dim=300):
        self.dim = dim
        self.model = None

    def _encode(self, utterance):
        embs = [self.model[word] for word in utterance.split(' ') if word and word in self.model]
        # average of embeddings
        if len(embs):
            return np.mean(embs, axis=0)
        else:
            return np.zeros([self.dim], np.float32)

    def train(self, corpus_path):
        sentences = word2vec.Text8Corpus(corpus_path)
        print(':: creating new word2vec model')
        model = word2vec.Word2Vec(sentences, size=self.dim)
        self.model = model
        return model

    def infer(self, utterance):
        return self._encode(utterance)

    def load(self, path):
        print(':: model loaded from path %s' % path)
        self.model = word2vec.Word2Vec.load(path)

    def save(self, path):
        self.model.save(path)
        print(':: model saved to path %s' % path)
