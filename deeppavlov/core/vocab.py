from deeppavlov.core.components import Component
from deeppavlov.core.registrable import Registrable
from collections import Counter, defaultdict
import numpy as np
from overrides import overrides

import logging

logger = logging.getLogger(__name__)


class Vocabulary:
    def __init__(self, tokens=None, special_tokens=tuple(), dict_file_path=None):
        if tokens is None and dict_file_path is not None:
            tokens = self.load(dict_file_path)
        self._t2i = dict()
        # We set default ind to position of <UNK> in SPECIAL_TOKENS
        # because the tokens will be added to dict in the same order as
        # in special_tokens
        default_ind = 0
        self._t2i = defaultdict(lambda: default_ind)
        self._i2t = dict()
        self.frequencies = Counter()

        self.counter = 0
        for token in special_tokens:
            self._t2i[token] = self.counter
            self.frequencies[token] += 0
            self._i2t[self.counter] = token
            self.counter += 1
        if tokens is not None:
            self.update_dict(tokens)

    def update_dict(self, tokens):
        for token in tokens:
            if not isinstance(token, str):
                self.update_dict(token)
            else:
                if token not in self._t2i:
                    self._t2i[token] = self.counter
                    self._i2t[self.counter] = token
                    self.counter += 1
                self.frequencies[token] += 1

    def idx2tok(self, idx):
        return self._i2t[idx]

    def idxs2toks(self, idxs, filter_paddings=False):
        toks = []
        for idx in idxs:
            if not filter_paddings or idx != self.tok2idx('<PAD>'):
                toks.append(self._i2t[idx])
        return toks

    def process(self, tokens):
        if not isinstance(tokens, str):
            return [self.process(token) for token in tokens]
        else:
            return self.tok2idx(tokens)

    def tok2idx(self, tok):
        return self._t2i[tok]

    def toks2idxs(self, toks):
        return [self._t2i[tok] for tok in toks]

    def batch_toks2batch_idxs(self, b_toks):
        max_len = max(len(toks) for toks in b_toks)
        # Create array filled with paddings
        batch = np.ones([len(b_toks), max_len]) * self.tok2idx('<PAD>')
        for n, tokens in enumerate(b_toks):
            idxs = self.toks2idxs(tokens)
            batch[n, :len(idxs)] = idxs
        return batch

    def batch_idxs2batch_toks(self, b_idxs, filter_paddings=False):
        return [self.idxs2toks(idxs, filter_paddings) for idxs in b_idxs]

    def is_pad(self, x_t):
        assert type(x_t) == np.ndarray
        return x_t == self.tok2idx('<PAD>')

    def __getitem__(self, key):
        return self._t2i[key]

    def __len__(self):
        return self.counter

    def __contains__(self, item):
        return item in self._t2i

    def load(self, dict_file_path):
        tokens = list()
        with open(dict_file_path) as f:
            for line in f:
                if len(line) > 0:
                    tokens.append(line)
        return tokens

    def save(self, path):
        with open(path, "w+") as f:
            for token in self._t2i.keys():
                f.write("%s\n" % token)


@Registrable.register("vocab")
class VocabComponent(Component):
    def __init__(self, config):
        super().__init__(config)
        self.local_input_names = ['tokens']
        self.local_output_names = ['idxs']
        self.vocab = Vocabulary()

    @overrides
    def forward(self, smem, add_local_mem=False):
        if len(self.inputs) > 0 and len(self.outputs) > 0:
            samples = self.get_input("tokens", smem)
            result = self.vocab.process(samples)
            self.set_output("idxs", result, smem)

    @overrides
    def train(self, smem, add_local_mem=False):
        tokens = self.get_input("tokens", smem)
        self.vocab.update_dict(tokens)

    @overrides
    def save(self):
        if "save_to" in self.config:
            path = self.config["save_to"]
            import os
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.vocab.save(path)

    @overrides
    def load(self):
        if "load" in self.config:
            path = self.config["load"]
            self.vocab.update_dict(self.vocab.load(path))

    @overrides
    def setup(self, components={}):
        super().setup(components)
        self.load()


@Registrable.register("bow")
class BowComponent(VocabComponent):
    def __init__(self, config):
        super().__init__(config)
        self.local_input_names = ['utterance']
        self.local_output_names = ['bow']

    @overrides
    def forward(self, smem, add_local_mem=False):
        if len(self.inputs) > 0 and len(self.outputs) > 0:
            utterance = self.get_input("utterance", smem)
            bow = np.zeros([len(self.vocab)], dtype=np.int32)
            for word in utterance.split(' '):
                if word in self.vocab:
                    idx = self.vocab.index(word)
                    bow[idx] += 1
            self.set_output("bow", bow, smem)