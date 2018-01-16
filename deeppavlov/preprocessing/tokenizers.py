from deeppavlov.core.components import Component
from deeppavlov.core.registrable import Registrable
import spacy
import re

import logging
import nltk

logger = logging.getLogger(__name__)


def nltk_tokenizer(batch, tokenizer="wordpunct_tokenize"):
        tokenized_batch = []

        tokenizer_ = getattr(nltk.tokenize, tokenizer, None)
        if callable(tokenizer_):
            if type(batch) == str:
                tokenized_batch = " ".join(tokenizer_(batch))
            else:
                # list of str
                for text in batch:
                    tokenized_batch.append(" ".join(tokenizer_(text)))
            return tokenized_batch
        else:
            raise AttributeError("Tokenizer %s is not defined in nltk.tokenizer" % tokenizer)


def char_tokenizer(tokens):
    ch_tokens = []
    for token in tokens:
        if isinstance(token, str):
            ch_tokens.append([ch for ch in token])
        else:
            ch_tokens.append([[ch for ch in t] for t in token])
    return ch_tokens


NLP = spacy.load('en')


def spacy_tokenizer(x):

    def _tokenize(text, **kwargs):
        """Tokenize with spacy, placing service words as individual tokens."""
        return [t.text for t in NLP.tokenizer(text)]

    def _detokenize(tokens):
        """
        Detokenizing a text undoes the tokenizing operation, restoring
        punctuation and spaces to the places that people expect them to be.
        Ideally, `detokenize(tokenize(text))` should be identical to `text`,
        except for line breaks.
        """
        text = ' '.join(tokens)
        step0 = text.replace('. . .',  '...')
        step1 = step0.replace("`` ", '"').replace(" ''", '"')
        step2 = step1.replace(" ( ", " (").replace(" ) ", ") ")
        step3 = re.sub(r' ([.,:;?!%]+)([ \'"`])', r"\1\2", step2)
        step4 = re.sub(r' ([.,:;?!%]+)$', r"\1", step3)
        step5 = step4.replace(" '", "'").replace(" n't", "n't")\
            .replace(" nt", "nt").replace("can not", "cannot")
        step6 = step5.replace(" ` ", " '")
        return step6.strip()

    if isinstance(x, str):
        return _tokenize(x)
    if isinstance(x, list):
        return _detokenize(x)
    raise TypeError("SpacyTokenize.infer() not implemented for `{}`"\
                    .format(type(x)))


@Registrable.register("tokenizer.chars")
class CharTokenizerComponent(Component):
    def __init__(self, config):
        super().__init__(config)
        self.local_input_names = ['tokens']
        self.local_output_names = ['ch_tokens']

    def forward(self, smem, add_local_mem=False):
        tokens = self.get_input('tokens', smem)
        ch_tokens = char_tokenizer(tokens)
        self.set_output('ch_tokens', ch_tokens, smem)


@Registrable.register("tokenizer.spacy")
class SpacyTokenizerComponent(Component):
    def __init__(self, config):
        super().__init__(config)
        self.local_input_names = ['text']
        self.local_output_names = ['tokens']

    def forward(self, shared_mem, add_local_mem=False):
        text = self.get_input('text', shared_mem)
        tokens = spacy_tokenizer(text)
        self.set_output('tokens', tokens, shared_mem)


@Registrable.register("tokenizer.nltk")
class NLTKTokenizerComponent(Component):
    def __init__(self, config):
        super().__init__(config)
        self.local_input_names = ['text']
        self.local_output_names = ['tokens']

    def forward(self, shared_mem, add_local_mem=False):
        text = self.get_input('text', shared_mem)
        tokens = nltk_tokenizer(text)
        self.set_output('tokens', tokens, shared_mem)