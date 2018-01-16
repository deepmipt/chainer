"""
Copyright 2017 Neural Networks and Deep Learning lab, MIPT

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import json
import logging
from overrides import overrides
import os
import random
import itertools
from pathlib import Path
import copy
from sklearn.model_selection import train_test_split
import numpy as np

from deeppavlov.data.utils import is_done, mark_done, download_untar, download

from deeppavlov.core.data import DatasetReader, DatasetProvider

from deeppavlov.core.registrable import Registrable

from deeppavlov.data.utils import PREPROCESSORS


logger = logging.getLogger(__name__)


class DSTC2Reader(DatasetReader):

    @staticmethod
    def build(data_path: str):
        data_path = os.path.join(data_path, 'dstc2')
        if not is_done(data_path):
            url = 'http://lnsigo.mipt.ru/export/datasets/dstc2.tar.gz'
            print('Loading DSTC2 from: {}'.format(url))
            download_untar(url, data_path)
            mark_done(data_path)
            print('DSTC2 dataset is built in {}'.format(data_path))
        return os.path.join(data_path, 'dstc2-trn.jsonlist')

    @staticmethod
    def read(data_path, *args, **kwargs):
        file_path = DSTC2Reader.build(data_path)
        logger.info("Reading instances from lines in file at: {}".format(file_path))
        utterances, responses, dialog_indices = \
            DSTC2Reader._read_turns(file_path, with_indices=True)

        data = [{'context': {'text': u['text'],
                             'intents': u['dialog_acts'],
                             'db_result': u.get('db_result', None)},
                 'response': {'text': r['text'],
                              'act': r['dialog_acts'][0]['act']}} \
                for u, r in zip(utterances, responses)]

        return {
            "train": [data[idx['start']:idx['end']] for idx in dialog_indices]
        }

    @staticmethod
    def _read_turns(file_path, with_indices=False):
        def _filter(turn):
            del turn['index']
            return turn

        utterances = []
        responses = []
        dialog_indices = []
        n = 0
        num_dial_utter, num_dial_resp = 0, 0
        with open(file_path, 'rt') as f:
            for ln in f:
                if not ln.strip():
                    if num_dial_utter != num_dial_resp:
                        raise RuntimeError("Datafile in the wrong format.")
                    n += num_dial_utter
                    dialog_indices.append({
                        'start': n - num_dial_utter,
                        'end': n,
                    })
                    num_dial_utter, num_dial_resp = 0, 0
                else:
                    replica = _filter(json.loads(ln))
                    if 'goals' in replica:
                        utterances.append(replica)
                        num_dial_utter += 1
                    else:
                        responses.append(replica)
                        num_dial_resp += 1

        if with_indices:
            return utterances, responses, dialog_indices
        return utterances, responses


@Registrable.register("provider.ner.dstc2")
class DSTC2NerProvider(DatasetProvider):

    @overrides
    def __init__(self, data, seed):
        r""" Dataset takes a dict with fields 'train', 'test', 'valid'. A list of samples (pairs x, y) is stored
             in each field.
             Args:
                data: list of (x, y) pairs. Each pair is a sample from the dataset. x as well as y can be a tuple
                    of different input features.
        """
        super().__init__(data, seed)
        # TODO: add external building
        with open("/home/aleksandr/Downloads/slot_vals.json") as f:
            self._slot_vals = json.load(f)
        for data_type in ['train', 'test', 'valid']:
            bio_markup_data = self._preprocess(data.get(data_type, []))
            setattr(self, data_type, bio_markup_data)
        self.data = {
            'train': self.train,
            'valid': self.valid,
            'test': self.test,
            'all': self.train + self.test + self.valid
        }

    def _preprocess(self, data_part):
        processed_data_part = list()
        slots = []
        for sample in data_part:
            for utterance in sample:
                text = utterance['context']['text']
                intents = utterance['context'].get('intents', dict())
                slots = list()
                for intent in intents:

                    current_slots = intent.get('slots', [])
                    for slot_type, slot_val in current_slots:
                        if slot_type in self._slot_vals:
                            slots.append((slot_type, slot_val,))

                processed_data_part.append(self._add_bio_markup(text, slots))
        return processed_data_part

    def _add_bio_markup(self, utterance, slots):
        tokens = utterance.split()
        n_toks = len(tokens)
        tags = ['O' for _ in range(n_toks)]
        for n in range(n_toks):
            for slot_type, slot_val in slots:
                for entity in self._slot_vals[slot_type][slot_val]:
                    slot_tokens = entity.split()
                    slot_len = len(slot_tokens)
                    if n + slot_len < n_toks and self._is_equal_sequences(tokens[n: n + slot_len], slot_tokens):
                        tags[n] = 'B-' + slot_type
                        for k in range(1, slot_len):
                            tags[n + k] = 'I-' + slot_type
                        break
        return tokens, tags

    def _is_equal_sequences(self, seq1, seq2):
        equality_list = [tok1 == tok2 for tok1, tok2 in zip(seq1, seq2)]
        return all(equality_list)

    def _build_slot_vals(self, slot_vals_json_path='data/'):
        url = 'http://lnsigo.mipt.ru/export/datasets/dstc_slot_vals.json'
        download(slot_vals_json_path, url)

    @overrides
    def batch_generator(self, batch_size, data_type='train'):
        for batch in super().batch_generator(batch_size, data_type):
            x_batch = batch[0]
            y_batch = batch[1]

            yield {
                "tokens": x_batch,
                "tags": y_batch
            }


@Registrable.register("provider.dialog.dstc2")
class DSTC2DialogProvider(DatasetProvider):

    @overrides
    def __init__(self, data, seed, *args, **kwargs):
        super().__init__(data, seed)

        def _wrap(turn):
            if isinstance(turn, list):
                turn[0]['context']['episode_done'] = True
                return list(map(_wrap, turn))
            else:
                x = turn['context']['text']
                y = turn['response']['text']
                other = dict()
                other['act'] = turn['response']['act']
                if turn['context'].get('db_result') is not None:
                    other['db_result'] = turn['context']['db_result']
                if turn['context'].get('episode_done'):
                    other['episode_done'] = True
                return x, y, other

        self.train = list(map(_wrap, data.get('train', [])))
        self.valid = list(map(_wrap, data.get('valid', [])))
        self.test = list(map(_wrap, data.get('test', [])))
        self.split(*args, **kwargs)
        self.data = {
            'train': self.train,
            'valid': self.valid,
            'test': self.test,
            'all': self.train + self.test + self.valid
        }

    @overrides
    def batch_generator(self, batch_size, data_type='train'):
        # Ignore batch_size for now
        for sample in self.data[data_type]:
            for utterance in sample:
                yield {
                    "text": utterance[0],
                    "response": utterance[1],
                    "other": utterance[2]
                }

    @overrides
    def iter_all(self, data_type: str = 'train'):
        for sample in self.data[data_type]:
            for utterance in sample:
                yield {
                    "text": utterance[0],
                    "response": utterance[1],
                    "other": utterance[2]
                }


@Registrable.register("provider.intents.dstc2")
class DSTC2IntentsProvider(DatasetProvider):
    def __init__(self, data,
                 seed=None, extract_classes=True, classes_file=None,
                 fields_to_merge=None, merged_field=None,
                 field_to_split=None, split_fields=None, split_proportions=None,
                 prep_method_name: str = None,
                 dataset_path=None, dataset_dir='intents', dataset_file='classes.txt',
                 *args, **kwargs):

        super().__init__(data, seed)
        self.classes = None

        # Reconstruct data to the necessary view
        # (x,y) where x - text, y - list of corresponding intents
        new_data = dict()
        new_data['train'] = []
        new_data['valid'] = []
        new_data['test'] = []

        for field in ['train', 'valid', 'test']:
            for turn in self.data[field]:
                for sample in turn:
                    reply = sample['context']
                    curr_intents = []
                    if reply['intents']:
                        for intent in reply['intents']:
                            for slot in intent['slots']:
                                if slot[0] == 'slot':
                                    curr_intents.append(intent['act'] + '_' + slot[1])
                                else:
                                    curr_intents.append(intent['act'] + '_' + slot[0])
                            if len(intent['slots']) == 0:
                                curr_intents.append(intent['act'])
                    else:
                        if reply['text']:
                            curr_intents.append('unknown')
                        else:
                            continue
                    new_data[field].append((reply['text'], curr_intents))

        self.data = new_data

        if extract_classes:
            self.classes = self._extract_classes()
            if classes_file is None:
                if dataset_path is None:
                    ser_dir = Path('.').joinpath(dataset_dir)
                    if not ser_dir.exists():
                        ser_dir.mkdir()
                    classes_file = Path('.').joinpath(dataset_dir, dataset_file)
                else:
                    ser_dir = Path(dataset_path).joinpath(dataset_dir)
                    if not ser_dir.exists():
                        ser_dir.mkdir()
                    classes_file = ser_dir.joinpath(dataset_file)

            print("No file name for classes provided. Classes are saved to file {}".format(
                classes_file))
            with open(Path(classes_file), 'w') as fin:
                for i in range(len(self.classes)):
                    fin.write(self.classes[i] + '\n')
        if fields_to_merge is not None:
            if merged_field is not None:
                print("Merging fields <<{}>> to new field <<{}>>".format(fields_to_merge,
                                                                         merged_field))
                self._merge_data(fields_to_merge=fields_to_merge.split(' '),
                                 merged_field=merged_field)
            else:
                raise IOError("Given fields to merge BUT not given name of merged field")

        if field_to_split is not None:
            if split_fields is not None:
                print("Splitting field <<{}>> to new fields <<{}>>".format(field_to_split,
                                                                           split_fields))
                self._split_data(field_to_split=field_to_split,
                                 split_fields=split_fields.split(" "),
                                 split_proportions=[float(s) for s in
                                                    split_proportions.split(" ")])
            else:
                raise IOError("Given field to split BUT not given names of split fields")

        self.prep_method_name = prep_method_name

        if prep_method_name:
            self.data = self.preprocess(PREPROCESSORS[prep_method_name])

    def _extract_classes(self):
        intents = []
        all_data = self.iter_all(data_type='train')
        for sample in all_data:
            intents.extend(sample[0][1])
        if 'valid' in self.data.keys():
            all_data = self.iter_all(data_type='valid')
            for sample in all_data:
                intents.extend(sample[0][1])
        intents = np.unique(intents)
        return np.array(sorted(intents))

    def _split_data(self, field_to_split, split_fields, split_proportions):
        data_to_div = self.data[field_to_split].copy()
        data_size = len(self.data[field_to_split])
        for i in range(len(split_fields) - 1):
            self.data[split_fields[i]], \
            data_to_div = train_test_split(data_to_div,
                                           test_size=
                                           len(data_to_div) - int(
                                               data_size * split_proportions[i]))
        self.data[split_fields[-1]] = data_to_div
        return True

    def _merge_data(self, fields_to_merge, merged_field):
        data = self.data.copy()
        data[merged_field] = []
        for name in fields_to_merge:
            data[merged_field] += self.data[name]
        self.data = data
        return True

    def preprocess(self, prep_method):

        data_copy = copy.deepcopy(self.data)

        for data_type in self.data:
            chunk = self.data[data_type]
            for i, sample in enumerate(chunk):
                data_copy[i] = (prep_method([sample[0]])[0], chunk[i][1])
        return data_copy

    @overrides
    def batch_generator(self, batch_size, data_type='train'):
        for sample in super().batch_generator(batch_size, data_type=data_type):
            yield {
                "text": [sample[0][0]],
                "intents": [sample[1][0]]
            }
