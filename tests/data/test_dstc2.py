from deeppavlov.testing.test_case import DPTestCase
from deeppavlov.data.dstc2 import DSTC2Reader, DSTC2NerProvider
from deeppavlov.core.registrable import Registrable


class TestDSTC2(DPTestCase):

    def test_dstc2reader(self):
        data = DSTC2Reader.read(data_path=self.TEST_DIR)
        assert "train" in data
        assert len(data["train"]) == 967

        sample = data["train"][0]

        assert 7 == len(sample)

        assert sample[0] == {
            'context': {
                'text': '',
                'intents': [],
                'db_result': None
            },
            'response': {
                'text': 'Hello, welcome to the Cambridge restaurant system. You can ask for restaurants by area, price range or food type. How may I help you?',
                'act': 'welcomemsg'
            }
        }

    def test_dstc2provider(self):
        data = DSTC2Reader.read(data_path=self.TEST_DIR)
        provider = DSTC2NerProvider(data, 1)
        batches = provider.batch_generator(10)
        batch = next(batches)
        assert batch == {
            'tokens': (['airatarin'], ['thank', 'you', 'good', 'bye'], [], [], ['restaurant'], ['thank', 'you', 'good', 'bye'], [], [], ['what', 'about', 'any', 'area'], ['im', 'looking', 'for', 'an', 'expensive', 'restaurant', 'in', 'the', 'east', 'part', 'of', 'town']),
            'tags': (['O'], ['O', 'O', 'O', 'O'], [], [], ['O'], ['O', 'O', 'O', 'O'], [], [], ['O', 'O', 'O', 'O'], ['O', 'O', 'O', 'O', 'B-pricerange', 'O', 'O', 'O', 'B-area', 'O', 'O', 'O'])
        }

    def test_dstc2_ner_dataset(self):
        ds = Registrable.by_name("provider.ner.dstc2")
        self.assertIsInstance(ds, DSTC2NerProvider.__class__)

