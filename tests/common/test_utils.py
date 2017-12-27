from deeppavlov.testing.test_case import DPTestCase
from deeppavlov.core.components import read_configuration, init_component, load_cls, TrainPipeline


class TestUtils(DPTestCase):

    def test_read_config(self):
        cfg = read_configuration("./conf/provider.ner.dstc2.json")
        assert cfg["batch_size"] == 10

    def test_init_component(self):
        cfg = read_configuration("./conf/train.ner.json")
        cmp = init_component(cfg)
        assert isinstance(cmp, TrainPipeline)

