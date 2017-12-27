from deeppavlov.testing.test_case import DPTestCase
from deeppavlov.core.components import read_configuration, init_component
from deeppavlov.core.vocab import VocabComponent
import deeppavlov.ner
import os


class TestTrain(DPTestCase):

    def test_ner_train(self):
        cfg = read_configuration("./conf/infer.ner.json")
        cmp = init_component(cfg)
        smem = {"text": "Билл Гейтс президент компании Майкрософт открыл новый офис в Москве"}
        cmp.forward(smem)
        assert "tags" in smem
        cmp.shutdown()
