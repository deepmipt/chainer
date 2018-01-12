from deeppavlov.testing.test_case import DPTestCase
from deeppavlov.core.components import read_configuration, init_component
from deeppavlov.core.vocab import VocabComponent
import deeppavlov.ner
import os


class TestInfer(DPTestCase):

    def test_ner_infer(self):
        cfg = read_configuration("./conf/infer.ner.json")
        cmp = init_component(cfg)
        smem = {"text": "west of the town"}
        cmp.forward(smem)
        assert "tags" in smem
        cmp.shutdown()

    def test_bow_infer(self):
        cfg = read_configuration("./conf/infer.bow.json")
        cmp = init_component(cfg)
        smem = {"text": "cheap restaurant"}
        cmp.forward(smem)
        assert "bow" in smem
        for i, v in enumerate(smem["bow"]):
            if i == 5 or i == 36:
                assert v == 1
            else:
                assert v == 0
        cmp.shutdown()

    def test_w2v_infer(self):
        cfg = read_configuration("./conf/infer.w2v.json")
        cmp = init_component(cfg)
        smem = {"text": "Билл Гейтс президент компании Майкрософт открыл новый офис в Москве"}
        cmp.forward(smem)
        assert "emb" in smem
        cmp.shutdown()

