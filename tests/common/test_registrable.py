from deeppavlov.testing.test_case import DPTestCase
from deeppavlov.preprocessing.tokenizers import SpacyTokenizerComponent


class TestRegistrable(DPTestCase):

    def test_registrable(self):
        base_class = SpacyTokenizerComponent

        assert "dummy" not in base_class.list_available()

        @base_class.register("dummy")
        class Dummy(base_class):
            pass

        assert base_class.by_name("dummy") == Dummy

        assert "dummy" in base_class.list_available()
        assert 1 == len(base_class.list_available())
