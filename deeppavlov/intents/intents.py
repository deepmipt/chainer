from deeppavlov.core.components import Component
from deeppavlov.core.registrable import Registrable
from deeppavlov.intents.model import KerasMulticlassModel
from overrides import overrides
import logging

logger = logging.getLogger(__name__)


@Registrable.register("intents")
class IntentsComponent(Component):
    def __init__(self, config):
        super().__init__(config)
        self.local_input_names = ['tokens', 'intents']
        self.local_output_names = ['result']

        self._is_model_initialized = False

        self.model = None

    @overrides
    def setup(self, components={}):
        super().setup(components)
        if self.model is None:
            self.model = KerasMulticlassModel(self.config)
            self.load()

    @overrides
    def save(self):
        if "save_to" in self.config:
            path = self.config["save_to"]
            self.model.save(path)

    @overrides
    def load(self):
        if "load" in self.config:
            path = self.config["load"]
            self.model.load(path)

    @overrides
    def forward(self, smem, add_local_mem=False):
        tokens = self._get_input_by_idx(0, smem)

        if isinstance(tokens, list):
            prediction = self.model.infer(tokens)
        else:
            prediction = self.model.infer([tokens])

        self.set_output("result", prediction, smem)

    @overrides
    def train(self, smem, add_local_mem=False):

        tokens_batch = self.get_input("tokens", smem)

        intents_batch = self.get_input("intents", smem)

        loss = self.model.train_on_batch((tokens_batch, intents_batch,))

        self.set_output("result", loss, smem)
        logger.debug("Loss %s" % loss)

