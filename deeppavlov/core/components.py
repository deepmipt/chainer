from deeppavlov.core.registrable import Registrable
from deeppavlov.core.data import DatasetProvider
import copy
import importlib
import logging
from pyhocon import ConfigFactory
from overrides import overrides

logger = logging.getLogger(__name__)


class Component(Registrable):
    def __init__(self, config):
        self.config = copy.deepcopy(config)
        self.disable = self._is_disable()
        self.shared_mem = None

        self.inputs = self._get_inputs()
        self.outputs = self._get_outputs()

        self.inputs_alias = self._get_inputs_alias()
        self.outputs_alias = self._get_outputs_alias()

        self.local_input_names = []
        self.local_output_names = []

        self._setup = {}

    def _is_disable(self):
        return "disable" in self.config and self.config["disable"]

    def _get_outputs(self):
        return self.config["out"] if "out" in self.config else []

    def _get_inputs(self):
        return self.config["in"] if "in" in self.config else []

    def _get_outputs_alias(self):
        if "out_alias" in self.config:
            return self.config["out_alias"]
        else:
            return self._get_outputs()

    def _get_inputs_alias(self):
        if "in_alias" in self.config:
            return self.config["in_alias"]
        else:
            return self._get_inputs()

    def _get_component_id(self):
        return self.config["id"] if "id" in self.config else None

    def _get_input_by_idx(self, idx, shared_mem):
        return shared_mem[self.inputs[idx]]

    def get_input(self, name, shared_mem):
        return self._get_input_by_idx(self.local_input_names.index(name), shared_mem)

    def _set_output_by_idx(self, idx, value, shared_mem):
        key = self.outputs[idx]
        shared_mem[key] = value

    def set_output(self, name, value, smem):
        self._set_output_by_idx(self.local_output_names.index(name), value, smem)

    def forward(self, shared_mem, add_local_mem=False):
        pass

    def train(self, shared_mem, add_local_mem=False):
        pass

    def save(self):
        pass

    def load(self):
        pass

    def shutdown(self):
        pass

    def setup(self, components={}):
        if "init" in self.config:
            for local_key, ext_key in self.config["init"].items():
                if ext_key in components:
                    self._setup[local_key] = components[ext_key]


class DatasetProviderWrapper(Component):
    def __init__(self, config):
        super().__init__(config)

        self.data_path = self.config["data_path"] if "data_path" in self.config else None
        self.batch_size = self.config["batch_size"] if "batch_size" in self.config else -1
        self.data_type = self.config["data_type"] if "data_type" in self.config else 'train'
        self.seed = self.config["seed"] if "seed" in self.config else 1

        self.reader_cls = load_cls(self.config["reader"])

        self.provider_cls = self.config["provider"]

        self.provider = self.provider_cls(self._read_data(), self.seed)
        self.generator = self.provider.batch_generator(self.batch_size, self.data_type)
        self.current_epoch = 0
        self.batch_num = 0

    def _read_data(self):
        return self.reader_cls().read(self.data_path) if self.data_path is not None else self.reader_cls.read()

    @overrides
    def forward(self, shared_mem, add_local_mem=False):
        self.train(shared_mem, add_local_mem=add_local_mem)

    @overrides
    def train(self, shared_mem, add_local_mem=False):
        epoch = shared_mem["epoch"]
        if epoch > self.current_epoch:
            self.current_epoch = epoch
            self.batch_num = 0
            self.generator = self.provider.batch_generator(self.batch_size, self.data_type)

        batch = next(self.generator)
        self.batch_num += 1
        logger.debug("Train on batch %s" % self.batch_num)
        for k_out, k_in in zip(self.outputs, batch.keys()):
            shared_mem[k_out] = batch[k_in]


class Pipeline(Component):
    def __init__(self, config):
        super().__init__(config)
        self.pipeline = []
        self._build_pipeline()

        self._components = {}

        self._setup = {}

    def _build_pipeline(self):
        for component_config in self.config['pipe']:
            self.pipeline.append(init_component(component_config))

    def prepare_pipeline(self):
        pipe = []
        for c in self.pipeline:
            if isinstance(c, TrainPipeline):
                c.train({})
                c.save()
                pipe.append(c.get_trained_component())
            else:
                pipe.append(c)
        self.pipeline = pipe

    @overrides
    def forward(self, shared_mem, add_local_mem=False, train=False):
        self.prepare_pipeline()
        self.setup()
        for c in self.pipeline:
            if not c.disable:
                c.forward(shared_mem, add_local_mem=add_local_mem)

    @overrides
    def setup(self, components={}):
        comps = {}
        for c in self.pipeline:
            if "id" in c.config:
                comps.update({c.config["id"]: c})
        for c in self.pipeline:
            c.setup({**components, **comps})

    @overrides
    def train(self, shared_mem, add_local_mem=False):
        raise NotImplementedError

    @overrides
    def save(self):
        for c in self.pipeline:
            c.save()

    @overrides
    def load(self):
        for c in self.pipeline:
            c.load()

    @overrides
    def shutdown(self):
        for c in self.pipeline:
            c.shutdown()


class TrainPipeline(Pipeline):
    def __init__(self, config):
        super().__init__(config)

    @overrides
    def train(self, shared_mem, add_local_mem=False):
        self.prepare_pipeline()
        self.setup()

        n = int(self.config["train"]["num_epochs"])
        local_mem = {}
        for e in range(n):
            logger.info("Start epoch %s" % e)
            local_mem["epoch"] = e
            pipe = self.pipeline[:-1]
            trained = self.pipeline[-1]
            if len(pipe) > 0:
                try:
                    while True:
                        for c in pipe:
                            c.forward(local_mem, add_local_mem=add_local_mem)
                        trained.train(local_mem, add_local_mem=add_local_mem)
                except StopIteration:
                    logger.info("End of epoch %s" % e)
            else:
                trained.train(local_mem, add_local_mem=add_local_mem)
                logger.info("End of epoch %s" % e)
        self.save()

    def get_trained_component(self):
        cmp = self.pipeline[-1]
        cmp.inputs = self.inputs
        cmp.inputs_alias = self.inputs_alias
        cmp.outputs = self.outputs
        cmp.outputs_alias = self.outputs_alias
        if "id" in self.config:
            cmp.config["id"] = self.config["id"]
        return cmp


def read_configuration(file):
    config = ConfigFactory.parse_file(file)
    return config


def load_cls(cls):
    module_name = cls[:cls.rfind(".")]
    class_name = cls[cls.rfind(".") + 1:]
    _module = importlib.import_module(module_name)
    _class = getattr(_module, class_name)
    return _class


def init_component(cfg):
    # if ("train" not in cfg) and ("pipe" not in cfg) and ("config" not in cfg):
    #     cfg["config"] = {}

    if "config" in cfg:
        if isinstance(cfg["config"], str):
            cfg_upd = read_configuration(cfg["config"])
        else:
            cfg_upd = cfg["config"]

        cfg_copy = copy.deepcopy(cfg)

        del cfg_copy["config"]

        if "in" not in cfg_copy:
            cfg_copy["in"] = []

        cfg_copy["in_alias"] = cfg_copy["in"]

        if "out" not in cfg_copy:
            cfg_copy["out"] = []

        cfg_copy["out_alias"] = cfg_copy["out"]

        cfg_upd.update(cfg_copy)
        return init_component(cfg_upd)
    else:
        if "train" in cfg:
            return TrainPipeline(cfg)
        elif "pipe" in cfg:
            return Pipeline(cfg)
        else:
            cls = get_component_class(cfg)
            if issubclass(cls, DatasetProvider):
                cfg["provider"] = cls
                return DatasetProviderWrapper(cfg)
            else:
                return cls(cfg)


def get_component_class(cfg):
    return Registrable.by_name(cfg["component"])