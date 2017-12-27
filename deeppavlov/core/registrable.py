from collections import defaultdict


class Registrable:

    _registry = defaultdict(dict)

    @classmethod
    def register(cls, name):
        registry = Registrable._registry[cls]

        def add_to_register(subclass):
            if name in registry:
                message = "Can't register %s as %s. Name already in use for %s" % (
                    name, cls.__name__, registry[name].__name__)
                raise ConnectionError(message)
            registry[name]=subclass
            return subclass
        return add_to_register

    @classmethod
    def by_name(cls, name):
        registry = Registrable._registry[cls]
        if name not in registry:
            raise ConnectionError("%s is not registered name for %s" % (name, cls.__name__))
        return registry.get(name)

    @classmethod
    def list_available(cls):
        return list(Registrable._registry[cls])