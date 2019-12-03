from concern.config import State

from .data_process import DataProcess


class FilterKeys(DataProcess):
    required = State(default=[])
    superfluous = State(default=[])

    def __init__(self, **kwargs):
        super().__init__(self, **kwargs)

        self.required_keys = set(self.required)
        self.superfluous_keys = set(self.superfluous)
        if len(self.required_keys) > 0 and len(self.superfluous_keys) > 0:
            raise ValueError(
                'required_keys and superfluous_keys can not be specified at the same time.')

    def process(self, data):
        for key in self.required:
            assert key in data, '%s is required in data' % key

        superfluous = self.superfluous_keys
        if len(superfluous) == 0:
            for key in data.keys():
                if key not in self.required_keys:
                    superfluous.add(key)

        for key in superfluous:
            del data[key]
        return data
