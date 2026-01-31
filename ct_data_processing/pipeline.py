from abc import ABC, abstractmethod


class PipelinePart(ABC):
    @abstractmethod
    def __call__(self, *data, **params):
        pass


class PipelineStack(PipelinePart):
    def __init__(self, transforms):
        self._transforms = transforms

    def __call__(self, *data, **params):
        for t in self._transforms:
            data, params = t(*data, **params)
        return data, params

