from abc import ABC, abstractmethod


class PipelinePart(ABC):
    @abstractmethod
    def __call__(self, *data, **params):
        pass


class PipelineStack(PipelinePart):
    def __init__(self, pipeline_parts):
        self._pipeline_parts = pipeline_parts

    def __call__(self, *data, **params):
        for pp in self._pipeline_parts:
            data, params = pp(*data, **params)
        return data, params

