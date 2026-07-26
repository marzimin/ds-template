from abc import ABC, abstractmethod


class Pipeline(ABC):
    """Abstract base class for all pipelines."""

    @abstractmethod
    def run(self) -> None:
        """Execute the pipeline's workflow."""
        raise NotImplementedError
