from abc import ABC, abstractmethod


class AbstractFactory(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def run(self):
        """
        docstring
        """
        pass


class AbstractProduct(ABC):
    def __init__(self) -> None:
        pass

    def _get_credentials(self) -> None:
        pass

    @abstractmethod
    def _connect(self):
        """
        docstring
        """
        pass
