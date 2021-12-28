from abc import ABC, abstractmethod

class LightSource(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_E(self, E, xx, yy, λ):
        pass
