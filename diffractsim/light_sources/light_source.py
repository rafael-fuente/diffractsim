from abc import ABC, abstractmethod

class LightSource(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_E(self, E, xx, yy, λ):
        """Returns the complex electric field vector (Ex, Ey) at each point (xx,yy)
        for wavelength λ, given input field E"""
        pass
