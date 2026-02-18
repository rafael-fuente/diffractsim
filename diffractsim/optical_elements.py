# diffractsim/optical_elements.py

class OpticalElement:
    def __init__(self, name):
        self.name = name

    def apply(self, wavefront):
        raise NotImplementedError("Subclasses must implement this method")

class Mirror(OpticalElement):
    def apply(self, wavefront):
        # Apply mirror effect to the wavefront
        pass

class Lens(OpticalElement):
    def apply(self, wavefront):
        # Apply lens effect to the wavefront
        pass