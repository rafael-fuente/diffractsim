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

class LinearPolarizer(OpticalElement):
    def __init__(self, name, axis='vertical'):
        super().__init__(name)
        self.axis = axis

    def apply(self, wavefront):
        # Apply linear polarizer effect to the wavefront based on the axis
        pass

def __init__(self, name, axis='vertical', angle=0):
        super().__init__(name)
        self.axis = axis
        self.angle = angle