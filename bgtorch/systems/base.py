# TODO: write docstrings


class EnergyBasedModel(object):
    
    def sample(self, sample_shape, temperature=None):
        raise NotImplementedError()
        
    def energy(self, x, temperature=None):
        raise NotImplementedError()