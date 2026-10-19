"""Native PixArt-Alpha256: share all PixArt routing with Sigma."""
from .sigma import SigmaModel
class PixArtAlphaModel(SigmaModel):
    spec_name='pixart-alpha'
