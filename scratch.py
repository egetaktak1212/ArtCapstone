import numpy as np


def changeOverlay(self, guidelines, sketch, og):
    frame = np.zeros_like(og)

    if (og):
        frame = og
    if sketch:
        frame = self.masking(sketch, frame)
    if guidelines:
        frame = self.masking(guidelines, frame)

def masking(self, fg, bg):
    mask = np.any(fg != [0, 0, 0], axis=-1)
    result = bg.copy()
    result[mask] = fg[mask]
    return result