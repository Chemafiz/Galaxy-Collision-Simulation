import numpy as np


class Window:
    def __init__(self, window_size, title):
        self.window_size = window_size
        self.title = title

        self.window_matrix = np.zeros(window_size, dtype=np.uint8)
