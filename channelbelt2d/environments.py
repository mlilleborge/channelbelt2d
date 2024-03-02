import numpy as np

# Valley confining the river
class Valley:
    def __init__(self, valley_parameters, n=100):
        self._width = valley_parameters._get_width()
        self._depth = valley_parameters._get_depth()
        self._level = valley_parameters._get_initial_level()

        w = self._width
        d = self._depth
        self._xx = np.linspace(-w/2, w/2, n)
        self._y0 = self.shape(self._xx, w, d)
        
    def shape(self, x, w, d):
        return d * (1 - (2 * x / w)**4)

    def get_x(self):
        return self._xx

    def get_y(self):
        return np.minimum(self._y0, self._level * np.ones_like(self._xx))
    
    def get_y_point(self, x):
        return np.minimum(self.shape(x, self._width, self._depth), self._level)
    
    def get_x_min(self):
        return -self._width/2
    
    def get_x_max(self):
        return self._width/2

    def plot(self, ax):
        # initial valley
        ax.plot(self._xx, self._y0, color='black')

        # fill to current level
        ax.fill_between(self._xx, self._y0, self.get_y(), color='yellowgreen')

    def set_level(self, newlevel):
        self._level = newlevel
    
    def aggrade(self, aggradation):
        self._level -= aggradation
    
    def plot_background(self, ax):
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        x = np.concatenate(([x_min], self._xx, [x_max]))
        y_top = np.concatenate(([y_min], self._y0, [y_min]))
        y_bottom = np.repeat(y_max, x.size)
        ax.fill_between(x, y_top, y_bottom, color='darkgray')
    
    def get_minimum_depth(self):
        return self._y0.min()


# Parameters defining valley geometry
class ValleyParameters:
    def __init__(self, width, depth, initial_level=None):
        self._width = width
        self._depth = depth

        # Initial level equals depth by default
        self._initial_level = depth if initial_level is None else initial_level
    
    def _get_width(self):
        return self._width
    
    def _get_depth(self):
        return self._depth
    
    def _get_initial_level(self):
        return self._initial_level