# import pymc3 as pm
import numpy as np
from sklearn.linear_model import LogisticRegression
from skimage import io, color
from sklearn.cluster import KMeans

# https://docs.pymc.io/

# https://github.com/factorie/factorie/tree/master/src/main/scala/cc/factorie

# https://github.com/mbforbes/py-factorgraph

# https://www.youtube.com/watch?v=rftT_TRtGi4
# https://dritchie.github.io/pdf/patterncolor.pdf


# lab = color.rgb2lab(rgb)

def lightness(color):
    L,a,b = color
    return L

def saturation(color):
    L,a,b = color
    return np.sqrt(a**2+b**2)/np.sqrt(a**2+b**2+L**2)

def color_property(color):
    return (lightness(color),saturation(color))

def getbin(color):
    return None
    #returns the discretized bin of the color property of the color

def relative_size(pattern):
    

def spatial_features():
    return None

class histogram_thingy():
    def __init__(self):
        pass

    def train(self, x_train, y_train):
        # get spatial and color properties
        x_train = [spatial_features(x) for x in x_train]
        y_train = [getbin(color) for y in y_train]
        self.clf = LogisticRegression(multiclass='multinomial')
        self.clf.fit(x_train,y_train)

    def get_histogram(self, x):
        return self.clf.decision_function(x)

def discretizePi(color_property_values):
    kmeans = KMeans(n_clusters=10).fit(color_property_values)
    return kmeans


# Individual segments within a color group
class ColorGroupSegment:
    def __init__(self, area, pattern_area, )
    
