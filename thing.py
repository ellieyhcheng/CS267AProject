# import pymc3 as pm
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans

# https://docs.pymc.io/

# https://github.com/factorie/factorie/tree/master/src/main/scala/cc/factorie

# https://github.com/mbforbes/py-factorgraph

# https://www.youtube.com/watch?v=rftT_TRtGi4
# https://dritchie.github.io/pdf/patterncolor.pdf


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
    #returns the discretized bin (bin index) of the color property of the color

def relative_size(pattern):
    

def spatial_features():
    return None

# Properties of color relationship between two adjacent regions
def perceptual_diff(c1, c2):
    pass

def relative_lightness(c1, c2):
    pass

def relative_saturation(c1, c2):
    pass

def chromatic_difference(c1, c2):
    pass


def discretizePi(color_property_values):
    kmeans = KMeans(n_clusters=10).fit(color_property_values)
    return kmeans

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

# Individual segments within a color group
class ColorGroupSegment:
    def __init__(self, area, total_area, max_seg_area, min_bbox):
        self.relative_size_to_pattern = area / total area
        self.relative_size_to_max_seg = area / max_seg_area
        box_width, box_height = min_bbox
        self.elongation = 1 - (box_width / box_height)

        self.spatial_features = (self.relative_size_to_pattern, self.relative_size_to_max_seg, self.elongation)
        
        self.color_properties = () 

def score_grp(cg): #phi
    pass

def score_adj(cg1, cg2): 
    return 

def OD_compat(c1,c2,c3,c4,c5):
    #O'Donovan color compatibility model
    return 5

def score_compat(c1,c2,c3,c4,c5):
    return np.log(OD_compat(c1,c2,c3,c4,c5)/5)

def main():
    h = histogram_thingy()
    pass

if __name__ == '__main__':
    main()