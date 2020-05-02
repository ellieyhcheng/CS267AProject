# import pymc3 as pm
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from segment import preprocess_image, segment_image, get_color_groups
from cv2 import minAreaRect

# https://docs.pymc.io/

# https://github.com/factorie/factorie/tree/master/src/main/scala/cc/factorie

# https://github.com/mbforbes/py-factorgraph

# https://www.youtube.com/watch?v=rftT_TRtGi4
# https://dritchie.github.io/pdf/patterncolor.pdf

# http://graphics.stanford.edu/projects/patternColoring/

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

def centroid(segment):
    length = len(segment)
    s = np.array(segment)
    sum_x = np.sum(s[:, 0])
    sum_y = np.sum(s[:, 1])
    return sum_x/length, sum_y/length

# Spacial features
def relative_size(segments, color):
    color_group = segments[color]
    area = sum([len(s) for s in color_group])
    group_areas = [sum(len(s) for s in group) for group in segments.values()]
    max_group_area = max(group_areas)
    total_area = sum(group_areas)
    return area,total_area, area/max_group_area

def segment_spread(color_group):
    centroids = np.array([centroid(s) for s in color_group])
    return np.cov(centroids)

def segment_size_stats(color_group):
    sizes = np.array([len(s) for s in color_group])
    return sizes.min(), sizes.max(), sizes.mean(), sizes.std()

def number_segments(segments, color):
    num = len(segments[color])
    total = sum([len(group) for group in segments.values()])
    return num/total

def relative_size_ind(segment, segments):
    area = len(segment)
    segment_areas = [len(s) for s in group for group in segments.values()]
    total_area = sum(segment_areas)
    max_segment_area = max(segment_areas)
    return area/total_area, area/max_segment_area

def num_neighbors(matrix, i, j):
    width, height = matrix.shape
    count = 0
    if (i > 0 and matrix[i-1][j])
        or (i < height - 1 and matrix[i+1][j])
        or (j > 0 and matrix[i][j-1])
        or (j < width - 1 and matrix[i][j+1]):
        count += 1
    return count

def normalized_discrete_compactness(width, height, segment):
    matrix = np.zeros((width, height))
    for px in segment:
        matrix[px[0]][px[1]] = 1
    
    P = 0
    for i in range(width):
        for j in range(height):
            if matrix[i][j]:
                P += (4 - num_neighbors(matrix, i, j))

    T = 4 # number of sides to a cell
    n = len(segment)
    pc = (T*n - P)/2
    cd = pc
    cd_min = n - 1
    cd_max = (T*n - 4 * np.sqrt(n))/2
    cdn = (cd - cd_min) / (cd_max - cd_min)
    return cdn

def elongation(segment):
    rect = minAreaRect(segment)
    (x,y), (width,height), angle = rect
    return 1 - width/height

def centrality(width, height, segment):
    centroid_x, centroid_y = centroid(segment)
    x, y = width/2, height/2
    return np.sqrt((x-centroid_x)**2 + (y-centroid_y)**2)

# background = 0, foreground = 1
def role_labels(segments, palette):
    labels = {}
    largest_segments = {}
    for color in palette:
        largest_segments[color] = [max([len(s) for s in segments[color]])]
    background = None
    for color in palette:
        if largest_segments[color] > color:
            background = color 
    
    for color in palette:
        if color == background:
            labels[color] = 0
        else:
            labels[color] = 1
    return labels

def spatial_features():
    return None

# Properties of color relationship between two adjacent regions
def perceptual_diff(c1, c2):
    l1, a1, b1 = c1
    l2, a2, b2 = c2
    return np.sqrt(l1**2+a1**2+b1**2) - np.sqrt(l2**2+a2**2+b2**2)

def relative_lightness(c1, c2):
    l1, _, _ = c1
    l2, _, _ = c2
    return l1 - l2

def relative_saturation(c1, c2):
    return saturation(c1) - saturation(c2)

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

class ColorGroup:
    def __init__(self):
        pass

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