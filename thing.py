# import pymc3 as pm
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.neighbors import KernelDensity
from segment import segment_image, get_color_groups, hex2rgb
from cv2 import minAreaRect
from skimage.color import rgb2lab
import csv
from PIL import Image
import os
import pickle
import matplotlib.pyplot as plt

# https://docs.pymc.io/

# https://github.com/factorie/factorie/tree/master/src/main/scala/cc/factorie

# https://github.com/mbforbes/py-factorgraph

# https://www.youtube.com/watch?v=rftT_TRtGi4
# https://dritchie.github.io/pdf/patterncolor.pdf

# http://graphics.stanford.edu/projects/patternColoring/

def lightness(color):
    # L,a,b = color
    L = color[0]
    a = color[1]
    b = color[2]
    return (L/100.0)

def saturation(color):
    # print(color)
    # L,a,b = color
    L = color[0]
    a = color[1]
    b = color[2]
    # print(L,a,b)
    # print(np.sqrt(a**2+b**2), np.sqrt(a**2+b**2+L**2))
    if np.sqrt(a**2+b**2+L**2) == 0.0:
        return 0.0
    return np.sqrt(a**2+b**2)/(np.sqrt(a**2+b**2+L**2))

def color_property(color):
    return (lightness(color),saturation(color))

def discretize_color_property(color_property_values):
    kmeans = KMeans(n_clusters=10).fit(np.array(color_property_values).reshape(-1, 1))
    return kmeans

def getbin(kmeans, color_property):
    return kmeans.predict(np.array(color_property).reshape(-1,1))[0]
    #returns the discretized bin (bin index) of the color property of the color

def centroid(segment):
    length = len(segment)
    s = np.array(segment)
    sum_x = np.sum(s[:, 0])
    sum_y = np.sum(s[:, 1])
    return sum_x/length, sum_y/length

# Spatial features       
def relative_size(segments, color):
    color_group = segments[color]
    area = sum([len(s) for s in color_group])
    group_areas = [sum(len(s) for s in group) for group in segments.values()]
    max_group_area = max(group_areas)
    total_area = sum(group_areas)
    return area,total_area, area/max_group_area

def segment_spread(color_group):
    centroids = np.array([centroid(s) for s in color_group])
    return np.cov(centroids).flatten()

def segment_size_stats(color_group):
    sizes = np.array([len(s) for s in color_group])
    return sizes.min(), sizes.max(), sizes.mean(), sizes.std()

def number_segments(segments, color):
    num = len(segments[color])
    total = sum([len(group) for group in segments.values()])
    return num/total

def relative_size_ind(segment, segments):
    area = len(segment)
    # segment_areas = []
    # for group in segments.values:
    segment_areas = [len(s) for group in segments.values() for s in group]
    total_area = sum(segment_areas)
    max_segment_area = max(segment_areas)
    return area/total_area, area/max_segment_area

def num_neighbors(matrix, i, j):
    height, width = matrix.shape
    count = 0
    if (i > 0 and matrix[i-1][j]) or (i < height - 1 and matrix[i+1][j]) or (j > 0 and matrix[i][j-1]) or (j < width - 1 and matrix[i][j+1]):
        count += 1
    return count

def normalized_discrete_compactness(width, height, segment):
    matrix = np.zeros((height, width))
    # print('seg', segment)
    for px in segment:
        matrix[px[0]][px[1]] = 1
    
    P = 0
    # for i in range(height):
    #     for j in range(width):
    for px in segment:
        P += (4 - num_neighbors(matrix, px[0], px[1]))

    T = 4 # number of sides to a cell
    n = len(segment)
    pc = (T*n - P)/2
    cd = pc
    cd_min = n - 1
    cd_max = (T*n - 4 * np.sqrt(n))/2
    cdn = (cd - cd_min) / (cd_max - cd_min + 1)
    return cdn

def elongation(segment): 
    # TODO: BROKEN
    # rect = minAreaRect(segment)
    # (x,y), (width,height), angle = rect
    x = [px[0] for px in segment]
    y = [px[1] for px in segment]
    width = max(x) - min(x)+1
    height = max(y) - min(y)+1
    return 1 - width/height

def centrality(width, height, segment):
    centroid_x, centroid_y = centroid(segment)
    x, y = width/2, height/2
    return np.sqrt((x-centroid_x)**2 + (y-centroid_y)**2)

# background = 0, foreground = 1
def role_labels(segments):
    labels = {}
    largest_segments = {}
    palette = list(segments.keys())
    for color in palette:
        largest_segments[color] = max([len(s) for s in segments[color]])

    background = palette[0]
    for color in palette:
        if largest_segments[color] > largest_segments[background]:
            background = color 
    
    for color in palette:
        if color == background:
            labels[color] = 0
        else:
            labels[color] = 1
    return labels

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
    l1, a1, b1 = c1
    l2, a2, b2 = c2
    dasqr = (a1 - a2)**2
    dbsqr = (b1 - b2)**2
    dlsqr = (l1 - l2)**2
    return (dasqr + dbsqr) / (dasqr + dbsqr + dlsqr)

def enclosure_strength(segment_1, segment_2):
    # how much one segment in the adjacency encloses the other and vice versa. 
    # Enclosure Strength is defined as the number of pixels of the neighboring 
    # segment appearing within a 2-pixel neighborhood outside the segment’s
    # boundary, normalized by the area of that neighborhood. Out-of-image pixels 
    # are counted as part of the neighborhood area
    pass

class Histogram:
    def __init__(self):
        pass

    def train(self, spatial_properties, color_property):
        self.spatial_property_scaler = StandardScaler().fit(spatial_properties)
        self.kmeans = discretize_color_property(color_property)
        x_train = np.array(self.spatial_property_scaler.transform(spatial_properties))
        color_property = np.array(color_property)
        # y_train = np.array((color_property.shape[0],10))
        # for idx, cp in enumerate(color_property):
        #     y_train[idx][cp]
        y_train = np.array([getbin(self.kmeans, cp) for cp in color_property])
        self.clf = LogisticRegression(multi_class='multinomial', max_iter=10000)
        self.clf.fit(x_train,y_train)
        return self.clf.score(x_train,y_train)

    def get_histogram(self, x):
        return self.clf.predict_proba(self.spatial_property_scaler.transform(x.reshape(1, -1)))[0] 

    def get_range(self):
        return (self.kmeans.cluster_centers_).reshape(1,-1)[0]

    def get_prob_distribution(self, spatial_property):
        probs = self.get_histogram(spatial_property)
        color_property_values = self.get_range()
        ind = color_property_values.argsort()
        probs = probs[ind]
        color_property_values = color_property_values[ind]
        data = np.concatenate((color_property_values.reshape(-1,1), probs.reshape(-1,1)), axis=1)
        kde = KernelDensity(bandwidth=0.11, kernel='gaussian').fit(data[:,0].reshape(-1,1), sample_weight=data[:,1])
        # x = np.linspace(0,1,50)
        # log_dens = kde.score_samples(x.reshape(-1,1))
        def get_prob(color_property_value):
            log_dens = kde.score_samples([[color_property_value]])
            return np.exp(log_dens)[0]
        return get_prob

class ColorGroup:
    # u get a map from color to list of segments where each segment is jsut a list of coordinates of that segment
    # like {'FFFFFF' : [[(0,0),(0,1)...] , [(10,10),(10,11)...], ...],
    #       '123455' : [[]]}
    # so u just pass in each of the colors and then for each color, u make a bunch of ColorGroupSegments
    # its a map from color to a list of lists
    def __init__(self, id, segments, color, pattern_width, pattern_height, label):
        self.id = id
        r,g,b = hex2rgb(color)
        rgb = [[[r/255,g/255,b/255]]]
        light = lightness(rgb2lab(rgb)[0][0])
        sat = saturation(rgb2lab(rgb)[0][0])
        self.color_property = (light, sat)
        
        self.color = color # in hex
        color_segments = segments[color]
        self.area = sum([len(s) for s in color_segments])

        relative_size_to_pattern = relative_size(segments,color)[0]
        relative_size_to_max_group =relative_size(segments,color)[1]
        seg_spread = segment_spread(color_segments)
        label = label
        
        self.color_groups = [ColorGroupSegment(seg, segments,color, pattern_width, pattern_height, label) for seg in color_segments]

        # Segment size statistics
        min_segment_size, max_segment_size, mean_segment_size, std_segment_size = segment_size_stats(color_segments)
        
        self.spatial_property = np.array([relative_size_to_pattern, relative_size_to_max_group, label, min_segment_size, max_segment_size, mean_segment_size, std_segment_size])
        # self.spatial_property = np.concatenate((self.spatial_property, self.seg_spread))
        #TODO: How use a matrix as a feature bruh

# Individual segments within a color group
class ColorGroupSegment: 
    def __init__(self, segment, segments, color, pattern_width, pattern_height,label):
        self.color = color
        r,g,b = hex2rgb(color)
        rgb = [[[r/255,g/255,b/255]]]
        light = lightness(rgb2lab(rgb)[0][0])
        sat = saturation(rgb2lab(rgb)[0][0])
        self.color_property = (light, sat)
        self.area = sum([len(s) for s in segment])

        relative_size = relative_size_ind(segment,segments)
        num_neighbors = normalized_discrete_compactness(pattern_width, pattern_height, segment)
        elon = elongation(segment)
        label = label # foreground = 1, background = 0
        cent = centrality(pattern_width, pattern_height, segment)

        self.spacial_property = (relative_size, num_neighbors, elon, label, cent)

def score_grp(histogram, spatial_property, color_property, area): #phi
    prob_dist = histogram.get_prob_distribution(spatial_property)
    p = prob_dist(color_property)
    return np.log(p) * area, p

def score_seg(histogram, spatial_property, color_property, area): #phi
    prob_dist = histogram.get_prob_distribution(spatial_property)
    p = prob_dist(color_property)
    return np.log(p) * area, p

# def score_adj(histogram, spatial_property, )

def main():
    processing = True
    count = 1200
    end = 1500
    pickle_file = 'colorgroup2.pickle'
    color_group_tests = np.arange(0,20,1)

    # list of list of colorgroups
    if processing:
        all_color_groups = []
        with open(pickle_file, 'ab') as pf:
            with open(os.path.join('test_set2', 'test.csv')) as csvfile:
                reader = list(csv.DictReader(csvfile))
                while count != end and count < len(reader):
                    row = reader[count]
                    # print("hello", count)
                    if count % 100 == 0:
                        print("Finished", count, "Color groups")
                    count += 1
                    img_num = row['patternId']
                    palette = row['palette'].strip().split(' ')
                    # print(img_num)
                    testimg_file = os.path.join('test_set2', str(img_num)+'.png')
                    with Image.open(testimg_file) as imgfile:
                        img = imgfile.convert('RGBA')
                        img = np.array(img)

                    height,width,d = img.shape
                    # print("Start segmenting...")
                    segments = segment_image(img, palette)

                    new_palette = []
                    for color in palette:
                        if len(segments[color]) == 0:
                            segments.pop(color, None)
                        else:
                            new_palette.append(color)

                    # print("Finish segmenting...")
                    labels = role_labels(segments)
                    color_groups = [ColorGroup(img_num, segments, color, width, height, labels[color]) for color in new_palette]
                    all_color_groups += color_groups
                    pickle.dump(color_groups, pf, protocol=4)

            print("Finished all color groups...")
            
    else:
        all_color_groups = []
        with open(pickle_file, 'rb') as pf:
            while 1:
                try:
                    all_color_groups += (pickle.load(pf))
                except EOFError:
                    break
        print(len(all_color_groups))

        print("Start training...")
        spatial_properties = [np.array(x.spatial_property) for x in all_color_groups]
        l_values = [x.color_property[0] for x in all_color_groups]
        s_values = [x.color_property[1] for x in all_color_groups]

        lightness_histogram = Histogram()
        lacc = lightness_histogram.train(spatial_properties, l_values)
        print("Lightness Histogram done...")

        saturation_histogram = Histogram()
        sacc = saturation_histogram.train(spatial_properties, s_values)
        print("Saturation Histogram done...\n")

        segment_spatial_properties = [np.array(x.spatial_property) for x in cg.color_groups for cg in all_color_groups] 
        segment_l_values = [x.color_property[0] for x in cg.color_groups for cg in all_color_groups]
        segment_s_values = [x.color_property[1] for x in cg.color_groups for cg in all_color_groups]
        
        segment_lightness_histogram = Histogram()
        segment_lacc = segment_lightness_histogram.train(segment_spatial_properties, segment_l_values)
        print("Segment Lightness Histogram done...")

        segment_saturation_histogram = Histogram()
        segment_sacc = segment_saturation_histogram.train(segment_spatial_properties, segment_s_values)
        print("Segment Saturation Histogram done...\n")

        for i in color_group_tests:
            cg = all_color_groups[i]
            print('Results for color group color:', cg.color)

            lightness_score, lp = score_grp(lightness_histogram, cg.spatial_property, cg.color_property[0], cg.area)
            print("Lightness:")
            print("Prob:", np.round(lp, decimals=2))
            print("Score:", np.round(lightness_score, decimals=2))

            print()
            
            saturation_score, sp = score_grp(saturation_histogram, cg.spatial_property, cg.color_property[1], cg.area)
            print('Saturation:')
            print("Prob:", np.round(sp, decimals=2))
            print("Score:", np.round(saturation_score, decimals=2))
            print()
            print()

            cs = cg.color_groups[0]
            print('Results for color segment color:', cg.color)

            segment_lightness_score, seg_lp = score_seg(segment_lightness_histogram, cs.spatial_property, cs.color_property[0], cs.area)
            print("Lightness:")
            print("Prob:", np.round(seg_lp, decimals=2))
            print("Score:", np.round(segment_lightness_score, decimals=2))

            print()
            
            segment_saturation_score, seg_sp = score_seg(segment_saturation_histogram, cs.spatial_property, cs.color_property[1], cg.area)
            print('Saturation:')
            print("Prob:", np.round(seg_sp, decimals=2))
            print("Score:", np.round(segment_saturation_score, decimals=2))
            print()
            print()

if __name__ == '__main__':
    main()