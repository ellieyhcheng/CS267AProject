# import pymc3 as pm
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.neighbors import KernelDensity
from segment import segment_image, get_color_groups, hex2rgb, enclosure_strengths
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

def hex2lab(color):
    r,g,b = hex2rgb(color)
    rgb = [[[r/255,g/255,b/255]]]
    return rgb2lab(rgb)[0][0]

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

# Properties of color relationship between two adjacent segments
def perceptual_diff(c1, c2):
    l1 = c1[0]
    a1 = c1[1]
    b1 = c1[2]

    l2 = c2[0]
    a2 = c2[1]
    b2 = c2[2]
    
    return np.sqrt(l1**2+a1**2+b1**2) - np.sqrt(l2**2+a2**2+b2**2)

def relative_lightness(c1, c2):
    l1 = c1[0]
    a1 = c1[1]
    b1 = c1[2]

    l2 = c2[0]
    a2 = c2[1]
    b2 = c2[2]

    return l1 - l2

def relative_saturation(c1, c2):
    return saturation(c1) - saturation(c2)

def chromatic_difference(c1, c2):
    l1 = c1[0]
    a1 = c1[1]
    b1 = c1[2]

    l2 = c2[0]
    a2 = c2[1]
    b2 = c2[2]

    dasqr = (a1 - a2)**2
    dbsqr = (b1 - b2)**2
    dlsqr = (l1 - l2)**2
    return (dasqr + dbsqr) / (dasqr + dbsqr + dlsqr)

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

class Pattern:
    def __init__(self, img_num, img, palette):
        height,width,d = img.shape

        segments,px2id,adjacency = segment_image(img, palette)
        enc_str = enclosure_strengths(px2id, len(adjacency))

        new_palette = []
        for color in palette:
            if len(segments[color]) == 0:
                segments.pop(color, None)
            else:
                new_palette.append(color)

        self.img_num = img_num
        self.palette = new_palette
        labels = role_labels(segments)
        self.color_groups = [ColorGroup(segments, color, width, height, labels[color], px2id, enc_str) for color in new_palette]

class ColorGroup:
    # u get a map from color to list of segments where each segment is jsut a list of coordinates of that segment
    # like {'FFFFFF' : [[(0,0),(0,1)...] , [(10,10),(10,11)...], ...],
    #       '123455' : [[]]}
    # so u just pass in each of the colors and then for each color, u make a bunch of ColorGroupSegments
    # its a map from color to a list of lists
    def __init__(self, segments, color, pattern_width, pattern_height, label, px2id, enc_str):
        self.img_num = img_num
        lab = hex2lab(color)
        light = lightness(lab)
        sat = saturation(lab)
        self.color_property = (light, sat)
        
        self.color = color # in hex
        color_segments = segments[color]
        self.area = sum([len(s) for s in color_segments])

        relative_size_to_pattern = relative_size(segments,color)[0]
        relative_size_to_max_group =relative_size(segments,color)[1]
        seg_spread = segment_spread(color_segments)
        label = label
        
        self.color_segments = []
        for seg in color_segments:
            cs_id = px2id[seg[0][0]][seg[0][1]]
            enc = enc_str[cs_id]
            self.color_segments.append(ColorGroupSegment(cs_id, seg, segments,color, pattern_width, pattern_height, label, enc))

        # Segment size statistics
        min_segment_size, max_segment_size, mean_segment_size, std_segment_size = segment_size_stats(color_segments)
        
        self.spatial_property = np.array([relative_size_to_pattern, relative_size_to_max_group, label, min_segment_size, max_segment_size, mean_segment_size, std_segment_size])
        # self.spatial_property = np.concatenate((self.spatial_property, self.seg_spread))
        #TODO: How use a matrix as a feature bruh

# Individual segments within a color group
class ColorGroupSegment: 
    def __init__(self, cs_id, segment, segments, color, pattern_width, pattern_height,label, enclosure_strength):

        self.id = cs_id
        self.color = color
        light = lightness(hex2lab(color))
        sat = saturation(hex2lab(color))
        self.color_property = (light, sat)
        self.area = sum([len(s) for s in segment])
        self.enclosure_strength = enclosure_strength

        relative_size = relative_size_ind(segment,segments)
        num_neighbors = normalized_discrete_compactness(pattern_width, pattern_height, segment)
        elon = elongation(segment)
        label = label # foreground = 1, background = 0
        cent = centrality(pattern_width, pattern_height, segment)

        self.spacial_property = (relative_size, num_neighbors, elon, label, cent)

# phi
def score_grp(histogram, spatial_property, color_property, area):
    prob_dist = histogram.get_prob_distribution(spatial_property)
    p = prob_dist(color_property)
    return np.log(p) * area, p

def score_seg(histogram, spatial_property, color_property, area):
    prob_dist = histogram.get_prob_distribution(spatial_property)
    p = prob_dist(color_property)
    return np.log(p) * area, p

def score_adj(h, sp12, cp12, enc_str):
    prob_dist = h.get_prob_distribution(sp12)
    p = prob_dist(cp12)
    return np.log(p) * enc_str, p

def main():
    processing = True
    count = 0
    end = 300
    pickle_file = 'patterns.pickle'
    test_idx = np.arange(0,1,1)

    # list of list of colorgroups
    if processing:
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

                    patt = Pattern(img_num, img, palette)
                    
                    pickle.dump(patt, pf, protocol=4)

            print("Finished all color groups...")
            
    else:
        all_patterns = []
        with open(pickle_file, 'rb') as pf:
            while 1:
                try:
                    all_patterns += (pickle.load(pf))
                except EOFError:
                    break
        print(len(all_patterns))

        all_color_groups = [cg for cg in patt.color_groups for cg in all_patterns]

        print("Start training...")

        # Unary
        spatial_properties = [np.array(x.spatial_property) for x in all_color_groups]
        l_values = [x.color_property[0] for x in all_color_groups]
        s_values = [x.color_property[1] for x in all_color_groups]

        lightness_histogram = Histogram()
        lacc = lightness_histogram.train(spatial_properties, l_values)
        print("Lightness Histogram done...")

        saturation_histogram = Histogram()
        sacc = saturation_histogram.train(spatial_properties, s_values)
        print("Saturation Histogram done...\n")

        segment_spatial_properties = [np.array(x.spatial_property) for x in cg.color_segments for cg in all_color_groups] 
        segment_l_values = [x.color_property[0] for x in cg.color_segments for cg in all_color_groups]
        segment_s_values = [x.color_property[1] for x in cg.color_segments for cg in all_color_groups]
        
        segment_lightness_histogram = Histogram()
        segment_lacc = segment_lightness_histogram.train(segment_spatial_properties, segment_l_values)
        print("Segment Lightness Histogram done...")

        segment_saturation_histogram = Histogram()
        segment_sacc = segment_saturation_histogram.train(segment_spatial_properties, segment_s_values)
        print("Segment Saturation Histogram done...\n")

        # Pairwise 
        adj_spatial_properties = []
        per_diff = []
        rel_light = []
        rel_sat = []
        chrom_diff = []
        
        for patt in all_patterns:
            for cg in patt.color_groups:
                for seg in cg.color_segments:
                    adj_ids = [x if seg.enclosure_strength[x] != 0 for x in range(seg.enclosure_strength)]

                    for cg1 in patt.color_groups:
                        if cg.color != cg1.color:
                            for seg1 in cg1.color_segments:
                                if seg1.id in adj_ids:
                                    adj_spatial_properties.append(np.concatenate((np.array(seg.spatial_property)), np.array(seg1.spatial_property)))
                                    per_diff.append(perceptual_diff(seg.color, seg1.color))
                                    rel_light.append(relative_lightness(seg.color, seg1.color))
                                    rel_sat.append(relative_saturation(seg.color, seg1.color))
                                    chrom_diff.append(chromatic_difference(seg.color, seg1.color))

        per_diff_histogram = Histogram()
        per_diff_acc = per_diff_histogram.train(adj_spatial_properties, per_diff)
        print("Pereceptual Difference Histogram done...")
        
        rel_light_histogram = Histogram()
        rel_light_acc = rel_light_histogram.train(adj_spatial_properties, rel_light)
        print("Relative Lightness Histogram done...")

        rel_sat_histogram = Histogram()
        rel_sat_acc = rel_sat_histogram.train(adj_spatial_properties, rel_sat)
        print("Relative Saturation Histogram done...")

        chrom_diff_histogram = Histogram()
        chrom_diff_acc = chrom_diff_histogram.train(adj_spatial_properties, chrom_diff)
        print("Chromatic Difference Histogram done...\n")

        for i in test_idx:
            patt = all_patterns[i]
            cg = patt.color_groups[0]
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

            cs = cg.color_segments[0]
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

            adj_ids = [x if cs.enclosure_strength[x] != 0 for x in range(cs.enclosure_strength)]
            
            for cg1 in cg:
                for seg1 in cg1.color_segments:
                    if seg1.id == adj_ids[0]:
                        sp12 = np.concatenate((np.array(cs.spatial_property), np.array(seg1.spatial_property)))
                        enc = cs.enclosure_strength[seg1.id]

                        per_diff_score, per_diff_p = score_adj(per_diff_histogram, sp12, perceptual_diff(cs.color, seg1.color), enc)
                        print('Perceptual Difference:')
                        print('Prob:', np.round(per_diff_p, decimals=2))
                        print('Score:', np.round(per_diff_score, decimals=2))
                        print()
                        
                        rel_light_score, rel_light_p = score_adj(rel_light_histogram, sp12, perceptual_diff(cs.color, seg1.color), enc)
                        print('Perceptual Difference:')
                        print('Prob:', np.round(rel_light_p, decimals=2))
                        print('Score:', np.round(rel_light_score, decimals=2))
                        print()
                        
                        rel_sat_score, rel_sat_p = score_adj(rel_sat_histogram, sp12, perceptual_diff(cs.color, seg1.color), enc)
                        print('Perceptual Difference:')
                        print('Prob:', np.round(rel_sat_p, decimals=2))
                        print('Score:', np.round(rel_sat_score, decimals=2))
                        print()

                        chrom_diff_score, chrom_diff_p = score_adj(chrom_diff_histogram, sp12, perceptual_diff(cs.color, seg1.color), enc)
                        print('Perceptual Difference:')
                        print('Prob:', np.round(chrom_diff_p, decimals=2))
                        print('Score:', np.round(chrom_diff_score, decimals=2))
                        print()

if __name__ == '__main__':
    main()