import numpy as np
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.neighbors import KernelDensity
from sklearn.metrics.pairwise import euclidean_distances
from segment import segment_image, get_color_groups, hex2rgb, enclosure_strengths, rgb2hex, get_color
from skimage.color import rgb2lab, rgb2hsv
import csv
from PIL import Image
import os
import pickle
import matplotlib.pyplot as plt
import time
import datetime
from math import cos, sin,ceil
import cv2
from sklearn.cluster import KMeans
import argparse

prefix = 'good_'
pickle_file = '{}patterns.pickle'.format(prefix)
histogram_file = '{}histogram.pickle'.format(prefix)
weights_file = '{}weights.pickle'.format(prefix)

def hex2lab(color):
    r,g,b = hex2rgb(color)
    rgb = [[[r/255,g/255,b/255]]]
    return rgb2lab(rgb)[0][0]/100.0

def lightness(color):
    # L,a,b = color
    L = color[0]
    a = color[1]
    b = color[2]
    return L

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
    kmeans = KMeans(n_clusters=10).fit(color_property_values)
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
    return area/total_area, area/max_group_area

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
    
    return np.sqrt((l1 - l2)**2 + (a1 - a2)**2 + (b1-b2)**2)
    # return abs(np.sqrt(l1**2+a1**2+b1**2) - np.sqrt(l2**2+a2**2+b2**2))

def relative_lightness(c1, c2):
    l1 = c1[0]
    a1 = c1[1]
    b1 = c1[2]

    l2 = c2[0]
    a2 = c2[1]
    b2 = c2[2]

    return abs((l1 - l2))

def relative_saturation(c1, c2):
    return abs(saturation(c1) - saturation(c2))

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

    denom = (dasqr + dbsqr + dlsqr)
    if denom == 0:
        return 0

    return (dasqr + dbsqr) / denom

# colors given in hex
def compat_features(c1, c2, c3, c4, c5):
    colors = [c1, c2, c3, c4, c5]

    rgb = np.empty((5, 3))
    lab = np.empty((5, 3))
    hsv = np.empty((5, 3))
    chsv = np.empty((5, 3))

    for idx,c in enumerate(colors):
        r,g,b = hex2rgb(c)
        rgb1 = np.array([r/255, g/255, b/255])
        lab1 = hex2lab(c)
        hsv1 = rgb2hsv([[rgb1]])[0][0]
        chsv1 = [hsv1[1] * cos(hsv1[0]), hsv1[0] * sin(hsv1[0]), hsv1[2]]

        rgb[idx] = rgb1
        lab[idx] = lab1
        hsv[idx] = hsv1
        chsv[idx] = chsv1

    sort_idx = np.argsort(lab[:,0])
    rgb_sorted = rgb[sort_idx].flatten()
    lab_sorted = lab[sort_idx].flatten()
    hsv_sorted = hsv[sort_idx].flatten()
    chsv_sorted = chsv[sort_idx].flatten()

    rgb_diff = np.zeros((3, 4))
    lab_diff = np.zeros((3, 4))
    hsv_diff = np.zeros((3, 4))
    chsv_diff = np.zeros((3, 4))

    for i in range(1,5):
        rgb_diff[0, i-1] = rgb[i, 0] - rgb[i - 1, 0]
        rgb_diff[1, i-1] = rgb[i, 1] - rgb[i - 1, 1]
        rgb_diff[2, i-1] = rgb[i, 2] - rgb[i - 1, 2]

        lab_diff[0, i-1] = lab[i, 0] - lab[i - 1, 0]
        lab_diff[1, i-1] = lab[i, 1] - lab[i - 1, 1]
        lab_diff[2, i-1] = lab[i, 2] - lab[i - 1, 2]

        minSatVal = min(np.concatenate((hsv[i-1:i, 1], hsv[i-1:i, 2])))
        if minSatVal >= 0.2:
            pts = np.sort([hsv[i, 1], hsv[i-1, 1]])
            hsv_diff[0, i-1] = min(pts[1] - pts[0], 1-(pts[1] - pts[0]))
        hsv_diff[1, i-1] = hsv[i, 1] - hsv[i - 1, 1]
        hsv_diff[2, i-1] = hsv[i, 2] - hsv[i - 1, 2]

        chsv_diff[0, i-1] = chsv[i, 0] - chsv[i - 1, 0]
        chsv_diff[1, i-1] = chsv[i, 1] - chsv[i - 1, 1]
        chsv_diff[2, i-1] = chsv[i, 2] - chsv[i - 1, 2]
    
    sort_rgb_diff = np.concatenate((np.sort(rgb_diff[0]), np.sort(rgb_diff[1]), np.sort(rgb_diff[2])))
    sort_lab_diff = np.concatenate((np.sort(lab_diff[0]), np.sort(lab_diff[1]), np.sort(lab_diff[2])))
    sort_hsv_diff = np.concatenate((np.sort(hsv_diff[0]), np.sort(hsv_diff[1]), np.sort(hsv_diff[2])))
    sort_chsv_diff = np.concatenate((np.sort(chsv_diff[0]), np.sort(chsv_diff[1]), np.sort(chsv_diff[2])))

    rgb_mean = np.mean(rgb, axis=0)
    lab_mean = np.mean(lab, axis=0)
    hsv_mean = np.mean(hsv, axis=0)
    chsv_mean = np.mean(chsv, axis=0)

    rgb_std = np.std(rgb, axis=0)
    lab_std = np.std(lab, axis=0)
    hsv_std = np.std(hsv, axis=0)
    chsv_std = np.std(chsv, axis=0)

    rgb_median = np.median(rgb, axis=0)
    lab_median = np.median(lab, axis=0)
    hsv_median = np.median(hsv, axis=0)
    chsv_median = np.median(chsv, axis=0)

    rgb_max = np.amax(rgb, axis=0)
    lab_max = np.amax(lab, axis=0)
    hsv_max = np.amax(hsv, axis=0)
    chsv_max = np.amax(chsv, axis=0)

    rgb_min = np.amin(rgb, axis=0)
    lab_min = np.amin(lab, axis=0)
    hsv_min = np.amin(hsv, axis=0)
    chsv_min = np.amin(chsv, axis=0)

    rgb_range = rgb_max - rgb_min
    lab_range = lab_max - lab_min
    hsv_range = hsv_max - hsv_min
    chsv_range = chsv_max - chsv_min

    return np.concatenate((
        chsv.flatten(),
        chsv_sorted.flatten(),
        chsv_diff.flatten(),
        sort_chsv_diff.flatten(),
        chsv_mean.flatten(),
        chsv_std.flatten(),
        chsv_median.flatten(),
        chsv_max.flatten(),
        chsv_min.flatten(),
        chsv_range.flatten(),
        lab.flatten(),
        lab_sorted.flatten(),
        lab_diff.flatten(),
        sort_lab_diff.flatten(),
        lab_mean.flatten(),
        lab_std.flatten(),
        lab_median.flatten(),
        lab_max.flatten(),
        lab_min.flatten(),
        lab_range.flatten(),
        hsv.flatten(),
        hsv_sorted.flatten(),
        hsv_diff.flatten(),
        sort_hsv_diff.flatten(),
        hsv_mean.flatten(),
        hsv_std.flatten(),
        hsv_median.flatten(),
        hsv_max.flatten(),
        hsv_min.flatten(),
        hsv_range.flatten(),
        rgb.flatten(),
        rgb_sorted.flatten(),
        rgb_diff.flatten(),
        sort_rgb_diff.flatten(),
        rgb_mean.flatten(),
        rgb_std.flatten(),
        rgb_median.flatten(),
        rgb_max.flatten(),
        rgb_min.flatten(),
        rgb_range.flatten()
    ))

class Histogram:
    def __init__(self):
        pass

    def train(self, spatial_properties, color_property):
        self.spatial_property_scaler = StandardScaler().fit(spatial_properties)
        x_train = np.array(self.spatial_property_scaler.transform(spatial_properties))
        color_property = np.array(color_property).reshape(-1, 1)
        self.color_property_scaler = MinMaxScaler().fit(color_property)
        color_property = self.color_property_scaler.transform(color_property)
        self.kmeans = discretize_color_property(color_property)

        # color_property = np.array(color_property)
        # y_train = np.array((color_property.shape[0],10))
        # for idx, cp in enumerate(color_property):
        #     y_train[idx][cp]
        y_train = np.array([getbin(self.kmeans, cp) for cp in color_property])
        self.clf = LogisticRegression(multi_class='multinomial', max_iter=5000)
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
        dists = euclidean_distances(color_property_values.reshape(-1,1))
        bw = dists.mean()
        # bw = abs(color_property_values[1] - color_property_values[0])
        kde = KernelDensity(bandwidth=bw, kernel='gaussian').fit(data[:,0].reshape(-1,1), sample_weight=data[:,1])
        # x = np.linspace(0,1,50)
        # log_dens = kde.score_samples(x.reshape(-1,1))
        def get_prob(color_property_value):
            v = self.color_property_scaler.transform([[color_property_value]])
            log_dens = kde.score_samples(v)
            return np.exp(log_dens)[0]
        return get_prob

class Pattern:
    def __init__(self, img_num, width, height, segments, px2id, enc_str, palette, rating):

        self.img_num = img_num
        self.palette = palette
        self.rating = rating
        labels = role_labels(segments)

        self.color_groups = [ColorGroup(segments, color, width, height, labels[color], px2id, enc_str) for color in palette]
        self.color_groups.sort(key=lambda colorgroup: colorgroup.area)

class ColorGroup:
    # u get a map from color to list of segments where each segment is jsut a list of coordinates of that segment
    # like {'FFFFFF' : [[(0,0),(0,1)...] , [(10,10),(10,11)...], ...],
    #       '123455' : [[]]}
    # so u just pass in each of the colors and then for each color, u make a bunch of ColorGroupSegments
    # its a map from color to a list of lists
    def __init__(self, segments, color, pattern_width, pattern_height, label, px2id, enc_str):
        lab = hex2lab(color)

        self.color = color # in hex
        color_segments = segments[color]

        relative_size_to_pattern = relative_size(segments,color)[0]
        relative_size_to_max_group =relative_size(segments,color)[1]
        seg_spread = segment_spread(color_segments)
        label = label
        
        self.area = relative_size_to_pattern
        
        self.color_segments = []
        for seg in color_segments:
            cs_id = px2id[seg[0][0]][seg[0][1]]
            enc = enc_str[cs_id]
            enc_map = {}
            for i in range(len(enc)):
                if enc[i] != 0:
                    enc_map[i] = enc[i]
            
            self.color_segments.append(ColorGroupSegment(cs_id, seg, segments,color, pattern_width, pattern_height, label, enc_map))

        # Segment size statistics
        min_segment_size, max_segment_size, mean_segment_size, std_segment_size = segment_size_stats(color_segments)
        
        self.spatial_property = np.array([relative_size_to_pattern, relative_size_to_max_group, label, min_segment_size, max_segment_size, mean_segment_size, std_segment_size])
        # self.spatial_property = np.concatenate((self.spatial_property, self.seg_spread))
        #TODO: How use a matrix as a feature bruh

# Individual segments within a color group
class ColorGroupSegment: 
    def __init__(self, cs_id, segment, segments, color, pattern_width, pattern_height, label, enclosure_strength):

        self.id = cs_id
        self.color = color
        self.enclosure_strength = enclosure_strength

        relative_size_to_pattern, relative_size_to_group = relative_size_ind(segment,segments)
        self.area = relative_size_to_pattern

        num_neighbors = normalized_discrete_compactness(pattern_width, pattern_height, segment)
        elon = elongation(segment)
        label = label # foreground = 1, background = 0
        cent = centrality(pattern_width, pattern_height, segment)

        self.spatial_property = np.array([relative_size_to_pattern, relative_size_to_group, num_neighbors, elon, label, cent])

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

def score_cmp(model, palette):
    colors = palette.copy()[:5]
    while len(colors) < 5:
        colors.append(palette[-1])
    compat_f = compat_features(colors[0], colors[1], colors[2], colors[3], colors[4])
    p = model.predict([compat_f])[0]
    if p <= 0:
        return -200, 0
    return np.log(p/5), p

def train_weights(all_patterns):
    with open(histogram_file, 'rb') as hf:
        lightness_histogram = pickle.load(hf)
        saturation_histogram = pickle.load(hf)
        segment_lightness_histogram = pickle.load(hf)
        segment_saturation_histogram = pickle.load(hf)
        per_diff_histogram = pickle.load(hf)
        rel_light_histogram = pickle.load(hf)
        rel_sat_histogram = pickle.load(hf)
        chrom_diff_histogram = pickle.load(hf)
        compat_model = pickle.load(hf)

    # if checkpointing start with old weights
    # with open(weights_file, 'rb') as wf:
    #     weights = pickle.load(wf)

    weights = np.ones((9,))

    N = len(all_patterns)
    learning_rate = 4
    sample_k = 10

    cg_lightness_w = weights[0]
    cs_lightness_w = weights[1]
    cg_saturation_w = weights[2]
    cs_saturation_w = weights[3]
    adj_per_diff_w = weights[4]
    adj_rel_light_w = weights[5]
    adj_rel_sat_w = weights[6]
    adj_chrom_diff_w = weights[7]
    cmp_w = weights[8]
    
    for it in range(5):
        diff = np.zeros((9,))
        for pattern in all_patterns:
            # do MCMC for k steps to get c_hat
            c_hat = sample(weights, pattern, sample_k, pattern.palette)
            
            scores = np.zeros((2,9))

            for k, palette in enumerate([pattern.palette, c_hat]):
                compat_score, _ = score_cmp(compat_model, palette)
                scores[k][8] = compat_score

                for i, cg in enumerate(pattern.color_groups):
                    cur_color = palette[i]

                    lightness_score, _ = score_grp(lightness_histogram, cg.spatial_property, lightness(hex2lab(cur_color)), cg.area)
                    saturation_score, _ = score_grp(saturation_histogram, cg.spatial_property, saturation(hex2lab(cur_color)), cg.area)
                    sum_segment_lightness_score, sum_segment_saturation_score, sum_per_diff_score, sum_rel_light_score, sum_rel_sat_score, sum_chrom_diff_score = 0, 0, 0, 0, 0, 0

                    for cs in cg.color_segments:
                        if cs.area < 0.05:
                            continue
                        segment_lightness_score, _ = score_seg(segment_lightness_histogram, cs.spatial_property, lightness(hex2lab(cur_color)), cs.area)
                        segment_saturation_score, _ = score_seg(segment_saturation_histogram, cs.spatial_property, saturation(hex2lab(cur_color)), cs.area)
                        sum_segment_lightness_score += segment_lightness_score
                        sum_segment_saturation_score += segment_saturation_score

                        adj_ids = list(cs.enclosure_strength.keys())
                        # count = 0
                        # n_adj = 5

                        for j, cg1 in enumerate(pattern.color_groups):
                            # if count > n_adj:
                            #     break
                            adj_color = palette[j]
                            for seg1 in cg1.color_segments:
                                if seg1.area < 0.05:
                                    continue
                                # if count > n_adj:
                                #     break
                                if seg1.id in adj_ids:
                                    # count += 1
                                    sp12 = np.concatenate((cs.spatial_property, seg1.spatial_property))
                                    cp12 = perceptual_diff(hex2lab(cur_color), hex2lab(adj_color))
                                    enc = cs.enclosure_strength[seg1.id]

                                    per_diff_score, _ = score_adj(per_diff_histogram, sp12, cp12, enc)
                                    sum_per_diff_score += per_diff_score
                                    rel_light_score, _ = score_adj(rel_light_histogram, sp12, relative_lightness(hex2lab(cur_color), hex2lab(adj_color)), enc)
                                    sum_rel_light_score += rel_light_score
                                    rel_sat_score, _ = score_adj(rel_sat_histogram, sp12, relative_saturation(hex2lab(cur_color), hex2lab(adj_color)), enc)
                                    sum_rel_sat_score += rel_sat_score
                                    chrom_diff_score, _ = score_adj(chrom_diff_histogram, sp12, chromatic_difference(hex2lab(cur_color), hex2lab(adj_color)), enc)
                                    sum_chrom_diff_score += chrom_diff_score

                    scores[k][0] += lightness_score
                    scores[k][1] += sum_segment_lightness_score
                    scores[k][2] += saturation_score
                    scores[k][3] += sum_segment_saturation_score
                    scores[k][4] += sum_per_diff_score
                    scores[k][5] += sum_rel_light_score
                    scores[k][6] += sum_rel_sat_score
                    scores[k][7] += sum_chrom_diff_score
        
            diff = np.add(diff, np.subtract(scores[0], scores[1]))
            # if abs(diff[8]) < 1:
            #     return weights
            
        print(it, np.round(diff, decimals=3))
        # We subtract because the derivatives point in direction of steepest ascent
        weights = np.maximum(np.add(weights, (diff / float(N)) * learning_rate), 0)
    
    return weights

# should return a probability distribution
# after training weights
# factor_graph -> function get_prob
def factor_graph(pattern):
    with open(histogram_file, 'rb') as hf:
        lightness_histogram = pickle.load(hf)
        saturation_histogram = pickle.load(hf)
        segment_lightness_histogram = pickle.load(hf)
        segment_saturation_histogram = pickle.load(hf)
        per_diff_histogram = pickle.load(hf)
        rel_light_histogram = pickle.load(hf)
        rel_sat_histogram = pickle.load(hf)
        chrom_diff_histogram = pickle.load(hf)
        compat_model = pickle.load(hf)

    def get_prob(weights, palette):
        # only 9 weights
        cg_lightness_w = weights[0]
        cs_lightness_w = weights[1]
        cg_saturation_w = weights[2]
        cs_saturation_w = weights[3]
        adj_per_diff_w = weights[4]
        adj_rel_light_w = weights[5]
        adj_rel_sat_w = weights[6]
        adj_chrom_diff_w = weights[7]
        cmp_w = weights[8]

        factor_product = 1

        for i, cg in enumerate(pattern.color_groups):
            cur_color = palette[i]

            lightness_score, lightness_p = score_grp(lightness_histogram, cg.spatial_property, lightness(hex2lab(cur_color)), cg.area)
            saturation_score, _ = score_grp(saturation_histogram, cg.spatial_property, saturation(hex2lab(cur_color)), cg.area)
            # print(lightness_score, lightness_p)

            sum_segment_lightness_score, sum_segment_saturation_score, sum_per_diff_score, sum_rel_light_score, sum_rel_sat_score, sum_chrom_diff_score = 0, 0, 0, 0, 0, 0

            for cs in cg.color_segments:
                if cs.area < 0.05:
                    continue
                segment_lightness_score, _ = score_seg(segment_lightness_histogram, cs.spatial_property, lightness(hex2lab(cur_color)), cs.area)
                segment_saturation_score, _ = score_seg(segment_saturation_histogram, cs.spatial_property, saturation(hex2lab(cur_color)), cs.area)
                sum_segment_lightness_score += segment_lightness_score
                sum_segment_saturation_score += segment_saturation_score

                adj_ids = list(cs.enclosure_strength.keys())

                for j, cg1 in enumerate(pattern.color_groups):
                    adj_color = palette[j]
                    for seg1 in cg1.color_segments:
                        if seg1.area < 0.05:
                            continue
                        if seg1.id in adj_ids:
                            sp12 = np.concatenate((cs.spatial_property, seg1.spatial_property))
                            enc = cs.enclosure_strength[seg1.id]

                            per_diff_score, _ = score_adj(per_diff_histogram, sp12, perceptual_diff(hex2lab(cur_color), hex2lab(adj_color)), enc)
                            sum_per_diff_score += per_diff_score
                            rel_light_score, _ = score_adj(rel_light_histogram, sp12, relative_lightness(hex2lab(cur_color), hex2lab(adj_color)), enc)
                            sum_rel_light_score += rel_light_score
                            rel_sat_score, _ = score_adj(rel_sat_histogram, sp12, relative_saturation(hex2lab(cur_color), hex2lab(adj_color)), enc)
                            sum_rel_sat_score += rel_sat_score
                            chrom_diff_score, _ = score_adj(chrom_diff_histogram, sp12, chromatic_difference(hex2lab(cur_color), hex2lab(adj_color)), enc)
                            sum_chrom_diff_score += chrom_diff_score

            lightness_unary_factor = np.exp((cg_lightness_w*lightness_score) + (cs_lightness_w*sum_segment_lightness_score))
            saturation_unary_factor = np.exp((cg_saturation_w*saturation_score) + (cs_saturation_w*sum_segment_saturation_score))
            per_diff_pairwise_factor = np.exp(adj_per_diff_w*sum_per_diff_score)
            rel_light_pairwise_factor = np.exp(adj_rel_light_w*sum_rel_light_score)
            rel_sat_pairwise_factor = np.exp(adj_rel_sat_w*sum_rel_sat_score)
            chrom_diff_pairwise_factor = np.exp(adj_chrom_diff_w*sum_chrom_diff_score)

            # print(lightness_unary_factor)
            # print(saturation_unary_factor)
            # print(per_diff_pairwise_factor)
            # print(rel_light_pairwise_factor)
            # print(rel_sat_pairwise_factor)
            # print(chrom_diff_pairwise_factor)

            factor_product *= lightness_unary_factor*saturation_unary_factor*per_diff_pairwise_factor*rel_light_pairwise_factor*rel_sat_pairwise_factor*chrom_diff_pairwise_factor

        compat_score, _ = score_cmp(compat_model, palette)
        compat_factor = np.exp(cmp_w*compat_score)
        
        factor_product *= compat_factor
        
        return factor_product

    return get_prob

def train_perturb(palette, temp, fixed=None):
    #fixed is list of palette indeces that should not be changed

    unfixed = []
    if fixed is None:
        unfixed = [i for i in range(len(palette))]
    else:
        for i in range(len(palette)):
            if i not in fixed:
                unfixed.append(i)

    num = len(unfixed)

    # perturb randomly chosen color
    rate = 1
    sigma = rate * temp

    color = np.random.randint(0,num)
    r,g,b = hex2rgb(palette[unfixed[color]])
    rp = np.random.normal(0,sigma)
    gp = np.random.normal(0,sigma)
    bp = np.random.normal(0,sigma)

    if r + rp > 255 or r + rp < 0:
        r = r - rp
    else:
        r = r + rp
    if g + gp > 255 or g + gp < 0:
        g = g - gp
    else:
        g = g + gp
    if b + bp > 255 or b + bp < 0:
        b = b - bp
    else:
        b = b + bp

    # r = int(r)
    # g = int(g)
    # b = int(b)

    r = max(0, min(255, int(r)))
    g = max(0, min(255, int(g)))
    b = max(0, min(255, int(b)))

    newhex = rgb2hex((r,g,b))
    palette[unfixed[color]] = newhex

    return palette    

def sample(weights, pattern, num_iters, start=None):
    # metropolis hasitngs:
    # propose new state, accept with probability proportional to model score
    # different temperatures
    # - perturb randomly chosen color ~ N(0,sigma) in RGB
    # - swap colors
    num_colors = len(pattern.palette)
    palette = start
    #initialize palette
    if palette is None:
        palette = []
        for i in range(num_colors):
            r = np.random.randint(0,256)
            g = np.random.randint(0,256)
            b = np.random.randint(0,256)
            palette.append(rgb2hex((r,g,b)))
        
    get_prob = factor_graph(pattern)
    
    temp = 5
    for i in range (num_iters):
        prop = perturb(palette.copy(), temp)
        denom = get_prob(weights,palette)
        if denom == 0:
            palette = prop
            continue
        acceptance = get_prob(weights,prop)/denom
        u = np.random.rand()
        if u <= acceptance:
            palette = prop

    return palette

def find_good_images(weights, pattern, num_iters, start_palette=None,fixed=None):
    get_prob = factor_graph(pattern)

    num_colors = len(pattern.palette)

    #initialize mcmc chain
    curr_palette = start_palette
    if curr_palette is None:
        curr_palette = []
        for j in range(num_colors):
            r = np.random.randint(0,256)
            g = np.random.randint(0,256)
            b = np.random.randint(0,256)
            curr_palette.append(rgb2hex((r,g,b)))

    high_temp = 42
    low_temp = 3

    #jump 10 times at each temperature and output highest probability image
    max_prob = 0
    best_palette = None
    for i in range(num_iters):
        curr_temp = low_temp
        if i % 20 < 10:
            curr_temp = high_temp
        prop = perturb(curr_palette.copy(), curr_temp, fixed)
        denom = get_prob(weights,curr_palette)
        if denom == 0:
            curr_palette = prop
            continue
        curr_prob = get_prob(weights,prop)

        acceptance = curr_prob/denom
        u = np.random.rand()
        if u <= acceptance:
            curr_palette = prop

        if curr_prob >= max_prob:
            max_prob = curr_prob
            best_palette = curr_palette
    return best_palette

def find_good_images_3(weights, pattern, num_iters, num_images, start_palette=None, fixed=None):
    good_images = [find_good_images(weights,pattern,num_iters,start_palette,fixed) for i in range(num_images)]
    return good_images
    
def perturb(palette, temp, fixed=None):
    #fixed is list of palette indeces that should not be changed

    unfixed = []
    if fixed is None:
        unfixed = [i for i in range(len(palette))]
    else:
        for i in range(len(palette)):
            if i not in fixed:
                unfixed.append(i)

    num = len(unfixed)
    
    #swap 2 colors
    color1 = np.random.randint(0,num)
    color2 = np.random.randint(0,num)
    oopsies = palette[unfixed[color1]]
    palette[unfixed[color1]] = palette[unfixed[color2]]
    palette[unfixed[color2]] = oopsies

    # perturb randomly chosen color
    rate = 1
    sigma = rate * temp

    color = np.random.randint(0,num)
    r,g,b = hex2rgb(palette[unfixed[color]])
    rp = np.random.normal(0,sigma)
    gp = np.random.normal(0,sigma)
    bp = np.random.normal(0,sigma)

    if r + rp > 255 or r + rp < 0:
        r = r - rp
    else:
        r = r + rp
    if g + gp > 255 or g + gp < 0:
        g = g - gp
    else:
        g = g + gp
    if b + bp > 255 or b + bp < 0:
        b = b - bp
    else:
        b = b + bp

    r = max(0, min(255, int(r)))
    g = max(0, min(255, int(g)))
    b = max(0, min(255, int(b)))

    newhex = rgb2hex((r,g,b))
    palette[unfixed[color]] = newhex
    return palette

def find_good_images_2(weights, pattern, num_iters, max_images, start_palette=None):
    get_prob = factor_graph(pattern)
    original_prob = get_prob(weights, pattern.palette)
    num_images = 0
    num_colors = len(pattern.palette)

    good_images = []

    #initialize mcmc chain
    curr_palette = start_palette
    if curr_palette is None:
        curr_palette = []
        for j in range(num_colors):
            r = np.random.randint(0,256)
            g = np.random.randint(0,256)
            b = np.random.randint(0,256)
            curr_palette.append(rgb2hex((r,g,b)))
    
    high_temp = 12
    low_temp = 1

    #jump 10 times at each temperature and output highest probability image
    max_prob = 0
    best_palette = None
    for i in range(num_iters):
        if num_images >= max_images:
            break
        curr_temp = high_temp
        if i % 20 < 10:
            curr_temp = high_temp
        else:
            curr_temp = low_temp

        prop = perturb(curr_palette.copy(), curr_temp)
        denom = get_prob(weights,curr_palette)
        if denom == 0:
            curr_palette = prop
            continue
        curr_prob = get_prob(weights,prop)
        print(curr_palette, denom)
        print(prop, curr_prob)
        print()
        acceptance = curr_prob/denom
        u = np.random.rand()
        if u <= acceptance:
            curr_palette = prop
            
        if curr_prob >= max_prob:
            max_prob = curr_prob
            best_palette = curr_palette

        # if abs(curr_prob - original_prob) < 0.03:
        #     # curr_palette = prob
        #     good_images.append(curr_palette)
        #     curr_palette = prop
        #     num_images += 1

        if i % 20 == 19:
            max_prob = 0
            good_images.append(curr_palette)
    
    return good_images

def recolor(img, original, new_palette):
    h,w,d = img.shape
    newimg = np.empty((h, w, 3))
    
    for i in range(h):
        for j in range(w):
            r,g,b=hex2rgb(new_palette[original.index(get_color(img, i,j, original))])

            newimg[i,j] = [r,g,b]
    
    return np.uint8(newimg)

def main(img_path, sample_iter, sample_num, k):
    training = False
    testing = False

    hist = False

    unary = True
    pairwise = True
    compat = True

    wei = True

    num_patterns = 1700
    # test_idx = np.random.choice(np.arange(0,45),5,False)
    test_idx = [x for x in range(0, 5)]

    if training or testing:
        print('Retrieving patterns')
        all_patterns = []
        with open(pickle_file, 'rb') as pf:
            i = 0
            while i < num_patterns:
                i += 1
                try:
                    all_patterns.append(pickle.load(pf))
                except EOFError:
                    break
        print('# Patterns:', len(all_patterns))
        print()

    if training:
        if hist:
            with open(histogram_file, 'wb') as hf:
                if unary:
                    all_color_groups = [cg for patt in all_patterns for cg in patt.color_groups]
                    spatial_properties = [x.spatial_property for x in all_color_groups]
                    l_values = [lightness(hex2lab(x.color)) for x in all_color_groups]
                    s_values = [saturation(hex2lab(x.color)) for x in all_color_groups]

                    print('--- Unary Training ---')
                    lightness_histogram = Histogram()
                    lacc = lightness_histogram.train(spatial_properties, l_values)
                    print("Lightness Histogram done...")

                    saturation_histogram = Histogram()
                    sacc = saturation_histogram.train(spatial_properties, s_values)
                    print("Saturation Histogram done...\n")

                    segment_spatial_properties = [x.spatial_property for cg in all_color_groups for x in cg.color_segments if x.area >= 0.05] 
                    segment_l_values = [lightness(hex2lab(x.color)) for cg in all_color_groups for x in cg.color_segments if x.area >= 0.05]
                    segment_s_values = [saturation(hex2lab(x.color)) for cg in all_color_groups for x in cg.color_segments if x.area >= 0.05]
                    
                    segment_lightness_histogram = Histogram()
                    segment_lacc = segment_lightness_histogram.train(segment_spatial_properties, segment_l_values)
                    print("Segment Lightness Histogram done...")

                    segment_saturation_histogram = Histogram()
                    segment_sacc = segment_saturation_histogram.train(segment_spatial_properties, segment_s_values)
                    print("Segment Saturation Histogram done...\n")

                    pickle.dump(lightness_histogram, hf, protocol=4)
                    pickle.dump(saturation_histogram, hf, protocol=4)
                    pickle.dump(segment_lightness_histogram, hf, protocol=4)
                    pickle.dump(segment_saturation_histogram, hf, protocol=4)

                if pairwise:
                    adj_spatial_properties = []
                    per_diff = []
                    rel_light = []
                    rel_sat = []
                    chrom_diff = []
                    
                    for patt in all_patterns:
                        for cg in patt.color_groups:
                            for seg in cg.color_segments:
                                if seg.area < 0.05:
                                    continue
                                adj_ids = list(seg.enclosure_strength.keys())

                                for cg1 in patt.color_groups:
                                        for seg1 in cg1.color_segments:
                                            if seg1.area < 0.05:
                                                continue
                                            if seg1.id in adj_ids:
                                                adj_spatial_properties.append(np.concatenate((seg.spatial_property, seg1.spatial_property)))
                                                per_diff.append(perceptual_diff(hex2lab(seg.color), hex2lab(seg1.color)))
                                                rel_light.append(relative_lightness(hex2lab(seg.color), hex2lab(seg1.color)))
                                                rel_sat.append(relative_saturation(hex2lab(seg.color), hex2lab(seg1.color)))
                                                chrom_diff.append(chromatic_difference(hex2lab(seg.color), hex2lab(seg1.color)))

                    print('--- Pairwise Training ---')
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

                    pickle.dump(per_diff_histogram, hf, protocol=4)
                    pickle.dump(rel_light_histogram, hf, protocol=4)
                    pickle.dump(rel_sat_histogram, hf, protocol=4)
                    pickle.dump(chrom_diff_histogram, hf, protocol=4)

                if compat:
                    print('--- Compatibility Training ---')
                    all_compat_features = []
                    ratings = []

                    with open('compat_set.csv') as csvfile:
                        reader = csv.DictReader(csvfile)

                        for row in reader:
                            colors = row['palette'].split()

                            idx = 0
                            while len(colors) < 5:
                                colors = np.append(colors, colors[idx])
                                idx += 1
                            
                            all_compat_features.append(compat_features(colors[0], colors[1], colors[2], colors[3], colors[4]))
                            ratings.append(float(row['rating']))

                        compat_model = Lasso(alpha=0.001, max_iter=10000).fit(all_compat_features, ratings)
                        print('Compatibility model done...\n')
                    
                    pickle.dump(compat_model, hf, protocol=4)

        if wei:
            print('--- Weights Training ---')
            weights = train_weights(all_patterns)
            print(weights)

            with open(weights_file, 'wb') as wf:
                pickle.dump(weights, wf, protocol=4)
            
            print("Weights training done...\n")
    else:
        with open(histogram_file, 'rb') as hf:
            lightness_histogram = pickle.load(hf)
            saturation_histogram = pickle.load(hf)
            segment_lightness_histogram = pickle.load(hf)
            segment_saturation_histogram = pickle.load(hf)
            per_diff_histogram = pickle.load(hf)
            rel_light_histogram = pickle.load(hf)
            rel_sat_histogram = pickle.load(hf)
            chrom_diff_histogram = pickle.load(hf)
            compat_model = pickle.load(hf)

        with open(weights_file, 'rb') as wf:
            weights = pickle.load(wf)

        # best weights
        # weights = [
        #     0.36275322 
        #     0.59691094 
        #     0.40329731 
        #     0.56617838 
        #     0.98325662 
        #     0.98577912
        #     0.98680531 
        #     0.99646319 
        #     2.71606898
        # ]

        if testing:
            for i in test_idx:
                pattern = all_patterns[i]
                print("Image ", pattern.img_num) 

                old_palette = pattern.palette
                
                while len(old_palette) < 5:
                    old_palette.append(pattern.palette[-1])

                testimg_file = os.path.join(img_folder, str(pattern.img_num)+'.png')
                testimg = Image.open(testimg_file)
                testimg = testimg.convert('RGB')
                ori_w, ori_h = testimg.size
                if ori_h or ori_h > 320:
                    resize_ratio = min(200/ori_w, 200/ori_h)
                    newsize = (int(resize_ratio*ori_w),int(resize_ratio*ori_h) )
                    testimg = testimg.resize(newsize) 
                testimg = np.array(testimg)
                
                num = sample_num
                new_palettes = np.array(find_good_images_3(weights, pattern, sample_iter, num))
                
                get_prob = factor_graph(pattern)

                probs = [get_prob(weights, p) for p in new_palettes]
                rin = np.argsort(np.array(probs))[::-1]
                # # rin = np.random.randint(0, num)
                new_palettes = new_palettes[rin]

                for new_palette in new_palettes:
                    new_img = recolor(testimg, old_palette, new_palette)
                    im = Image.fromarray(new_img)
                    im.save("results/{}_{}.png".format(pattern.img_num,new_palette[0]))

        else:
            print('Preprocessing {}...'.format(img_path))
            img_name = img_path.split('/')[-1].split('.')[0]
            img = Image.open(img_path)
            img = img.convert('RGB')
            ori_w, ori_h = img.size
            if ori_h or ori_h > 320:
                resize_ratio = min(200/ori_w, 200/ori_h)
                newsize = (int(resize_ratio*ori_w),int(resize_ratio*ori_h) )
                img = img.resize(newsize) 
            img = np.array(img)

            height,width,d = img.shape

            vectorized = np.float32(img.reshape((-1,3)))

            attempts = 10
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            ret,label,center = cv2.kmeans(vectorized,k,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
            center = np.uint8(center)

            res = center[label.flatten()]
            
            result_image = res.reshape((img.shape))

            palette = []
            for rgb in center:
                r,g,b = rgb[0],rgb[1],rgb[2]
                palette.append(rgb2hex((r,g,b)))
            
            segments,px2id,adjacency = segment_image(result_image, palette)

            if len(adjacency) > 5000:
                print('ERROR: {} is too many segments. Max of 5000. Try another image.'.format(len(adjacency)))
                return
            
            enc_str = enclosure_strengths(px2id, len(adjacency), adjacency)
            
            old_palette = []
            for color in palette:
                if len(segments[color]) == 0:
                    segments.pop(color, None)
                else:
                    old_palette.append(color)

            patt = Pattern(0, width, height, segments, px2id, enc_str, old_palette, 0)
            # print(old_palette)

            print('Sampling new colors...')
            new_palettes = find_good_images_3(weights, patt, sample_iter, sample_num)[::-1]

            if not os.path.exists('results'):
                os.mkdir('results')

            for palette in new_palettes:
                # print(' '.join(palette))
                new_img = recolor(img, old_palette, palette)
                im = Image.fromarray(new_img)
                
                name = os.path.join("results", "{}_{}.png".format(img_name,palette[0]))
                file_count = 1
                while os.path.exists(name):
                    name = os.path.join("results", "{}_{}_{}.png".format(img_name,palette[0], file_count))
                    file_count += 1
                im.save(name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', help='path to input image')
    parser.add_argument('--iter', help='number of sample interations', default=200, type=int)
    parser.add_argument('--num', help='number of graphics to generate', default=3, type=int)
    parser.add_argument('--clusters', help='number of clusters to segment the image', default=5, type=int)
    args = parser.parse_args()

    if not os.path.exists(args.file_path):
        print('ERROR: {} doesn\'t exist'.format(args.file_path))
        exit(2)
    
    if args.iter <= 0 or args.num <= 0 or args.clusters <= 0:
        print('ERROR: Postive args required')
        exit(2)
    
    if args.clusters > 5:
        print('ERROR: Max of 5 clusters allowed')
        exit(2)

    main(img_path=args.file_path, sample_iter=args.iter, sample_num=args.num, k=args.clusters)