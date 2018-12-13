import cv2
import numpy as np
from matplotlib import pyplot as plt
import random
MIN_MATCH_COUNT = 10


img1color = cv2.imread("E:\\data\\img1.jpg", 1)
img2color = cv2.imread("E:\\data\\img2.jpg", 1)
img1 = cv2.cvtColor(img1color, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2color, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()

(kps1, descs1) = sift.detectAndCompute(img1, None)
(kps2, descs2) = sift.detectAndCompute(img2, None)

sift_img1 = cv2.drawKeypoints(img1color, kps1, color=(100, 0, 100), outImage=np.array([]))
sift_img2 = cv2.drawKeypoints(img2color, kps2, color=(100, 0, 100), outImage=np.array([]))


cv2.imwrite("E:\\data\\sift1.jpg", sift_img1)
cv2.imwrite("E:\\data\\sift2.jpg", sift_img2)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(descs1, descs2, k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)

if (len(good) > MIN_MATCH_COUNT):
    src_pts = np.float32([kps1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kps2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
    matchesMask1 = (mask.ravel()==1).tolist()

    list_t = []
    for i in range(0, 10):
        xx = random.randint(0, 244)
        list_t.append(xx)


    for i in range(0,len(matchesMask1)):
        if (i in list_t):
            continue
        else:
            matchesMask1[i]=False
    
    h, w = img1.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, H)

#This warp function helps in warping two images.     
def warpImages(img1, img2, H):
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]

    list_of_points_1 = np.float32([[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
    temp_points = np.float32([[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]]).reshape(-1, 1, 2)

    list_of_points_2 = cv2.perspectiveTransform(temp_points, H)
    list_of_points = np.concatenate((list_of_points_1, list_of_points_2), axis=0)

    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)

    translation_dist = [-x_min, -y_min]
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

    output_img = cv2.warpPerspective(img2, H_translation.dot(H), (x_max - x_min, y_max - y_min))
    output_img[translation_dist[1]:rows1 + translation_dist[1], translation_dist[0]:cols1 + translation_dist[0]] = img1
    return output_img


warpimg = warpImages(img2color, img1color, H)
cv2.imwrite("E:\\data\\warp.jpg", warpimg)

print("The homography matrix is")
print(H)

draw_params = dict(matchColor=(100, 180, 130),
                   singlePointColor=None,
                   matchesMask=matchesMask,  
                   flags=2)
draw_params_2 = dict(matchColor=(80, 100, 255),
                   singlePointColor=None,
                   matchesMask=matchesMask1,  
                   flags=2)

img3 = cv2.drawMatches(img1color, kps1, img2color, kps2, good, None, **draw_params)

cv2.imwrite("E:\\data\\matches1_knn.jpg", img3)

img4 = cv2.drawMatches(img1color, kps1, img2color, kps2, good, None, **draw_params_2)

cv2.imwrite("E:\\data\\matches2.jpg", img4)

