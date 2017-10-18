import cv2
from random import randint
import numpy as np
from matplotlib import pyplot as plt
#
# img = cv2.imread('image.jpg',0)
# img2 = img.copy()
# template = cv2.imread('image1.jpg',0)
# w, h = template.shape[::-1]
#
# # All the 6 methods for comparison in a list
# methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED',
#             'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
#
# for meth in methods:
#     img = img2.copy()
#     method = eval(meth)
#
#     # Apply template Matching
#     res = cv2.matchTemplate(img,template,method)
#     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
#
#     # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
#     if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
#         top_left = min_loc
#     else:
#         top_left = max_loc
#     bottom_right = (top_left[0] + w, top_left[1] + h)
#
#     cv2.rectangle(img,top_left, bottom_right, 255, 2)
#
#     #plt.subplot(121),plt.imshow(res,cmap = 'gray')
#     #plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
#     plt.imshow(img,cmap = 'gray')
#     plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
#     plt.suptitle(meth)
#     plt.show()


MIN_MATCH_COUNT = 10
img1 = cv2.imread('images/bh3/1.jpg',0) # queryImage
#img2 = cv2.imread('image2.jpg',0) # trainImage
img2=cv2.imread('imagelnm.jpg',0)




def check(img1,img2):
    k = 5
    avggood = 0
    for i in range(k):
        w, h = img2.shape
        # print w,h
        x = randint(0, w - 300)
        y = randint(0, h - 300)
        imgtry = img2[x:x + 300, y:y + 300]

        # plt.imshow(imgtry), plt.show()
        # Initiate SIFT detector
        try:
            sift = cv2.xfeatures2d.SIFT_create()

            # find the keypoints and descriptors with SIFT
            kp1, des1 = sift.detectAndCompute(img1, None)
            kp2, des2 = sift.detectAndCompute(imgtry, None)

            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            # print des1,des2
            flann = cv2.FlannBasedMatcher(index_params, search_params)

            matches = flann.knnMatch(des1, des2, k=2)

            # store all the good matches as per Lowe's ratio test.
            good = []
            for m, n in matches:
                if m.distance < 0.5 * n.distance:
                    good.append(m)
            print len(good)
            if len(good) > MIN_MATCH_COUNT:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                matchesMask = mask.ravel().tolist()

                h, w = img1.shape
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M)

                imgtry = cv2.polylines(imgtry, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

            else:
                print "Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT)
                matchesMask = None


        except:
            continue
            matchesMask=None
        avggood = avggood + len(good)



        #print len(matchesMask)
        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                           singlePointColor = None,
                           matchesMask = matchesMask, # draw only inliers
                           flags = 2)

        img3 = cv2.drawMatches(img1,kp1,imgtry,kp2,good,None,**draw_params)

        plt.imshow(img3, 'gray'),plt.show()
    return (avggood / 5)

x=check(img1,img2)
print x,"avg"
# k=5
# avggood=0
# for i in range(k):
#     w, h = img2.shape
#     #print w,h
#     x = randint(0, w - 100)
#     y = randint(0, h - 100)
#     imgtry = img2[x:x + 100, y:y + 100]
#
#     #plt.imshow(imgtry), plt.show()
# # Initiate SIFT detector
#     try:
#         sift = cv2.xfeatures2d.SIFT_create()
#
# # find the keypoints and descriptors with SIFT
#         kp1, des1 = sift.detectAndCompute(img1,None)
#         kp2, des2 = sift.detectAndCompute(imgtry,None)
#
#         FLANN_INDEX_KDTREE = 0
#         index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
#         search_params = dict(checks = 50)
#     #print des1,des2
#         flann = cv2.FlannBasedMatcher(index_params, search_params)
#
#         matches = flann.knnMatch(des1,des2,k=2)
#
# # store all the good matches as per Lowe's ratio test.
#         good = []
#         for m,n in matches:
#             if m.distance < 0.2*n.distance:
#                 good.append(m)
#         print len(good)
#         if len(good)>MIN_MATCH_COUNT:
#             src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
#             dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
#
#             M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
#             matchesMask = mask.ravel().tolist()
#
#             h,w = img1.shape
#             pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
#             dst = cv2.perspectiveTransform(pts,M)
#
#             imgtry = cv2.polylines(imgtry,[np.int32(dst)],True,255,3, cv2.LINE_AA)
#
#         else:
#             print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
#             matchesMask = None
#
#
#     except:
#         continue
#     avggood = avggood + len(good)
#
# print "avg",avggood/5
# #    print len(matchesMask)
# #     draw_params = dict(matchColor = (0,255,0), # draw matches in green color
# #                        singlePointColor = None,
# #                        matchesMask = matchesMask, # draw only inliers
# #                        flags = 2)
# #
# #     img3 = cv2.drawMatches(img1,kp1,imgtry,kp2,good,None,**draw_params)
#
#     #plt.imshow(img3, 'gray'),plt.show()
