#!/usr/bin/python

__author__ = "Hamideh Kerdegari"
__copyright__ = "Copyright 2018"
__credits__ = ["Hamideh Kerdegari"]
__license__ = ""
__version__ = "1.0"
__maintainer__ = "Hamideh Kerdegari"
__email__ = "hamideh.kerdegari@gmail.com"
__status__ = "Development"
# ==============================================
# Description:
# This Python script receives two images and stores the output in the outFolderPath.
#
# usage: python homography.py [-h] image1Path image2Path outFolderPath
#
# positional arguments:
#   image1Path     First image which will be used as reference.
#   image2Path     Second image to be aligned.
#   outFolderPath  Path to a directory to store the outputs.
#
# optional arguments:
#   -h, --help     show this help message and exit
# ==============================================

import cv2, argparse, os, sys
import numpy as np

# ========== START alignImages function ===========
def alignImages(im1, im2):
    max_features = 500
    good_match_percent = 0.15

    # Making gray images
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(max_features)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create("BruteForce-Hamming")
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * good_match_percent)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    homog, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, homog, (width, height))

    return im1Reg, homog, imMatches
# ========== END alignImages function ===========

# ========== START saveResults function ===========
# Saving images and homogMatrix into file
def saveResults(m_outFolderPath, imgReg, img1, img2, homogMat):
    imgAll = np.hstack((img1, imgReg, img2))

    print("Saving aligned and matches images ...\n")
    # Write image to disk.
    cv2.imwrite(m_outFolderPath + "/aligned.jpg", imgAll)

    #----------------------------
    # Writing estimated homography matrix
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.pcolor(homogMat * 0, cmap='binary', edgecolors='k', linewidths=4)

    # Loop over data dimensions and create text annotations.
    for i in range(len(homogMat)):
        for j in range(len(homogMat[0])):
            homogMatrixImage = ax.text(j + 0.5, i + 0.5, homogMat[len(homogMat[0]) - i - 1, j],
                           ha="center", va="center", color="k")

    ax.set_title("Estimated homography matrix")
    fig.savefig(m_outFolderPath + "/homographyMatrix.jpg", dpi=fig.dpi)
# ========== END saveResults function ===========

# ========== START showResults function ===========
# Ploting the images and the homogMatrix
def showResults(imgReg, img1, img2, homogMat):
    imgRegSmall = cv2.resize(imgReg, (0, 0), None, 0.08, 0.08)
    img1Small = cv2.resize(img1, (0, 0), None, 0.08, 0.08)
    img2Small = cv2.resize(img2, (0, 0), None, 0.08, 0.08)

    imgAll = np.hstack((img1Small, imgRegSmall, img2Small))
    cv2.imshow('Press any key to exit)', imgAll)
    cv2.waitKey()


    # -----------------------
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.pcolor(homogMat * 0, cmap='binary', edgecolors='k', linewidths=4)

    # Loop over data dimensions and create text annotations.
    for i in range(len(homogMat)):
        for j in range(len(homogMat[0])):
            homogMatrixImage = ax.text(j + 0.5, i + 0.5, homogMat[len(homogMat[0]) - i - 1, j],
                                       ha="center", va="center", color="k")

    ax.set_title("Estimated homography matrix")
    plt.show()
# ========== END showResults function ===========

#####################################################################################
if __name__ == '__main__':

    # Reading paths to input images and output directory
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('image1Path', help='First image which will be used as reference.')
    parser.add_argument('image2Path', help='Second image to be aligned.')
    parser.add_argument('outFolderPath', help='Path to a directory to store the outputs.')
    args = parser.parse_args()

    image1Path = args.image1Path
    image2Path = args.image2Path
    outFolderPath = args.outFolderPath

    # Check if input path are valid
    if not (os.path.exists(image1Path) and os.path.exists(image2Path)):
        sys.exit("Fatal error! Image1 or Image2 does not exist.")

    # Making output dir if not exists
    if not os.path.exists(outFolderPath):
        os.makedirs(outFolderPath)

    # -----------------------------
    print("Reading reference image : ", image1Path)
    image1 = cv2.imread(image1Path, cv2.IMREAD_COLOR)

    print("Reading image to align : ", image2Path)
    image2 = cv2.imread(image2Path, cv2.IMREAD_COLOR)

    print("Aligning images ...")
    # Registered image will be resotred in imReg.
    # The estimated homography will be stored in h.
    imageRegistered, homogMatrix, imageMatches = alignImages(image2, image1)

    # showing the results
    showResults(imageRegistered, image1, image2, homogMatrix)

    # Saving results in the output folder
    saveResults(outFolderPath, imageRegistered, image1, image2, homogMatrix)
