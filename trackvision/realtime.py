from collections import deque
import numpy as np
import argparse
import imutils
import cv2
import math
import scipy.ndimage

# Define the font for text
font = cv2.FONT_HERSHEY_SIMPLEX

def orientated_non_max_suppression(mag, ang):
    ang_quant = np.round(ang / (np.pi/4)) % 4
    winE = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]])
    winSE = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    winS = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
    winSW = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])

    magE = non_max_suppression(mag, winE)
    magSE = non_max_suppression(mag, winSE)
    magS = non_max_suppression(mag, winS)
    magSW = non_max_suppression(mag, winSW)

    mag[ang_quant == 0] = magE[ang_quant == 0]
    mag[ang_quant == 1] = magSE[ang_quant == 1]
    mag[ang_quant == 2] = magS[ang_quant == 2]
    mag[ang_quant == 3] = magSW[ang_quant == 3]
    return mag

def non_max_suppression(data, win):
    data_max = scipy.ndimage.filters.maximum_filter(data, footprint=win, mode='constant')
    data_max[data != data_max] = 0
    return data_max

# Define color range for detection
greenLower = (0, 0, 0)
greenUpper = (87, 255, 222)

# Open a video capture object
cap = cv2.VideoCapture(0)  # 0 is typically the default webcam

# Set the parameters for processing
with_nmsup = True  # apply non-maximal suppression
fudgefactor = 1.5  # adjust threshold
sigma = 50  # Gaussian Kernel standard deviation
kernel = 2 * math.ceil(2 * sigma) + 1  # Kernel size

try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture video.")
            break
        
        frame = cv2.resize(frame, (800, 600))
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        gray_image = gray_image / 255.0
        blur = cv2.GaussianBlur(gray_image, (kernel, kernel), sigma)
        gray_image = cv2.subtract(gray_image, blur)

        # Compute Sobel response
        sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

        mag = np.hypot(sobelx, sobely)
        ang = np.arctan2(sobely, sobelx)

        # Threshold
        threshold = 4 * fudgefactor * np.mean(mag)
        mag[mag < threshold] = 0

        if not with_nmsup:
            mag = cv2.normalize(mag, 0, 255, cv2.NORM_MINMAX)
            kernel = np.ones((5, 5), np.uint8)
            result = cv2.morphologyEx(mag, cv2.MORPH_CLOSE, kernel)
            num_white = np.sum(result == 255)
            num_black = np.sum(result == 0)
            print(num_white)
            print(num_black)
            print((num_white / num_black) * 100)
            cv2.imshow('Original', frame)
            cv2.imshow('Processed', result)
        else:
            mag = orientated_non_max_suppression(mag, ang)
            mag[mag > 0] = 255
            mag = mag.astype(np.uint8)

            kernel = np.ones((5, 5), np.uint8)
            result = cv2.morphologyEx(mag, cv2.MORPH_CLOSE, kernel)
            num_white = np.sum(result == 255)
            num_black = np.sum(result == 0)
            ratio = (num_white / num_black) * 100
            print(num_white)
            print(num_black)
            print(ratio)
            if ratio > 0.7:
                print("Cracked")
                cv2.putText(frame, 'Cracked', (0, 30), font, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                print("Not Cracked")
                cv2.putText(frame, 'Not Cracked', (0, 30), font, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            
            cv2.imshow('Original', frame)
            cv2.imshow('Processed', result)
        
        # Exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # When everything is done, release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()
