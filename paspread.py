from imutils.contours import sort_contours
import numpy as np
import pytesseract
import argparse
import imutils
import sys
import cv2
import re


rus = ['А','Б','В','Г','Д','Е','Ё','Ж','З','И','Й','К','Л','М','Н','О','П','Р','С','Т','У','Ф','Х','Ц','Ч','Ш','Щ','Ъ','Ы','Ь','Э','Ю','Я']
eng = ['A','B','V','G','D','E','2','J','Z','I','Q','K','L','M','N','O','P','R','S','T','U','F','H','C','3','4','W','X','Y','9','6','7','8']



def pasp_read(path):
    image = cv2.imread(path)
    final_wide = 1000
    r = float(final_wide) / image.shape[1]
    dim = (final_wide, int(image.shape[0] * r))
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (H, W) = gray.shape
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
    grad = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    grad = np.absolute(grad)
    (minVal, maxVal) = (np.min(grad), np.max(grad))
    grad = (grad - minVal) / (maxVal - minVal)
    grad = (grad * 255).astype("uint8")
    grad = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, rectKernel)
    thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
    thresh = cv2.erode(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sort_contours(cnts, method="bottom-to-top")[0]
    mrzBox = None
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        percentWidth = w / float(W)
        percentHeight = h / float(H)
        if percentWidth > 0.3 and percentHeight > 0.005:
            mrzBox = (x, y, w, h)
            break
    if mrzBox is None:
        print("[INFO] MRZ could not be found")
        sys.exit(0)
    (x, y, w, h) = mrzBox
    pX = int((x + w) * 0.05)
    pY = int((y + h) * 0.05)
    (x, y) = (x - pX, y - pY)
    (w, h) = (w + (pX * 2), h + (pY * 2))
    mrz = image[y:y + h, x:x + w]
    custom_config = r'--oem 3 --psm 6'
    mrzText = pytesseract.image_to_string(mrz, lang='eng', config = custom_config)
    mrzText = mrzText.replace(" ", "")
    mrzText = mrzText.split()
    el1 = mrzText[0]
    el2 = mrzText[1]
    el1 = el1[5:]
    el1 = re.split("<<|<|\n", el1)
    el1 = list(filter(None, el1))
    el1 = list(map(list, el1))
    el1 = el1[0:3]
    el2 = re.split("<", el2)
    el2 = list(filter(None, el2))
    for i in el1:
        for c, j in enumerate(i):
            ind = eng.index(str(j))
            i[c] = rus[ind]
    surname = ''.join(el1[0])
    name = ''.join(el1[1])
    otch = ''.join(el1[2])
    seria = el2[0][0:3] + el2[1][0:1]
    nomer = el2[0][3:9]
    pasdata = {'Surname': surname, 'Name': name, 'Mid': otch, 'Series': seria, 'Number': nomer}
    return pasdata