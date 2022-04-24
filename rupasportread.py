import imutils
from imutils.contours import sort_contours
import numpy as np
import pytesseract
import sys
import cv2
import re


rus = ['А','Б','В','Г','Д','Е','Ё','Ж','З','И','Й','К','Л','М','Н','О','П','Р','С','Т','У','Ф','Х','Ц','Ч','Ш','Щ','Ъ','Ы','Ь','Э','Ю','Я']
eng = ['A','B','V','G','D','E','2','J','Z','I','Q','K','L','M','N','O','P','R','S','T','U','F','H','C','3','4','W','X','Y','9','6','7','8']



def resize(img):
#     загрузка изображения
    img = cv2.imread(img)
#     изменение размеров
    final_wide = 1200
    r = float(final_wide) / img.shape[1]
    dim = (final_wide, int(img.shape[0] * r))
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
#     фильтры ( оттенки серого, размытие по Гауссу, пороговая обработка)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    kernel = np.ones((7,7), np.uint8)
#     морфология изображения (открытие и закрытие изображения)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
#     поиск контуров (извлечение внешних контуров, получение только 2х основных точек)
    contours = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    area_thresh = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area > area_thresh:
            area_thresh = area
            big_contour = c
    page = np.zeros_like(img)
#     отрисовка контуров
    cv2.drawContours(page, [big_contour], 0, (255,255,255), -1)
    peri = cv2.arcLength(big_contour, True)
    corners = cv2.approxPolyDP(big_contour, 0.04 * peri, True)
    global polygon
    polygon = img.copy()
    cv2.polylines(polygon, [corners], True, (0,0,255), 1, cv2.LINE_AA)
    yarr = list()
    xarr = list()
    nr = np.empty((0,2), dtype="int32")
    for a in corners:
        for b in a:
            nr = np.vstack([nr, b])
    for i in nr:
        yarr.append(i[0])
        xarr.append(i[1])
    x = min(yarr)
    pX = max(yarr)
    y = min(xarr)
    pY = max(xarr)
    global photo
    photo = img[y:pY, x:pX]
    cv2.waitKey(0)
    return photo

def pasp_read(photo):
    image = photo
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
        if percentWidth > 0.29 and percentHeight > 0.005:
            mrzBox = (x, y, w, h)
            break
    if mrzBox is None:
        print("[INFO] MRZ could not be found")
        sys.exit(0)
    (x, y, w, h) = mrzBox
    pX = int((x + w) * 0.03)
    pY = int((y + h) * 0.083)
    (x, y) = (x - pX, y - pY)
    (w, h) = (w + (pX * 2), h + (pY * 2))
    mrz = image[y:y + h, x:x + w]
    config = (" --oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789><")
    mrzText = pytesseract.image_to_string(mrz, lang='eng', config = config)
    mrzText = mrzText.replace(" ", "")
    mrzText = mrzText.split()
    if mrzText[0][0:1] != 'P':
        del mrzText[0]
    el1 = mrzText[0]
    el2 = mrzText[1]
    el1 = el1.replace('1','I')
    el2 = el2.replace('O','0')
    el1 = el1[5:]
    el1 = re.split("<<|<|\n", el1)
    el2 = re.split("RUS|<", el2)
    el1 = list(filter(None, el1))
    el1 = list(map(list, el1))
    el1 = el1[0:3]
    el2 = list(filter(None, el2))
    for i in el1:
        for c, j in enumerate(i):
            ind = eng.index(str(j))
            i[c] = rus[ind]
    surname = ''.join(el1[0])
    name = ''.join(el1[1])
    otch = ''.join(el1[2])
    seria = el2[0][0:3] + el2[2][0:1]
    nomer = el2[0][3:9]
    data = el2[1][0:6]
    if int(data[0:1]) > 2:
        data = '19' + data
    else:
        data = '20' + data
    data = data[6:8] + '.' + data[4:6] + '.' + data[0:4]
    global pasdata
    pasdata = {'Surname': surname, 'Name': name, 'Mid': otch, 'Date' : data, 'Series': seria, 'Number': nomer}
    return pasdata

def download(image, filename):
    resize(image)
    cv2.imwrite(filename, photo) 

def catching(image):
    try:
        photo = resize(image)
        pasp_read(photo)
        print(pasdata)
    except ValueError:
        photo = cv2.imread(image)
        pasp_read(photo)
        print(pasdata)
