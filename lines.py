import cv2
import numpy as np

def getHoughLines(gray):
    edges = cv2.Canny(gray, 70, 130, apertureSize=5)
    #cv2.imshow("canny", edges)

   
    minLineLength = 30 # minimalno rastojanje u pikselima za koje algoritam smatra da je to linija
    maxLineGap = 10 # maximalno rastojanje izmedju linija za koje algoritam smatra da je jedna linija

    
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80, minLineLength, maxLineGap)
    #print("cv2.HoughLinesP = ", lines)

    minval = (-1, -1)
    maxval = (10000, 10000)

    #print("FOR-------------")

    for p in lines:
        #print(p)
        if (p[0][1] > minval[1]):
            minval = (p[0][0], p[0][1])
        if (p[0][3] < maxval[1]):
            maxval = (p[0][2], p[0][3])

    #print("ENDFOR-------------")

    return [minval, maxval] # [(pocetak_x, pocetak_y), (kraj_x, kraj_y)]

def get_green_line(hsv, img):
    lower_green = np.array([36, 25, 25],np.uint8)
    upper_green = np.array([70, 255,255],np.uint8)
    mask_green = cv2.inRange(hsv, lower_green, upper_green) # inRange vraca binarnu sliku na osnovu tresholda. ako piksel upada u opseg 
    # lower_green - upper_green, u rezultatu ce on biti bele boje (1)

    # cv2.imshow("green", mask_green)
    green_gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("green_gray", green_gry)
    # cv2.imshow("bitwise", cv2.bitwise_and(green_gry, mask_green))

    # Posmatramo grayscale sliku. Svaki piksel ce zadrzati svoju vrednost ukoliko je u masci taj piksel imao vrednost 1, u protivnom dobija vrednost 0
    line_green = getHoughLines(cv2.bitwise_and(green_gry, mask_green))
    
    return line_green

def get_blue_line(hsv, img):
    lower_blue = np.array([110,50,50],np.uint8)
    upper_blue = np.array([130,255,255],np.uint8)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    blue_gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("bitwise", cv2.bitwise_and(blue_gry, mask_blue))
    line_blue = getHoughLines(cv2.bitwise_and(blue_gry, mask_blue))
    return line_blue
