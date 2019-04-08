import cv2
import numpy as np
from vector import distance

def getDigitEdges(img):
    dim=img.shape
    north=0
    south=0
    east=0
    west=0
    for r in range(0,dim[0]):
        for c in range(0, dim[1]):
            if north==0:
                if(img[r,c]==1):
                    north=r
            if south==0:
                if (img[dim[0] - 1 - r, c] == 1):
                    south=dim[0] - 1 - r

    for c in range(0,dim[1]):
        for r in range(0, dim[0]):
            if west==0:
                if(img[r,c]==1):
                    west=c
            if east==0:
                if (img[r, dim[1] - 1 - c] == 1):
                    east=dim[1] - 1 - c

    return north,south,west,east


def transform_img(img):
    n,s,w,e=getDigitEdges(img)
    ret = (np.zeros((28,28))/255.0)>0.5
    if(img.shape[0]==0 or img.shape[1]==0):
        return ret
    ret[0:s-n+2, 0:e-w+2] = img[n-1:s+1, w-1:e+1]
    return ret

def inRange(r, item, items):
    retVal = []
    for obj in items:
        mdist = distance(item['center'], obj['center'])
        if(mdist<r):
            retVal.append(obj)
    return retVal

def draw_history_and_future_arrays(img, element, t):
    for hist in element["history"]:
        ttt = t - hist['t']
        if (ttt < 100):
            cv2.circle(img, hist['center'], 1, (0, 255, 255), 1)

    for fu in element["future"]:
        ttt = fu[0] - t
        if (ttt < 100):
            cv2.circle(img, (fu[1], fu[2]), 1, (255, 255, 0), 1)

def draw_id_and_recognized_value(img, element):
    cv2.putText(img, str(element['id']),
        (element['center'][0] - 20, element['center'][1] - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0))

    cv2.putText(img, str(int(element['value'])),
        (element['center'][0] + 15, element['center'][1] + 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255))

def draw_line_endpoints(img, linesEndpoints):
    linesEndpointColors = [(0, 0, 255), (0, 255, 255)]
    for idx, linija in enumerate(linesEndpoints):
        cv2.circle(img, linija[1], 2, linesEndpointColors[idx], 2)
        cv2.circle(img, linija[0], 2, linesEndpointColors[idx], 2)

def write_file_header():
    with open("level3/out.txt", "w") as outfile:
        outfile.write("RA 54/2015 Jovan Grgur\n")
        outfile.write("file\tsum\n")
        outfile.close()

def append_result_to_file(videoNumber, result):
    with open("level3/out.txt", "a") as outfile:
        outfile.write("video-" + str(videoNumber) + ".avi\t" + str(result) + "\n")
