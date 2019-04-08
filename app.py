import cv2
import numpy as np
from scipy import ndimage
from vector import distance, pnt2line
import time
import os

from utils import *
from cnn import *
from knn import *
from lines import *

def nextId():
    global cc
    cc += 1
    return cc

recognition_method = "knn" #cnn

if recognition_method == "knn":
    train_knn()
else:
    train_cnn()

write_file_header()

# color filter
kernel = np.ones((2,2),np.uint8)
lower = np.array([230, 230, 230])
upper = np.array([255, 255, 255])

linesEndpoints = None

for videoNumber in range(0,10):
    videoName = "level3/video-" + str(videoNumber) + ".avi"
    cap = cv2.VideoCapture(videoName)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('rez/video-'+ str(videoNumber) +'-rez.avi',fourcc, 20.0, (640,480))

    linesEndpoints = []
    elements = []
    t =0
    counter = 0
    suma = 0
    times = []
    cc = -1

    while (1):
        start_time = time.time()
        ret, img = cap.read()

        if ret == False:
            break

        if(t==0):
            frm=img.copy()
            hsv = cv2.cvtColor(frm, cv2.COLOR_BGR2HSV)
            linesEndpoints.append(get_green_line(hsv, frm))
            linesEndpoints.append(get_blue_line(hsv, frm))        

        draw_line_endpoints(img, linesEndpoints)

        
        lower = np.array(lower, dtype="uint8") # brojevi su svelije nijanse sive
        upper = np.array(upper, dtype="uint8")
        mask = cv2.inRange(img, lower, upper)
        # cv2.imshow("maska", mask)
        img0 = 1.0 * mask

        img0 = cv2.dilate(img0, kernel)#ako je broj prekriven sumom, dilatacija ce popuniti te praznine na samoj cifri  
        img0 = cv2.dilate(img0, kernel)#jedna dilatacija nije bila dovoljna
        #cv2.imshow("dilat", img0)

        labeled, nr_objects = ndimage.label(img0)
        objects = ndimage.find_objects(labeled)
        for i in range(nr_objects):
            loc = objects[i]
            (xc, yc) = (int((loc[1].stop + loc[1].start) / 2),
                        int((loc[0].stop + loc[0].start) / 2))
            (dxc, dyc) = (int(loc[1].stop - loc[1].start),
                          int(loc[0].stop - loc[0].start))

            if (dxc < 12 and dyc < 12):
                continue

            cv2.circle(img, (xc, yc), 16, (25, 25, 255), 1)
            elem = {'center': (xc, yc), 'size': (dxc, dyc), 't': t}
           
            lst = inRange(20, elem, elements)
            nn = len(lst)
            if nn == 0:
                elem['id'] = nextId()
                elem['t'] = t                
                elem['pass'] = [False, False]
                elem['history'] = [{'center': (xc, yc), 'size': (dxc, dyc), 't': t}]
                elem['future'] = []
                img_slice = img0[yc-14 : yc+14, xc-14: xc+14]
                img_slice = (img_slice / 255.0) > 0.5 #nepotreban korak, isecak je vec binaran
                img_slice = transform_img(img_slice)

                img_slice = np.float32(img_slice.reshape(-1, 784))


                if recognition_method == "knn":
                    ret, result, neighbours, dist = knn.findNearest(img_slice, k=1)
                    elem['value']=ret
                else:
                    elem['value'] = predict(img_slice)

                elements.append(elem)
            elif nn == 1:
                lst[0]['center'] = elem['center']
                lst[0]['t'] = t
                lst[0]['history'].append({'center': (xc, yc), 'size': (dxc, dyc), 't': t})
                lst[0]['future'] = []

        for el in elements:
            tt = t - el['t']

            if tt >= 3:
                continue

            for le in range(len(linesEndpoints)): # [[(100, 200), (300, 400)], [(50, 100), (200, 300)]]
                dist, pnt, r = pnt2line(el['center'], linesEndpoints[le][0], linesEndpoints[le][1])

                if r == -1:
                    continue

                c = (25, 25, 255)

                if dist >= 9:
                    continue

                c = (0, 255, 160)
                cv2.circle(img, el['center'], 16, c, 2)

                if el['pass'][le] == True:
                    continue

                el['pass'][le] = True

                if le == 0:
                    suma -= int(el['value'])
                else:
                    suma += int(el['value'])

                counter += 1 # ne treba


            id = el['id'] #ne treba

            draw_history_and_future_arrays(img, el, t)
            draw_id_and_recognized_value(img, el)

        elapsed_time = time.time() - start_time #ne treba
        times.append(elapsed_time * 1000) # ne treba

        cv2.putText(img, 'Summ:  ' + str(suma), (420, 470), cv2.FONT_HERSHEY_SIMPLEX, 1, (90, 90, 255), 2)

        
        t += 1
        
        cv2.imshow('video-' + str(videoNumber), img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        if k == 32:
            while True:
                if cv2.waitKey(30) == 32:
                    break
        # out.write(img)

    print(str(videoName) + " -- suma=" + str(suma))
    append_result_to_file(videoNumber, suma)
    out.release()
    cap.release()
    cv2.destroyAllWindows()

    et = np.array(times)
    
