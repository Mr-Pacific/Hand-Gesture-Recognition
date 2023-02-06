#-------------------------Importing libraries-------------------------#

import cv2
import numpy as np
import math
# import vlc
from os import walk
from GestureAPI import *

#-----------------------Variables & parameters------------------------#

cam_region_Xbegin = 0.5
cam_region_Yend = 0.8
bg_captured = 0
capture_done = 0
capture_pos_X = 450
capture_pos_Y = 170
capture_box_dim = 20
capture_box_sep_X = 8
capture_box_sep_Y = 18
capture_box_count = 9
morph_elem_size = 13
gaussian_ksize = 11
gaussian_sigma = 0
median_ksize = 3
hsv_thresh_lower = 10
finger_thresh_l = 2.0
finger_thresh_u = 3.8
radius_thresh = 0.04
finger_ct_history = [0,0]
first_iteration=True
# play_list = []
# track_no = "char"

gestureDictionary = defineGesture()
frame_gesture = Gesture("frame_gesture")
# player = vlc.MediaPlayer()

#------------------------Function declaration------------------------#

# 1. Remove background from image

def remove_bg(frame):
    fg_mask = bg_model.apply(frame)
    kernal = np.ones((3,3), np.uint8)
    fg_mask = cv2.erode(fg_mask, kernal, iterations = 1)
    frame = cv2.bitwise_and(frame, frame, mask = fg_mask)
    return frame

# 2. Hand Capture Histrogram

def hand_capture(frame_in, box_X, box_Y):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    ROI = np.zeros([capture_box_dim*capture_box_count, capture_box_dim, 3], dtype = hsv.dtype)
    for i in xrange(capture_box_count):
        ROI[i*capture_box_dim : i*capture_box_dim + capture_box_dim, 0 : capture_box_dim] = hsv[box_Y[i] : box_Y[i] + capture_box_dim, box_X[i] : box_X[i] + capture_box_dim]
    hand_hist = cv2.calcHist([ROI], [0,1], None, [180, 256], [0, 180, 0, 256])
    cv2.normalize(hand_hist, hand_hist, 0, 255, cv2.NORM_MINMAX)
    return hand_hist

# 3. Filters & Threshold

def hand_threshold(frame_in, hand_hist):
    frame_in = cv2.medianBlur(frame_in, 3)
    hsv = cv2.cvtColor(frame_in, cv2.COLOR_BGR2HSV)
    hsv[0:int(cam_region_Yend*hsv.shape[0]),0:int(cam_region_Xbegin*hsv.shape[1])]=0
    hsv[int(cam_region_Yend*hsv.shape[0]):hsv.shape[0],0:hsv.shape[1]]=0
    back_projection = cv2.calcBackProject([hsv], [0,1],hand_hist, [00,180,0,256], 1)
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_elem_size,morph_elem_size))
    cv2.filter2D(back_projection, -1, disc, back_projection)
    back_projection=cv2.GaussianBlur(back_projection,(gaussian_ksize,gaussian_ksize), gaussian_sigma)
    back_projection=cv2.medianBlur(back_projection,median_ksize)
    ret, thresh = cv2.threshold(back_projection, hsv_thresh_lower, 255, 0)
    return thresh

# 4. Find hand contour

def hand_contour_find(contours):
    max_area=0
    largest_contour = -1
    for i in range(len(contours)):
        cont=contours[i]
        area=cv2.contourArea(cont)
        if(area>max_area):
            max_area=area
            largest_contour=i
    if(largest_contour==-1):
        return False,0
    else:
        h_contour=contours[largest_contour]
        return True,h_contour

# 5. Mark hand center circle

def mark_hand_center(frame_in, cont):
    max_d=0
    pt=(0,0)
    x,y,w,h = cv2.boundingRect(cont)
    for ind_y in xrange(int(y+0.3*h),int(y+0.8*h)):
        for ind_x in xrange(int(x+0.3*w),int(x+0.6*w)):
            dist= cv2.pointPolygonTest(cont,(ind_x,ind_y),True)
            if(dist>max_d):
                max_d=dist
                pt=(ind_x,ind_y)
    if(max_d>radius_thresh*frame_in.shape[1]):
        thresh_score=True
        cv2.circle(frame_in,pt,int(max_d),(255,0,0),2)
    else:
        thresh_score=False
    return frame_in,pt,max_d,thresh_score

# 6. Detect & mark Fingers

def mark_fingers(frame_in,hull,pt,radius):
    global first_iteration
    global finger_ct_history
    finger=[(hull[0][0][0],hull[0][0][1])]
    j=0

    cx = pt[0]
    cy = pt[1]
    
    for i in range(len(hull)):
        dist = np.sqrt((hull[-i][0][0] - hull[-i+1][0][0])**2 + (hull[-i][0][1] - hull[-i+1][0][1])**2)
        if (dist>18):
            if(j==0):
                finger=[(hull[-i][0][0],hull[-i][0][1])]
            else:
                finger.append((hull[-i][0][0],hull[-i][0][1]))
            j=j+1
    
    temp_len=len(finger)
    i=0
    while(i<temp_len):
        dist = np.sqrt( (finger[i][0]- cx)**2 + (finger[i][1] - cy)**2)
        if(dist<finger_thresh_l*radius or dist>finger_thresh_u*radius or finger[i][1]>cy+radius):
            finger.remove((finger[i][0],finger[i][1]))
            temp_len=temp_len-1
        else:
            i=i+1        
    
    temp_len=len(finger)
    if(temp_len>5):
        for i in range(1,temp_len+1-5):
            finger.remove((finger[temp_len-i][0],finger[temp_len-i][1]))
    
    palm=[(cx,cy),radius]

    if(first_iteration):
        finger_ct_history[0]=finger_ct_history[1]=len(finger)
        first_iteration=False
    else:
        finger_ct_history[0]=0.34*(finger_ct_history[0]+finger_ct_history[1]+len(finger))

    if((finger_ct_history[0]-int(finger_ct_history[0]))>0.8):
        finger_count=int(finger_ct_history[0])+1
    else:
        finger_count=int(finger_ct_history[0])

    finger_ct_history[1]=len(finger)

    count_text="FINGERS:"+str(finger_count)
    cv2.putText(frame_in,count_text,(int(0.62*frame_in.shape[1]),int(0.88*frame_in.shape[0])),cv2.FONT_HERSHEY_DUPLEX,1,(0,255,255),1,8)

    for k in range(len(finger)):
        cv2.circle(frame_in,finger[k],10,255,2)
        cv2.line(frame_in,finger[k],(cx,cy),255,2)
    return frame_in,finger,palm

# 7. Find and display gesture

def find_gesture(frame_in,finger,palm):
    frame_gesture.set_palm(palm[0],palm[1])
    frame_gesture.set_finger_pos(finger)
    frame_gesture.calc_angles()
    gesture_found = decideGesture(frame_gesture,gestureDictionary)
    gesture_text="GESTURE:"+str(gesture_found)
    cv2.putText(frame_in,gesture_text,(int(0.56*frame_in.shape[1]),int(0.97*frame_in.shape[0])),cv2.FONT_HERSHEY_DUPLEX,1,(0,255,255),1,8)
    return frame_in,gesture_found

# 8. instantiating music player and creating play list

def play_music():
    track_ct = 1
    for (dirpath, dirnames, filenames) in walk('C:\Users\Mr. Pacific\Desktop\Major Project\music library'):
            play_list.extend(filenames)
            for i in play_list:
                print (str(track_ct)+' => '+i+'\n')
                track_ct += 1
    track_ct = 1
    print('Enter song no.')
   
#--------------------------------BEGIN----------------------------#

# Camera initialising
cam = cv2.VideoCapture(0)

while(cam.isOpened()):
    # Capturing from camera
    ret, frame = cam.read()
    
    # Operation in the frame
    frame = cv2.bilateralFilter(frame, 5, 50, 100)
    frame = cv2.flip(frame,1)
    cv2.rectangle(frame, (int(cam_region_Xbegin*frame.shape[1]), 0),(frame.shape[1],int(cam_region_Yend*frame.shape[0])),(212,255,127),1)
    frame_original = np.copy(frame)

    if(bg_captured):
        fg_frame = remove_bg(frame)
        cv2.imshow('foreground model',fg_frame)
        
    if(not(capture_done and bg_captured)):
        if(not bg_captured):
            cv2.putText(frame, "Remove hand from the frame & press 'B' to capture background", (int(0.05*frame.shape[1]), int(0.97*frame.shape[0])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, 8)
        else:
            cv2.putText(frame, "Place hand inside boxes & press 'C' to capture hand histogram", (int(0.08*frame.shape[1]), int(0.97*frame.shape[0])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, 8)

            first_iteration = True
            finger_ct_history = [0,0]
            box_pos_X = np.array([capture_pos_X, capture_pos_X + capture_box_dim + capture_box_sep_X, capture_pos_X + 2*capture_box_dim +
                                  2*capture_box_sep_X, capture_pos_X, capture_pos_X + capture_box_dim + capture_box_sep_X, capture_pos_X +
                                  2*capture_box_dim + 2*capture_box_sep_X, capture_pos_X, capture_pos_X + capture_box_dim + capture_box_sep_X,
                                  capture_pos_X + 2*capture_box_dim + 2*capture_box_sep_X], dtype = int)
            box_pos_Y = np.array([capture_pos_Y, capture_pos_Y, capture_pos_Y, capture_pos_Y + capture_box_dim + capture_box_sep_Y, capture_pos_Y +
                                  capture_box_dim + capture_box_sep_Y, capture_pos_Y + capture_box_dim + capture_box_sep_Y, capture_pos_Y + 2*capture_box_dim +
                                  2*capture_box_sep_Y, capture_pos_Y + 2*capture_box_dim + 2*capture_box_sep_Y, capture_pos_Y + 2*capture_box_dim + 2*capture_box_sep_Y], dtype = int)
            for i in range(capture_box_count):
                cv2.rectangle(frame, (box_pos_X[i], box_pos_Y[i]), (box_pos_X[i] + capture_box_dim, box_pos_Y[i] + capture_box_dim), (212,255,127), 1)
            
    else:
        frame = hand_threshold(fg_frame, hand_histogram)
        cv2.imshow('medianblur',frame)
        contour_frame = np.copy(frame)
        image, contours, hierarchy = cv2.findContours(contour_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        found, hand_contour = hand_contour_find(contours)
        if (found):
            hand_convex_hull = cv2.convexHull(hand_contour)
            frame, hand_center, hand_radius, hand_size_score = mark_hand_center(frame_original,hand_contour)
            if(hand_size_score):
                frame, finger, palm = mark_fingers(frame, hand_convex_hull, hand_center, hand_radius)
                frame, gesture_found = find_gesture(frame, finger, palm)
                finger_ct = int(len(finger))
                if (gesture_found == "V"):
                    if not(player.is_playing()):   
                        player = vlc.MediaPlayer('C:\Users\Mr. Pacific\Desktop\Major Project\music library\\'+track_no)
                        player.play()
                elif finger_ct == 5:
                    player.stop()
                    
        else:
            frame = frame_original
            
    # Display frame in a window
    cv2.imshow('Hand Gesture Recognition',frame)
    interrupt = cv2.waitKey(10)

    # Quit by pressing 'q'
    if interrupt & 0xFF == ord('q'):
        break
    
    # Capturing background by pressing 'b'
    elif interrupt & 0xFF == ord('b'):
        bg_model = cv2.createBackgroundSubtractorMOG2(0,0.1)
        bg_captured = 1
        
    # Capturing hand by pressing 'c'
    elif interrupt & 0xFF == ord('c'):
        if (bg_captured):
            capture_done = 1
            hand_histogram = hand_capture(frame_original, box_pos_X, box_pos_Y)

    # Reset all
    elif interrupt & 0XFF == ord('r'):
        capture_done = 0
        bg_captured = 0
        track_no = "char"
        track_ct = 1
        player.release()

    #Preparing music player
    elif interrupt & 0xFF == ord('p'):
        print ('Initialising Music Player....\n\n')
        if bg_captured == 1 & capture_done == 1:
            play_music()
            track_no = input()
            track_no = play_list[track_no-1]
        
# Releasing Camera & destroting all windows
player.release()
cam.release()
cv2.destroyAllWindows()
