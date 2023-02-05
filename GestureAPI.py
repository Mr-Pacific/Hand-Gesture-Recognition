import math
import numpy as np

class Gesture(object):
    def __init__(self, name):
        self.name = name
    def getName(self):
        return self.name
    def set_palm(self, hand_center, hand_radius):
        self.hand_center = hand_center
        self.hand_radius = hand_radius
    def set_finger_pos(self, finger_pos):
        self.finger_pos = finger_pos
        self.finger_count = len(finger_pos)
    def calc_angles(self):
        self.angle = np.zeros(self.finger_count, dtype = int)
        for i in range(self.finger_count):
            y = self.finger_pos[i][1]
            x = self.finger_pos[i][0]
            self.angle[i] = abs(math.atan2((self.hand_center[1]-y),(x-self.hand_center[0]))*180/math.pi)

def defineGesture():
     dict={}
     V = Gesture("V")
     V.set_palm((475,225),10)
     V.set_finger_pos([(490,90), (415,105)])
     V.calc_angles()
     dict[V.getName()] = V

     L_right = Gesture("L_Right")
     L_right.set_palm((475,225),50)
     L_right.set_finger_pos([(450,62), (345,200)])
     L_right.calc_angles()
     dict[L_right.getName()] = L_right

     Index_pointing = Gesture("Index_pointing")
     Index_pointing.set_palm((480,230), 43)
     Index_pointing.set_finger_pos([(475,102)])
     Index_pointing.calc_angles()
     dict[Index_pointing.getName()] = Index_pointing

     return dict

def compareGesture(src1, src2):
    if(src1.finger_count == src2.finger_count):
        if(src1.finger_count == 1):
            angle_diff = src1.angle[0]-src2.angle[0]
            if (angle_diff>20):
                result = 0
            else:
                len1 = np.sqrt((src1.finger_pos[0][0]- src1.hand_center[0])**2 + (src1.finger_pos[0][1]-src1.hand_center[1])**2)
                len2 = np.sqrt((src2.finger_pos[0][0]- src2.hand_center[0])**2 + (src2.finger_pos[0][1]-src2.hand_center[1])**2)
                len_diff = len1/len2
                radii_diff = src1.hand_radius/src2.hand_radius
                len_score = abs(len_diff-radii_diff)
                if(len_score<0.09):
                    result = src2.getName()
                else:
                    result = 0
        else:
            angle_diff=[]
            for i in range(src1.finger_count):
                angle_diff.append(src1.angle[i]-src2.angle[i])
            angle_score = max(angle_diff)-min(angle_diff)
            if(angle_score<15):
                len_diff=[]
                for i in range(src1.finger_count):
                    len1 = np.sqrt((src1.finger_pos[i][0]- src1.hand_center[0])**2 + (src1.finger_pos[i][1] - src1.hand_center[1])**2)
                    len2 = np.sqrt((src2.finger_pos[i][0]- src2.hand_center[0])**2 + (src2.finger_pos[i][1] - src2.hand_center[1])**2)
                    len_diff.append(len1/len2)
                len_score = max(len_diff)- min(len_diff)
                if(len_score<0.06):
                    result = src2.getName()
                else:
                    result = 0
            else:
                result = 0
    else:
        result = 0
    return result

def decideGesture(src , gestureDictionary):
    result_list=[]
    for k in gestureDictionary.keys():
        src2 = '"'+k+'"'
        result = compareGesture(src , gestureDictionary[k])
        if(result != 0):
            return result
    return "None"                
