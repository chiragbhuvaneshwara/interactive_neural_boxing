import bpy, math
from mathutils import *


def SetFrame(f):
    bpy.context.scene.frame_set(f)
    bpy.context.view_layer.update()

class BoxingDetector():
    
    def __init__(self, o, fs, fe):
        self.o = o
        self.fs = fs
        self.fe = fe
        
        self.left = [self.o.pose.bones["LeftShoulder"], self.o.pose.bones["LeftHand"]]
        self.right = [self.o.pose.bones["RightShoulder"], self.o.pose.bones["RightHand"]]
        
        self.max_threshold = 45
            
            
    def detectPunches(self, arm):
        '''
        Detect punches for an arm
        Input:
            arm - self.left or self.right
        '''
        punch_labels = [0.0] * (self.fe - self.fs +1)
        
        next_start = self.fs
        npf, npd = self.detectNextPunch(arm, self.fs)
        while (npf < self.fe):
            start, npds = self.detectPunchStart(arm, npf, direction=-1)
            end, npde = self.detectPunchStart(arm, npf, direction=+1)
            
            print("punch detected")
            print("\t", start, npds)
            print("\t", npf, npd)
            print("\t", end, npde)
            for i in range(max(start, self.fs), npf):
                punch_labels[i] = (i - start) / (npf - start)
            punch_labels[npf] = 1
            for i in range(npf +1, min(end, self.fe)):
                punch_labels[i] = 1 - (i - npf) / (end - npf)
            npf, npd = self.detectNextPunch(arm, end)
          
        return punch_labels

        
    
    def detectPunchStart(self, arm, punch, direction = -1):
        '''
        Find min point of punch (least extended arm position)
        Input:
            arm - self.left or self.right
            punch - max extended frame number
            direction - -1 for backwards, +1 for forwards search
        Output:
            (frame_number, distance)        
        '''
        SetFrame(punch)    
        lastDistance = (arm[1].head - arm[0].head).length
        # iterate max 100 frames in direction
        for f in range(punch + direction, punch + direction * 100, direction):
            SetFrame(f)
            # calculate hand distance
            handDistance = (arm[1].head - arm[0].head).length
            
            if (handDistance - lastDistance) >= 0.01:
                # if hand is not moving backwards anymore, we have found the frame
                return (f, handDistance)
            lastDistance = handDistance
        return(punch + direction*100, lastDistance)
            
    
    def detectNextPunch(self, arm, start):
        '''
        Detect the next punch frame and distance of hand. 
        Input:
            arm - self.left or self.right
            start - current start frame from where we should look            
        '''
        lastDistance = 0.0
        for f in range(start, self.fe):
            SetFrame(f)
            handDistance = (arm[1].head - arm[0].head).length
            if handDistance > self.max_threshold:
                # hand is in punchable area. 
                if (handDistance - lastDistance) < -0.01:
                    # if hand starts moving back 
                    return (f-1, lastDistance)
            lastDistance = handDistance
        return (self.fe, lastDistance)

bd = BoxingDetector(bpy.context.object, 0, 5600)
print("right punches")
pl = bd.detectPunches(bd.right)
print(pl)

print("\nleft punches")
bd.detectPunches(bd.left)