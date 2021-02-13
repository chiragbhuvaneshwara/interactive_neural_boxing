import bpy # base blender python api
from mathutils import * # imports Vector, Matrix, Quaternion etc. 
import math # standard python math functionality
import os
import pandas as pd
import matplotlib.pyplot as plt

"""left arm instead of shoulder"""

def SetFrame(f):
    bpy.context.scene.frame_set(f)
    bpy.context.view_layer.update()

class BoxingDetector():
    
    def __init__(self, o, fs, fe, type='detailed'):
        self.o = o
        self.fs = fs
        self.fe = fe
                
        self.type = type
        
        # for awinda suit 
        self.left = [self.o.pose.bones["LeftShoulder"], self.o.pose.bones["LeftWrist"]]
        self.right = [self.o.pose.bones["RightShoulder"], self.o.pose.bones["RightWrist"]]
        self.neck = self.o.pose.bones["Neck"]
#        self.max_threshold = 0.13
        self.max_threshold = 0.4
        
        # for perception neuron suit
#        self.left = [self.o.pose.bones["LeftShoulder"], self.o.pose.bones["LeftHand"]]
#        self.right = [self.o.pose.bones["RightShoulder"], self.o.pose.bones["RightHand"]]
#        self.max_threshold = 0.52
        
        self.max_punch_distance = 0
        self.hand_distances = []
            
    def detectPunches(self, arm):
        '''
        Detect punches for an arm
        Input:
            arm - self.left or self.right
        '''
        
        punch_labels = [0.0] * (self.fe - self.fs +1)
        
        next_start = self.fs
        
        prev_end = 0
        
        #next punch frame and next punch distance
        npf, npd, heightStatus = self.detectNextPunch(arm, self.fs)
        while (npf < self.fe):
            
            # Obtain frame and hand distance for hand in starting position of punch
            # start, next punch distance start
            start, npds = self.detectPunchStart(arm, npf, npd, heightStatus, direction=-1)
            
            # Obtain frame and hand distance for hand in ending position of punch
            # end, next punch distance end
            end, npde = self.detectPunchStart(arm, npf, npd, heightStatus, direction=+1)
            
            
            if npd > self.max_punch_distance:
                self.max_punch_distance = npd
            
            
            #if npds > self.max_punch_distance or npde > self.max_punch_distance:
            #    if npds > npde:
            #        greater = npds
            #    else:
            #        greater = npde
                
            #    self.max_punch_distance = greater 
            
            if prev_end > start:
                start = prev_end
            
            prev_end = end
                    
#            print("punch detected")
            print("\t", start, npds)
            print("\t", npf, npd)
            print("\t", end, npde)
            print('\n')
            
            # Marking arm stretching out with increasing values of phase i.e from 0 to 1
            for i in range(max(start, self.fs), npf):
                
                if self.type == 'detailed':
                    punch_labels[i] = (i - start) / (npf - start)
                elif self.type == 'tertiary':
                    punch_labels[i] = 1#
                elif self.type == 'binary':
                    punch_labels[i] = 1#
#                print(npf,punch_labels[i])
            punch_labels[npf] = 1
#            print(punch_labels[i+1])
            
            # Marking arm stretching back to OG pos with decreasing values of phase i.e from .9 to 0
            for i in range(npf +1, min(end, self.fe)):
                punch_labels[i] = 1 - (i - npf) / (end - npf)
                punch_labels[i] = -1
                
                if self.type == 'detailed':
                    punch_labels[i] = 1 - (i - npf) / (end - npf)
                elif self.type == 'tertiary':
                    punch_labels[i] = -1
                elif self.type == 'binary':
                    punch_labels[i] = 1
#                print(npf,punch_labels[i])
            npf, npd, heightStatus = self.detectNextPunch(arm, end)
          
        return punch_labels

        
    
    def detectPunchStart(self, arm, punch, fextd, heightStatus, direction = -1):
        '''
        Find min point of punch (least extended arm position) 
        applicable for punch starting as well as punch ending
        
        Input:
            arm - self.left or self.right
            punch - max extended frame number
            fextd - full extension distance
            heightStatus - if Flase, fextd has distance. if true, fextd has height
            direction - -1 for backwards, +1 for forwards search
        Output:
            (frame_number, distance)        
        '''
        SetFrame(punch)    
        lastDistance = (arm[1].head - arm[0].head).length
        # iterate max 100 frames in direction
        for f in range(punch + direction, punch + direction * 25, direction):
            SetFrame(f)
            
            if not heightStatus:
                # calculate hand distance
                handDistance = (arm[1].head - arm[0].head).length
                diff = fextd - handDistance

                if arm == self.left:
    #                diff_thresh = 0.05
                    diff_thresh = 0.19
                    
                elif arm == self.right:
    #                diff_thresh = 0.06
                    diff_thresh = 0.19
                
                if diff >= diff_thresh:
                    return (f, handDistance) 
                
                
                # dir=1 : last - hand >= 0.01 => stop and return
                # dir=-1 : last - hand <= -0.01 => stop and return
                
                #if dir == 1: 
                    #if (lastDistance - handDistance) >= 0.01:
                        # if hand is not moving backwards anymore, we have found the frame
                        #return (f, handDistance)
                #elif dir == -1:
                    # Original Janis cond: if (handDistance - lastDistance) >= 0.01:
                    #if (lastDistance - handDistance) <= -0.01:
                        # if hand is not moving backwards anymore, we have found the frame
                        #return (f, handDistance)
                
                lastDistance = handDistance
                
            elif heightStatus:
#                print('++++++++++++++++++++++++++')
                curr_punch_height = arm[1].tail.y
                curr_shoulder_height = arm[0].head.y          
                if fextd > curr_punch_height and curr_punch_height > curr_shoulder_height:
                    return(f, curr_punch_height)  
            
        
        print('worst case')
        punch_start = punch + direction*25
        
        
        return(punch_start, lastDistance)
            
    
    def detectNextPunch(self, arm, start):
        '''
        Detect the next punch frame and distance of hand. 
        
        i.e obtain frame for full extension and distance at full extension
        i.e gives mid point frame of punch duration and hand distance at that frame
        
        Input:
            arm - self.left or self.right
            start - current start frame from where we should look            
        '''
        heightInfo = False # if False send Distance. if true send height
        lastDistance = 0.0
        lastHeight = 0.0
        for f in range(start, self.fe):
            SetFrame(f)
            handDistance = (arm[1].head - arm[0].head).length
            handHeight = arm[1].tail.y
            heightThreshold = (self.neck.head.y + self.neck.tail.y)/2
            
            # hand is in punchable area condition
            if handDistance > self.max_threshold:
                
                # if hand starts moving back
                if (handDistance - lastDistance) < -0.008:
                    heightInfo = False 
                    #print(lastDistance)
                    
                    # return frame and distance where hand was fully extended
                    return (f-1, lastDistance, heightInfo)
                
            elif handHeight > heightThreshold:
                if (handHeight - lastHeight) < -0.008:
                    heightInfo = True
                    return (f-1, lastHeight, heightInfo)
            
            lastDistance = handDistance
            lastHeight = handHeight
        
        heightInfo = False
        # returning next punch frame and next punch distance
        return (self.fe, lastDistance, heightInfo)

def load_bvh(path):
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = 2
    bpy.ops.import_anim.bvh(filepath=path, 
        axis_forward='Z', axis_up='Y', update_scene_fps=True,
        update_scene_duration=True, rotate_mode="YXZ")
    
    return (bpy.context.scene.objects[-1], 
            bpy.context.scene.frame_end)





print("Start.")
#o, b = load_bvh("/home/chirag/Documents/Uni/Thesis/Data/MocapBoxing/axis_neuron_processed/4_Stepping_AxisNeuronProcessed_Char00.bvh")
#o, b = load_bvh(r'C:\Users\chira\OneDrive\Documents\Uni\Thesis\VCS-boxing-predictor\Data\boxing_chirag\processed\boxing_11.bvh')

base_path = 'C:/Users/chira/OneDrive/Documents/Uni/Thesis/VCS-boxing-predictor/Data/boxing_chirag/processed'
#base_path = 'C:/Users/chira/OneDrive/Documents/Uni/Thesis/VCS-boxing-predictor/Data/boxing_chirag/'
df_path = 'C:/Users/chira/OneDrive/Documents/Uni/Thesis/VCS-boxing-predictor/Blender Code Snippets/data annotation res/new_data'
type = 'tertiary'
#type = 'binary'
#type = 'detailed'

#base_path = r'C:\Users\chira\OneDrive\Documents\Uni\Thesis\VCS-boxing-predictor\Data\boxing_chirag\processed'
dir_files = os.listdir(base_path)
for file in dir_files[3:4]:
#    file = 'boxing_2_temp.bvh'
    print(os.path.join(base_path,file))
    o, b = load_bvh(os.path.join(base_path,file))
    bpy.context.view_layer.objects.active = o
#    bpy.data.objects[file].select_set(True)
    print(bpy.data.objects)
    bd = BoxingDetector(bpy.context.object, 1, b, type)
    print("right punches")
    pr = bd.detectPunches(bd.right)

    print('####################################################################')
    print("\nleft punches")
    pl = bd.detectPunches(bd.left)

    df = pd.DataFrame({'right punch': pr, 'left punch': pl})
    ##print('\/')
    print(df)
    df.to_csv(os.path.join(df_path, file.split('.')[0]+'_'+type+'.csv'))
    #df.to_csv(r'C:\Users\chira\OneDrive\Documents\Uni\Thesis\VCS-boxing-predictor\Blender Code Snippets\data annotation res\PunchWithMaxDist.csv')

    print('done with ', file)
    print('\n')    
    
    
#    bpy.ops.object.select_all(action='DESELECT')
#    bpy.data.objects[file.split('.')[0]].select_set(True)    
#    bpy.ops.object.delete() 

