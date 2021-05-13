# Removing unwanted finger joints
# right_unwanted_ids = [i for i in range(17, 36)]
# left_unwanted_ids = [i for i in range(40, 59)]
# indices_unwanted_joints = right_unwanted_ids + left_unwanted_ids
# diff = 36 - 17  # or 59 - 40
# new_left_hand_id = bc.char.hand_left - diff
# new_right_hand_id = bc.char.hand_right
#
# print('Left_id:', new_left_hand_id, 'Right_id:', new_right_hand_id)
# new_poses = []
# for pose in poses:
#     new_pose = np.delete(pose, indices_unwanted_joints, axis=0)
#     new_poses.append(new_pose)
# poses = np.array(new_poses)