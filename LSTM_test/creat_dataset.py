import cv2
import numpy as np
import time
import os
from draw_function import set_cam, custom_landmarks
from operation import get_points_angle
from sklearn.preprocessing import Normalizer


actions = ['crunch', 'lying_leg_raise',
           'side_lunge', 'standing_knee_up',
           'standing_side_crunch']

person_c = [7, 8, 10]
person_llr = [1, 2, 8]
person_sl = [1, 24]
person_ssc = [9, 14]
person_sku = [1, 2]
person_all = [person_c, person_llr, person_sl, person_sku, person_ssc]

pose_angles = ['C']
created_time = int(time.time())
sequence_length = 15
N_scaler = Normalizer()

pose = set_cam()

path_dir = 'D:/fitness_image_data/Training'
os.makedirs('created_dataset', exist_ok=True)

for i, _ in enumerate(actions):
    os.makedirs('C:/Users/UCL7/VS_kwix/created_dataset/dataset_version9'+ '/'+ str(_), exist_ok=True)
    for idx, action in enumerate(actions):
        start = time.time()
        data = []
        path1 = path_dir + '/' + actions[idx]     # D:/fitness_image_data/Training/lying_leg_raise
        print('action:', action)
        person = person_all[idx]

        for p in person:
            path2 = path1 + '/' + str(p)          # D:/fitness_image_data/Training/lying_leg_raise/1
            print('person:', p)

            for p_angle in pose_angles:
                path3 = path2 + '/' + p_angle  # D:/fitness_image_data/Training/lying_leg_raise/1/A
                img_file = os.listdir(path3)

                for frame in img_file:

                    img = path3 + '/' + frame
                    img = cv2.imread(img, cv2.IMREAD_COLOR)
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    results = pose.process(img_rgb)
                    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

                    if results.pose_landmarks is not None:
                        landmark_subset = custom_landmarks(results)
                        joint, angle = get_points_angle(landmark_subset)

                        reshape_angle = np.degrees(angle).reshape(-1, 1)
                        scaled_angle = N_scaler.fit_transform(reshape_angle)

                        angle_label = np.array([scaled_angle], dtype=np.float32)
                        if idx == i:
                            label = 0
                        else: 
                            label = 1
                        angle_label = np.append(angle_label, label)

                        d = np.concatenate([joint.flatten(), angle_label])
                        data.append(d)


        data = np.array(data)
        print(action, data.shape)
        np.save(os.path.join('C:/Users/UCL7/VS_kwix/created_dataset/dataset_version9' + '/' + str(_),
                f'raw_{action}_{created_time}'), data)

        full_seq_data = []
        for seq in range(len(data) - sequence_length):
            full_seq_data.append(data[seq:seq + sequence_length])

        full_seq_data = np.array(full_seq_data)
        print(action, full_seq_data.shape)
        np.save(os.path.join('C:/Users/UCL7/VS_kwix/created_dataset/dataset_version9' + '/' + str(_),
                f'seq_{action}_{created_time}'), full_seq_data)
        print("Working time : ", time.time() - start)
