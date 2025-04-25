import os
import cv2
import numpy as np
import EAV_datasplit
from facenet_pytorch import MTCNN
import torch

import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

class DataLoadVision:
    def __init__(self, subject='all', parent_directory=r'/home/hice1/jliu3196/scratch/EAV_Dataset/EAV', face_detection=False,
                 image_size=224):
        self.IMG_HEIGHT, self.IMG_WIDTH = 480, 640
        self.subject = subject
        self.parent_directory = parent_directory
        self.file_path = list()
        self.file_emotion = list()
        self.images = list()
        self.image_label = list()  # actual class name
        self.image_label_idx = list()
        self.face_detection = face_detection
        self.image_size = image_size
        self.face_image_size = 56 #
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.mtcnn = MTCNN(
            image_size=self.face_image_size, margin=0, min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
            device=self.device
        )

    def data_files(self):
        subject = f'subject{self.subject:02d}'
        print(subject, " Loading")
        file_emotion = []
        subjects = []
        path = os.path.join(self.parent_directory, subject, 'Video')
        for i in os.listdir(path):
            emotion = i.split('_')[4]
            self.file_emotion.append(emotion)
            self.file_path.append(os.path.join(path, i))

    def data_load(self):

        for idx, file in enumerate(self.file_path):
            nm_class = file.split("_")[-1].split(".")[0]  # we extract the class label from the file

            if "Speaking" in file and file.endswith(".mp4"):
                print(idx)
                cap = cv2.VideoCapture(file)
                # total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # ~600
                # frame_rate = cap.get(cv2.CAP_PROP_FPS) # 30 frame
                a1 = []
                if cap.isOpened():
                    frame_index = 1
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        # (30 framerate * 20s) * 100 Speaking, Select every 6th frame from the first 600 frames
                        # face detection, we converted it into 0-255 again from the [-1 - 1] tensor, you can directly return the tensor
                        if (frame_index - 1) % 6 == 0 and frame_index <= 600:
                            if self.face_detection:
                                with torch.no_grad():
                                    x_aligned, prob = self.mtcnn(frame, return_prob=True)
                                    if prob > 0.3:
                                        x_aligned = (x_aligned + 1) / 2
                                        x_aligned = np.clip(x_aligned * 255, 0, 255)
                                        x_aligned = np.transpose(x_aligned.numpy().astype('uint8'), (1, 2, 0))
                                        a1.append(x_aligned)
                                    else:
                                        print("Face is not detected, original is saved")
                                        a1.append(x_aligned)  # incase that face has not been detected, add previous one
                                    pass
                            else:
                                resizedImg = cv2.resize(frame, (self.image_size, self.image_size))
                                a1.append(resizedImg) # sabina: dlkfjefoie

                            if len(a1) == 25:  # 25 frame is 5s each
                                self.images.append(a1)  # this will contain 400 samples [400, 25, (225, 225, 3)]
                                a1 = []
                                self.image_label.append(nm_class)
                        frame_index += 1
                    cap.release()
                else:
                    print(f"Error opening video file: {file}")
        emotion_to_index = {
            'Neutral': 0,
            'Happiness': 3,
            'Sadness': 1,
            'Anger': 2,
            'Calmness': 4
        }
        self.image_label_idx = [emotion_to_index[emotion] for emotion in self.image_label]

    def process(self):
        self.data_files()
        self.data_load()
        return self.images, self.image_label_idx


if __name__ == '__main__':
    x_vis = np.full((16800, 235200), np.nan)
    y_vis = np.full((16800), np.nan)

    for sub in range(1, 43):
        print('SUBJECT', str(sub))
        file_path = "/home/hice1/jliu3196/scratch/EAV_Dataset/Input_images/Vision"
        file_name = f"subject_{sub:02d}_vis.pkl"
        file_ = os.path.join(file_path, file_name)
        
        # You can directly work from here        
        with open(file_, 'rb') as f:
            Vis_list = pickle.load(f)
        tr_x_vis, tr_y_vis, te_x_vis, te_y_vis = Vis_list
        
        # need to reconcatenate the data into one set
        x_vis[80*(sub-1):80*sub] = np.concatenate((np.reshape(tr_x_vis[0:56], (56, -1)), np.reshape(te_x_vis[0:24], (24, -1))))
        y_vis[80*(sub-1):80*sub] = np.concatenate((tr_y_vis[0:56], te_y_vis[0:24]))
        
        x_vis[80*(sub-1)+3360:80*sub+3360] = np.concatenate((np.reshape(tr_x_vis[56:112], (56, -1)), np.reshape(te_x_vis[24:48], (24, -1))))
        y_vis[80*(sub-1)+3360:80*sub+3360] = np.concatenate((tr_y_vis[56:112], te_y_vis[24:48]))
        
        x_vis[80*(sub-1)+6720:80*sub+6720] = np.concatenate((np.reshape(tr_x_vis[112:168], (56, -1)), np.reshape(te_x_vis[48:72], (24, -1))))
        y_vis[80*(sub-1)+6720:80*sub+6720] = np.concatenate((tr_y_vis[112:168], te_y_vis[48:72]))
        
        x_vis[80*(sub-1)+10080:80*sub+10080] = np.concatenate((np.reshape(tr_x_vis[168:224], (56, -1)), np.reshape(te_x_vis[72:96], (24, -1))))
        y_vis[80*(sub-1)+10080:80*sub+10080] = np.concatenate((tr_y_vis[168:224], te_y_vis[72:96]))
        
        x_vis[80*(sub-1)+13440:80*sub+13440] = np.concatenate((np.reshape(tr_x_vis[224:280], (56, -1)), np.reshape(te_x_vis[96:120], (24, -1))))
        y_vis[80*(sub-1)+13440:80*sub+13440] = np.concatenate((tr_y_vis[224:280], te_y_vis[96:120]))
        
    # randomly split each of 5 classes into training, validation, and test sets
    np.random.seed(2025)
    data_id_num = np.arange(3360)
    np.random.shuffle(data_id_num)
    
    # train set is 70% of 70%
    train_id = data_id_num[0:1646]
    
    # validation set is 30% of 70%
    val_id = data_id_num[1646:2352]
    
    # test set is 30%
    test_id = data_id_num[2352:3360]
        
    # concatenate across all classes
    all_train_id = np.concatenate((train_id, train_id + 3360, train_id + 6720, train_id + 10080, train_id + 13440))
    all_val_id = np.concatenate((val_id, val_id + 3360, val_id + 6720, val_id + 10080, val_id + 13440))
    all_test_id = np.concatenate((test_id, test_id + 3360, test_id + 6720, test_id + 10080, test_id + 13440))
    
    train_x_vis = x_vis[all_train_id]
    train_y_vis = y_vis[all_train_id]
    val_x_vis = x_vis[all_val_id]
    val_y_vis = y_vis[all_val_id]
    test_x_vis = x_vis[all_test_id]
    test_y_vis = y_vis[all_test_id]
    
    
    # hyperparameters to select from
    n_estimators_params = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250]
    max_depth_params = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    
    X_train = np.concatenate((train_x_vis, val_x_vis))
    y_train = np.concatenate((train_y_vis, val_y_vis))
    
    # initialize results vector
    cv_results = np.array([0.546938775510204,0.587159863945578,0.58095238095238,0.580952380952381,0.583078231292517,0.584353741496598,0.586904761904761,0.577806122448979,0.589965986394557,0.580187074829932,0.587244897959183,0.644727891156462,0.640221088435374,0.64345238095238,0.644557823129251,0.638010204081632,0.642942176870748,0.641326530612244,0.642176870748299,0.640816326530612,0.614965986394557,0.66547619047619,0.669982993197278,0.667091836734693,0.667517006802721,0.663775510204081,0.67125850340136,0.666156462585034,0.663180272108843,0.662244897959183,0.618112244897959,0.677891156462585,0.675680272108843,0.681037414965986,0.679336734693877,0.687074829931972,0.676955782312925,0.68078231292517,0.681377551020408,0.687840136054421,0.625425170068027,0.685799319727891,0.694472789115646,0.688775510204081,0.694132653061224,0.685374149659863,0.693792517006802,0.694812925170068,0.692687074829932,0.692942176870748,0.63188775510204,0.698979591836734,0.698214285714285,0.694047619047619,0.701870748299319,0.701530612244898,0.698809523809523,0.696598639455782,0.698384353741496,0.702210884353741,0.636054421768707,0.700085034013605,0.699404761904761,0.703231292517006,0.703656462585034,0.70484693877551,0.699404761904761,0.705272108843537,0.706632653061224,0.708163265306122,0.638775510204081,0.704081632653061,0.708673469387755,0.70688775510204,0.707568027210884,0.70204081632653,0.708418367346938,0.708078231292517,0.710289115646258,0.706292517006802,0.638690476190476,0.708588435374149,0.71267006802721,0.712159863945578,0.714200680272108,0.712585034013605,0.71267006802721,0.712925170068027,0.70875850340136,0.710969387755102,0.636224489795918,0.714795918367346,0.711819727891156,0.720578231292517,0.712755102040816,0.716326530612244,0.714285714285714,0.71641156462585,0.711479591836734,0.709438775510204])
    
    '''
    for i in n_estimators_params:
        for j in max_depth_params:
            if (i > 250):
                clf = RandomForestClassifier(n_estimators = i, max_depth = j)
    
                with open("VISION ALL SUBJECTS RESULTS.txt", "a") as f:
                    f.write('n_estimators: ' + str(i) + ', max_depth: ' + str(j) + '\n')
                cv_results[int(i/5*2 + j/10 - 11)] = np.mean(cross_val_score(clf, X_train, y_train, cv=5))
                with open("VISION ALL SUBJECTS RESULTS.txt", "a") as f:
                    f.write(str(cv_results[int(i/5*2 + j/10 - 11)]) + '\n')
    '''     
            
    # determine maximum score
    max_score_id = np.min(np.where(cv_results == np.max(cv_results)))
    best_n_estimators = int(np.ceil((max_score_id+1)/10)*25)
    best_max_depth = int(((max_score_id % 10) + 1) * 10)
    
    with open("VISION ALL SUBJECTS RESULTS.txt", "a") as f:
        f.write('BEST n_estimators: ' + str(best_n_estimators) + ', BEST max_depth: ' + str(best_max_depth) + '\n')
    
    # fit to training and validation sets
    clf = RandomForestClassifier(n_estimators = best_n_estimators, max_depth = best_max_depth)
    clf.fit(X_train, y_train)
    
    pred_y_vis = clf.predict(test_x_vis)
    conf_mat = confusion_matrix(test_y_vis, pred_y_vis)
    
    with open("VISION ALL SUBJECTS RESULTS.txt", "a") as f:
        f.write(str(conf_mat) + '\n' + str(clf.score(test_x_vis, test_y_vis)) + '\n\n')
