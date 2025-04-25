import os
import cv2
import numpy as np
import EAV_datasplit
from facenet_pytorch import MTCNN
import torch

import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


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

    for sub in range(29, 43):
        with open("VISION RESULTS.txt", "a") as f:
            f.write('SUBJECT ' + str(sub) + '\n')
        print('SUBJECT', str(sub))
        file_path = "/home/hice1/jliu3196/scratch/EAV_Dataset/Input_images/Vision"
        file_name = f"subject_{sub:02d}_vis.pkl"
        file_ = os.path.join(file_path, file_name)
        
        '''
        #if not os.path.exists(file_):

        vis_loader = DataLoadVision(subject=sub, parent_directory=r'C://Users//minho.lee//Dropbox//Datasets//EAV', face_detection=True)
        [data_vis, data_vis_y] = vis_loader.process()

        eav_loader = EAV_datasplit.EAVDataSplit(data_vis, data_vis_y)

        #each class contains 80 trials, 5/5 radio (h_idx=40), 7/3 ratio (h_dix=56)
        [tr_x_vis, tr_y_vis, te_x_vis, te_y_vis] = eav_loader.get_split(h_idx=56)  # output(list): train, trlabel, test, telabel
        data = [tr_x_vis, tr_y_vis, te_x_vis, te_y_vis]
        
        # Here you can write / load vision features tr:{280}(25, 56, 56, 3), te:{120}(25, 56, 56, 3): trials, frames, height, weight, channel
        import pickle        
        Vis_list = [tr_x_vis, tr_y_vis, te_x_vis, te_y_vis]
        with open(file_, 'wb') as f:
            pickle.dump(Vis_list, f)
        '''
        
        # You can directly work from here        
        with open(file_, 'rb') as f:
            Vis_list = pickle.load(f)
        tr_x_vis, tr_y_vis, te_x_vis, te_y_vis = Vis_list
        
        # need to reconcatenate the data into one set
        x_vis = np.concatenate((tr_x_vis[0:56], te_x_vis[0:24], tr_x_vis[56:112], te_x_vis[24:48], tr_x_vis[112:168], te_x_vis[48:72], tr_x_vis[168:224], te_x_vis[72:96], tr_x_vis[224:280], te_x_vis[96:120]))
        y_vis = np.concatenate((tr_y_vis[0:56], te_y_vis[0:24], tr_y_vis[56:112], te_y_vis[24:48], tr_y_vis[112:168], te_y_vis[48:72], tr_y_vis[168:224], te_y_vis[72:96], tr_y_vis[224:280], te_y_vis[96:120]))
        
        # randomly split each of 5 classes into training, validation, and test sets
        np.random.seed(2025)
        data_id_num = np.arange(80)
        np.random.shuffle(data_id_num)
        
        # train set is 70% of 70%
        train_id = data_id_num[0:39]
        
        # validation set is 30% of 70%
        val_id = data_id_num[39:56]
        
        # test set is 30%
        test_id = data_id_num[56:80]
        
        # concatenate across all classes
        all_train_id = np.concatenate((train_id, train_id + 80, train_id + 160, train_id + 240, train_id + 320))
        all_val_id = np.concatenate((val_id, val_id + 80, val_id + 160, val_id + 240, val_id + 320))
        all_test_id = np.concatenate((test_id, test_id + 80, test_id + 160, test_id + 240, test_id + 320))
        
        train_x_vis = np.reshape(x_vis[all_train_id], (195,-1))
        train_y_vis = y_vis[all_train_id]
        val_x_vis = np.reshape(x_vis[all_val_id], (85,-1))
        val_y_vis = y_vis[all_val_id]
        test_x_vis = np.reshape(x_vis[all_test_id], (120,-1))
        test_y_vis = y_vis[all_test_id]
        
        '''
        # Transformer for Vision
        from Transformer_torch import Transformer_Vision

        mod_path = os.path.join('C:/Users/minho.lee/Dropbox/Projects/EAV', 'facial_emotions_image_detection')
        trainer = Transformer_Vision.ImageClassifierTrainer(data,
                                                            model_path=mod_path, sub=f"subject_{sub:02d}",
                                                            num_labels=5, lr=5e-5, batch_size=128)
        trainer.train(epochs=10, lr=5e-4, freeze=True)
        trainer.train(epochs=5, lr=5e-6, freeze=False)
        trainer.outputs_test

        # CNN for Vision
        from CNN_torch.CNN_Vision import ImageClassifierTrainer
        trainer = ImageClassifierTrainer(data, num_labels=5, lr=5e-5, batch_size=32)
        trainer.train(epochs=3, lr=5e-4, freeze=True)
        trainer.train(epochs=3, lr=5e-6, freeze=False)
        trainer._delete_dataloader()
        trainer.outputs_test
        '''
        
        # hyperparameters to select from
        n_estimators_params = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250]
        max_depth_params = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        
        X_train = np.concatenate((train_x_vis, val_x_vis))
        y_train = np.concatenate((train_y_vis, val_y_vis))
        
        # initialize results vector
        cv_results = np.zeros(len(n_estimators_params)*len(max_depth_params))
        
        for i in n_estimators_params:
            for j in max_depth_params:
                clf = RandomForestClassifier(n_estimators = i, max_depth = j)

                with open("VISION RESULTS.txt", "a") as f:
                    f.write('n_estimators: ' + str(i) + ', max_depth: ' + str(j) + '\n')
                cv_results[int(i/5*2 + j/10 - 11)] = np.mean(cross_val_score(clf, X_train, y_train, cv=5))
                with open("VISION RESULTS.txt", "a") as f:
                    f.write(str(cv_results[int(i/5*2 + j/10 - 11)]) + '\n')
                
                
        # determine maximum score
        max_score_id = np.min(np.where(cv_results == np.max(cv_results)))
        best_n_estimators = int(np.ceil((max_score_id+1)/10)*25)
        best_max_depth = int(((max_score_id % 10) + 1) * 10)
        
        with open("VISION RESULTS.txt", "a") as f:
            f.write('BEST n_estimators: ' + str(best_n_estimators) + ', BEST max_depth: ' + str(best_max_depth) + '\n')
        
        # fit to training and validation sets
        clf = RandomForestClassifier(n_estimators = best_n_estimators, max_depth = best_max_depth)
        clf.fit(X_train, y_train)
        
        pred_y_vis = clf.predict(test_x_vis)
        with open("VISION RESULTS.txt", "a") as f:
            f.write(str(pred_y_vis) + '\n' + str(test_y_vis) + '\n' + str(clf.score(test_x_vis, test_y_vis)) + '\n\n')
            
