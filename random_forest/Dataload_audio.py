import os
import torchaudio
from torchaudio.transforms import Resample
from transformers import ASTFeatureExtractor

from EAV_datasplit import *
from Transformer_torch import Transformer_Audio

import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

class DataLoadAudio:
    def __init__(self, subject='all', parent_directory=r'/home/hice1/jliu3196/scratch/EAV_Dataset/EAV', target_sampling_rate=16000):
        self.parent_directory = parent_directory
        self.original_sampling_rate = int()
        self.target_sampling_rate = target_sampling_rate
        self.subject = subject
        self.file_path = list()
        self.file_emotion = list()

        self.seg_length = 5  # 5s
        self.feature = None
        self.label = None
        self.label_indexes = None
        self.test_prediction = list()

    def data_files(self):
        subject = f'subject{self.subject:02d}'
        file_emotion = []
        subjects = []
        path = os.path.join(self.parent_directory, subject, 'Audio')
        for i in os.listdir(path):
            emotion = i.split('_')[4]
            self.file_emotion.append(emotion)
            self.file_path.append(os.path.join(path, i))

    def feature_extraction(self):
        x = []
        y = []
        feature_extractor = ASTFeatureExtractor()
        for idx, path in enumerate(self.file_path):
            waveform, sampling_rate = torchaudio.load(path)
            self.original_sampling_rate = sampling_rate
            if self.original_sampling_rate is not self.target_sampling_rate:
                resampler = Resample(orig_freq=sampling_rate, new_freq=self.target_sampling_rate)
                resampled_waveform = resampler(waveform)
                resampled_waveform = resampled_waveform.squeeze().numpy()
            else:
                resampled_waveform = waveform

            segment_length = self.target_sampling_rate * self.seg_length
            num_sections = int(np.floor(len(resampled_waveform) / segment_length))

            for i in range(num_sections):
                t = resampled_waveform[i * segment_length: (i + 1) * segment_length]
                x.append(t)
                y.append(self.file_emotion[idx])
        print(f"Original sf: {self.original_sampling_rate}, resampled into {self.target_sampling_rate}")

        emotion_to_index = {
            'Neutral': 0,
            'Happiness': 3,
            'Sadness': 1,
            'Anger': 2,
            'Calmness': 4
        }
        y_idx = [emotion_to_index[emotion] for emotion in y]
        self.feature = np.squeeze(np.array(x))
        self.label_indexes = np.array(y_idx)
        self.label = np.array(y)

    def process(self):
        self.data_files()
        self.feature_extraction()
        return self.feature, self.label_indexes

    def label_emotion(self):
        self.data_files()
        self.feature_extraction()
        return self.label

if __name__ == "__main__":
    test_acc = []
    for sub in range(28, 43):
        with open("AUDIO RESULTS.txt", "a") as f:
            f.write('SUBJECT ' + str(sub) + '\n')
        print('SUBJECT', str(sub))
        file_path = "/home/hice1/jliu3196/scratch/EAV_Dataset/Input_images/Audio/"
        file_name = f"subject_{sub:02d}_aud.pkl"
        file_ = os.path.join(file_path, file_name)

        # aud_loader = DataLoadAudio(subject=sub, parent_directory=r'/home/hice1/jliu3196/scratch/EAV_Dataset/EAV')
        # [data_aud , data_aud_y] = aud_loader.process()
        ## audio_loader.label_emotion()

        # division_aud = EAVDataSplit(data_aud, data_aud_y)
        # [tr_x_aud, tr_y_aud, te_x_aud , te_y_aud] = division_aud.get_split(h_idx=56)
        # data = [tr_x_aud, tr_y_aud, te_x_aud , te_y_aud]

        ''' 
        # Here you can write / load vision features tr:{280}(80000), te:{120}(80000): trials, frames, height, weight, channel
        # This code is to store the RAW audio input to the folder: (400, 80000), 16000Hz
        import pickle        
        Aud_list = [tr_x_aud, tr_y_aud, te_x_aud, te_y_aud]
        with open(file_, 'wb') as f:
            pickle.dump(Aud_list, f)
        '''
        
        # You can directly work from here
        with open(file_, 'rb') as f:
            Aud_list = pickle.load(f)
        [tr_x_aud, tr_y_aud, te_x_aud, te_y_aud] = Aud_list
        
        # need to reconcatenate the data into one set
        x_aud = np.concatenate((tr_x_aud[0:56], te_x_aud[0:24], tr_x_aud[56:112], te_x_aud[24:48], tr_x_aud[112:168], te_x_aud[48:72], tr_x_aud[168:224], te_x_aud[72:96], tr_x_aud[224:280], te_x_aud[96:120]))
        y_aud = np.concatenate((tr_y_aud[0:56], te_y_aud[0:24], tr_y_aud[56:112], te_y_aud[24:48], tr_y_aud[112:168], te_y_aud[48:72], tr_y_aud[168:224], te_y_aud[72:96], tr_y_aud[224:280], te_y_aud[96:120]))
        
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
        
        train_x_aud = x_aud[all_train_id]
        train_y_aud = y_aud[all_train_id]
        val_x_aud = x_aud[all_val_id]
        val_y_aud = y_aud[all_val_id]
        test_x_aud = x_aud[all_test_id]
        test_y_aud = y_aud[all_test_id]
        
        '''
        mod_path = os.path.join(os.getcwd(), 'Pre_trained_models/ast-finetuned-audioset')
        Trainer = Transformer_Audio.AudioModelTrainer(data, model_path=mod_path, sub =f"subject_{sub:02d}",
                                                      num_classes=5, weight_decay=1e-5, lr=0.005, batch_size = 8)

        Trainer.train(epochs=10, lr=5e-4, freeze=True)
        Trainer.train(epochs=15, lr=5e-6, freeze=False)
        test_acc.append(Trainer.outputs_test)

        ## Add CNN - audio here, refer to the file Dataload_vision.py
        '''
        
        # hyperparameters to select from
        n_estimators_params = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250]
        max_depth_params = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        
        X_train = np.concatenate((train_x_aud, val_x_aud))
        y_train = np.concatenate((train_y_aud, val_y_aud))
        
        # initialize results vector
        cv_results = np.zeros(len(n_estimators_params)*len(max_depth_params))
        
        for i in n_estimators_params:
            for j in max_depth_params:
                clf = RandomForestClassifier(n_estimators = i, max_depth = j)

                with open("AUDIO RESULTS.txt", "a") as f:
                    f.write('n_estimators: ' + str(i) + ', max_depth: ' + str(j) + '\n')
                cv_results[int(i/5*2 + j/10 - 11)] = np.mean(cross_val_score(clf, X_train, y_train, cv=5))
                with open("AUDIO RESULTS.txt", "a") as f:
                    f.write(str(cv_results[int(i/5*2 + j/10 - 11)]) + '\n')
                
                
        # determine maximum score
        max_score_id = np.min(np.where(cv_results == np.max(cv_results)))
        best_n_estimators = int(np.ceil((max_score_id+1)/10)*25)
        best_max_depth = int(((max_score_id % 10) + 1) * 10)
        
        with open("AUDIO RESULTS.txt", "a") as f:
            f.write('BEST n_estimators: ' + str(best_n_estimators) + ', BEST max_depth: ' + str(best_max_depth) + '\n')
        
        # fit to training and validation sets
        clf = RandomForestClassifier(n_estimators = best_n_estimators, max_depth = best_max_depth)
        clf.fit(X_train, y_train)
        
        pred_y_aud = clf.predict(test_x_aud)
        with open("AUDIO RESULTS.txt", "a") as f:
            f.write(str(pred_y_aud) + '\n' + str(test_y_aud) + '\n' + str(clf.score(test_x_aud, test_y_aud)) + '\n\n')



''' This code is to store the RAW audio input to the folder: (400, 80000), 16000Hz
if __name__ == "__main__":
    test_acc = []
    for sub in range(1,43):
        print(sub)
        file_path = "C:/Users/minho.lee/Dropbox/Datasets/EAV/Input_images/Audio/"
        file_name = f"subject_{sub:02d}_aud.pkl"
        file_ = os.path.join(file_path, file_name)

        aud_loader = DataLoadAudio(subject=sub, parent_directory='C:/Users/minho.lee/Dropbox/Datasets/EAV')
        [data_aud , data_aud_y] = aud_loader.process()
        # audio_loader.label_emotion()

        division_aud = EAVDataSplit(data_aud, data_aud_y)
        [tr_x_aud, tr_y_aud, te_x_aud , te_y_aud] = division_aud.get_split(h_idx=56)

        Aud_list = [tr_x_aud, tr_y_aud, te_x_aud, te_y_aud]
        import pickle
        with open(file_, 'wb') as f:
            pickle.dump(Aud_list, f)
'''



''' test it with the current data
    import pickle
    with open("test_acc_audio.pkl", 'wb') as f:
        pickle.dump(test_acc, f)




with open("test_acc_audio.pkl", 'rb') as f:
    testacc = pickle.load(f)

    # test accuracy for 200 trials
    ## acquire the test label from one subject, it is same for all subjects
    from sklearn.metrics import f1_score
    file_name = f"subject_{1:02d}_vis.pkl"
    file_ = os.path.join(os.getcwd(), 'Feature_vision', file_name)
    with open(file_, 'rb') as f:
        vis_list2 = pickle.load(f)
    te_y_vis = vis_list2[3]

    # load test accuracy for all subjects: 5000 (200, 25) predictions
    with open("test_acc_vision.pkl", 'rb') as f:
        testacc = pickle.load(f)

    test_acc_all = list()
    test_f1_all = list()
    for sub in range(42):
        aa = testacc[sub]
        out1 = np.argmax(aa, axis = 1)
        accuracy = np.mean(out1 == te_y_vis)
        test_acc_all.append(accuracy)

        f1 = f1_score(te_y_vis, out1, average='weighted')
        test_f1_all.append(f1)

    test_acc_all = np.reshape(np.array(test_acc_all), (42, 1))
    test_f1_all = np.reshape(np.array(test_f1_all), (42, 1))





model = AutoModelForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

test_data = torch.tensor(data, dtype=torch.float32)
test_data = test_data.to(device)
aa = test_data[0:20]
with torch.no_grad(): # 572 classes. 
    logits = model(aa).logits

probs = torch.nn.functional.softmax(logits, dim=-1)
predicted_class_id = probs.argmax(dim=1)
bb = np.array(probs.cpu())
config = model.config
config.num_labels
'''
