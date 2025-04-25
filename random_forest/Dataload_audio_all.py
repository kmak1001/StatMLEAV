import os
import torchaudio
from torchaudio.transforms import Resample
from transformers import ASTFeatureExtractor

from EAV_datasplit import *
from Transformer_torch import Transformer_Audio

import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

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
    x_aud = np.full((16800, 80000), np.nan)
    y_aud = np.full((16800), np.nan)
    
    for sub in range(1, 43):
        print('SUBJECT', str(sub))
        file_path = "/home/hice1/jliu3196/scratch/EAV_Dataset/Input_images/Audio/"
        file_name = f"subject_{sub:02d}_aud.pkl"
        file_ = os.path.join(file_path, file_name)
        
        # You can directly work from here
        with open(file_, 'rb') as f:
            Aud_list = pickle.load(f)
        [tr_x_aud, tr_y_aud, te_x_aud, te_y_aud] = Aud_list
        
        # need to reconcatenate the data into one set
        x_aud[80*(sub-1):80*sub] = np.concatenate((tr_x_aud[0:56], te_x_aud[0:24]))
        y_aud[80*(sub-1):80*sub] = np.concatenate((tr_y_aud[0:56], te_y_aud[0:24]))
        
        x_aud[80*(sub-1)+3360:80*sub+3360] = np.concatenate((tr_x_aud[56:112], te_x_aud[24:48]))
        y_aud[80*(sub-1)+3360:80*sub+3360] = np.concatenate((tr_y_aud[56:112], te_y_aud[24:48]))
        
        x_aud[80*(sub-1)+6720:80*sub+6720] = np.concatenate((tr_x_aud[112:168], te_x_aud[48:72]))
        y_aud[80*(sub-1)+6720:80*sub+6720] = np.concatenate((tr_y_aud[112:168], te_y_aud[48:72]))
        
        x_aud[80*(sub-1)+10080:80*sub+10080] = np.concatenate((tr_x_aud[168:224], te_x_aud[72:96]))
        y_aud[80*(sub-1)+10080:80*sub+10080] = np.concatenate((tr_y_aud[168:224], te_y_aud[72:96]))
        
        x_aud[80*(sub-1)+13440:80*sub+13440] = np.concatenate((tr_x_aud[224:280], te_x_aud[96:120]))
        y_aud[80*(sub-1)+13440:80*sub+13440] = np.concatenate((tr_y_aud[224:280], te_y_aud[96:120]))
        
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
    
    train_x_aud = x_aud[all_train_id]
    train_y_aud = y_aud[all_train_id]
    val_x_aud = x_aud[all_val_id]
    val_y_aud = y_aud[all_val_id]
    test_x_aud = x_aud[all_test_id]
    test_y_aud = y_aud[all_test_id]
        
    # hyperparameters to select from
    n_estimators_params = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250]
    max_depth_params = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    
    X_train = np.concatenate((train_x_aud, val_x_aud))
    y_train = np.concatenate((train_y_aud, val_y_aud))
    
    # initialize results vector
    cv_results = np.zeros(len(n_estimators_params)*len(max_depth_params))
    
    cv_results[0:97] = np.array([0.334948979591836,0.316241496598639,0.331632653061224,0.32984693877551,0.326615646258503,0.335459183673469,0.326020408163265,0.335034013605442,0.327891156462585,0.33078231292517,0.356972789115646,0.348639455782312,0.353656462585033,0.345833333333333,0.344472789115646,0.349659863945578,0.350595238095238,0.345068027210884,0.352891156462585,0.344727891156462,0.355102040816326,0.353656462585034,0.360289115646258,0.36156462585034,0.36156462585034,0.362414965986394,0.351955782312925,0.354506802721088,0.355612244897959,0.359438775510204,0.360969387755102,0.361989795918367,0.365986394557823,0.372704081632653,0.364370748299319,0.357142857142857,0.35875850340136,0.35969387755102,0.359438775510204,0.364455782312925,0.364115646258503,0.362840136054421,0.361394557823129,0.363520408163265,0.368792517006802,0.366071428571428,0.364795918367346,0.369302721088435,0.367431972789115,0.36938775510204,0.364625850340136,0.364710884353741,0.367261904761904,0.372448979591836,0.361819727891156,0.366326530612244,0.367602040816326,0.372108843537414,0.374659863945578,0.368197278911564,0.366496598639455,0.370918367346938,0.372704081632653,0.372363945578231,0.38375850340136,0.370748299319727,0.37704081632653,0.370578231292517,0.379081632653061,0.37704081632653,0.364625850340136,0.370408163265306,0.375595238095238,0.375595238095238,0.376955782312925,0.372023809523809,0.382482993197278,0.375,0.378486394557823,0.372874149659863,0.370833333333333,0.374829931972789,0.375510204081632,0.380357142857142,0.380187074829931,0.374319727891156,0.38188775510204,0.37312925170068,0.37517006802721,0.376445578231292,0.374914965986394,0.371428571428571,0.384183673469387,0.374744897959183,0.37984693877551,0.384438775510204,0.374319727891156])
    
    for i in n_estimators_params:
        for j in max_depth_params:
            if (i > 250) or ((i == 250) and (j > 70)):
                clf = RandomForestClassifier(n_estimators = i, max_depth = j)
    
                with open("AUDIO ALL SUBJECTS RESULTS.txt", "a") as f:
                    f.write('n_estimators: ' + str(i) + ', max_depth: ' + str(j) + '\n')
                cv_results[int(i/5*2 + j/10 - 11)] = np.mean(cross_val_score(clf, X_train, y_train, cv=5))
                with open("AUDIO ALL SUBJECTS RESULTS.txt", "a") as f:
                    f.write(str(cv_results[int(i/5*2 + j/10 - 11)]) + '\n')
            
            
    # determine maximum score
    max_score_id = np.min(np.where(cv_results == np.max(cv_results)))
    best_n_estimators = int(np.ceil((max_score_id+1)/10)*25)
    best_max_depth = int(((max_score_id % 10) + 1) * 10)
    
    with open("AUDIO ALL SUBJECTS RESULTS.txt", "a") as f:
        f.write('BEST n_estimators: ' + str(best_n_estimators) + ', BEST max_depth: ' + str(best_max_depth) + '\n')
    
    # fit to training and validation sets
    clf = RandomForestClassifier(n_estimators = best_n_estimators, max_depth = best_max_depth)
    clf.fit(X_train, y_train)
    
    pred_y_aud = clf.predict(test_x_aud)
    conf_mat = confusion_matrix(test_y_aud, pred_y_aud)
    
    with open("AUDIO ALL SUBJECTS RESULTS.txt", "a") as f:
        f.write(str(conf_mat) + '\n' + str(clf.score(test_x_aud, test_y_aud)) + '\n\n')
