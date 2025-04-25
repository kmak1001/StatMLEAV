import os
import scipy.io
from scipy.signal import butter
from scipy import signal

from EAV_datasplit import *

import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

class DataLoadEEG:
    def __init__(self, subject='all', band=[0.3, 50], fs_orig=500, fs_target=100,
            parent_directory=r'/home/hice1/jliu3196/scratch/EAV_Dataset/EAV'):
        self.subject = subject
        self.band = band
        self.parent_directory = parent_directory
        self.fs_orig = fs_orig
        self.fs_target = fs_target
        self.seg = []
        self.label = []
        self.label_div = []
        self.seg_f = []
        self.seg_f_div = []

    def data_mat(self):
        subject = f'subject{self.subject:02d}'
        eeg_folder = os.path.join(self.parent_directory, subject, 'EEG')
        eeg_file_name = subject.rstrip('__') + '_eeg.mat'
        eeg_file_path = os.path.join(eeg_folder, eeg_file_name)

        label_file_name = subject.rstrip('__') + '_eeg_label.mat'
        label_file_path = os.path.join(eeg_folder, label_file_name)

        if os.path.exists(eeg_file_path):
            mat = scipy.io.loadmat(eeg_file_path)
            cnt_ = np.array(mat.get('seg1'))
            if np.ndim(cnt_) == 3:
                cnt_ = np.array(mat.get('seg1'))
            else:
                cnt_ = np.array(mat.get('seg'))

            mat_y = scipy.io.loadmat(label_file_path)
            label = np.array(mat_y.get('label'))

            self.seg = np.transpose(cnt_, [1, 0, 2])  # (10000, 30, 200) -> (30ch, 10000t, 200trial)
            self.label = label

            print(f'Loaded EEG data for {subject}')
        else:
            print(f'EEG data not found for {subject}')

    def downsampling(self, fs_target=100):
        [ch, t, tri] = self.seg.shape
        factor = fs_target / self.fs_orig
        tm = np.reshape(self.seg, [ch, t * tri], order='F')
        tm2 = signal.resample_poly(tm, up=1, down=int(self.fs_orig / fs_target), axis=1)
        self.seg = np.reshape(tm2, [ch, int(t * factor), tri], order='F')

    def bandpass(self):
        [ch, t, tri] = self.seg.shape
        dat = np.reshape(self.seg, [ch, t * tri], order='F')
        # bandpass after the downsample  -> fs_target
        sos = butter(5, self.band, btype='bandpass', fs=self.fs_target, output='sos')
        fdat = list()
        for i in range(np.size(dat, 0)):
            tm = signal.sosfilt(sos, dat[i, :])
            fdat.append(tm)
        self.seg_f = np.array(fdat).reshape((ch, t, tri), order='F')

    def data_div(self):
        # Here 2000 (20seconds) are divided into 4 splits
        [ch, t, tri] = self.seg_f.shape
        tm1 = self.seg_f.reshape((30, 500, 4, 200), order='F')
        self.seg_f_div = tm1.reshape((30, 500, 4 * 200), order='F')
        self.label_div = np.repeat(self.label, repeats=4, axis=1)

        # Here we only select the listening classes
        selected_classes = [1, 3, 5, 7, 9]
        label = self.label_div[selected_classes, :]
        selected_indices = np.isin(np.argmax(self.label_div, axis=0), selected_classes)
        label = label[:, selected_indices]
        x = self.seg_f_div[:, :, selected_indices]


        self.seg_f_div = np.transpose(x, (2, 0, 1))  # (30, 500, 400) -> (400, 30, 500)
        class_indices = np.argmax(label, axis=0)

        #self.label_div = label
        self.label_div = class_indices

    def data_split(self):
        selected_classes = [1, 3, 5, 7, 9]  # only listening classes
        label = self.label_div[selected_classes, :]

        selected_indices = np.isin(np.argmax(self.label_div, axis=0), selected_classes)
        label = label[:, selected_indices]

        x = self.seg_f_div[:, :, selected_indices]
        x_train_list = []
        x_test_list = []
        y_train_list = []
        y_test_list = []

        for i in range(5):  # Looping over each class
            class_indices = np.where(label.T[:, i] == 1)[0]  # Find indices where current class label is 1
            midpoint = len(class_indices) // 2  # Calculate the midpoint for 50% split

            # Split data based on found indices
            x_train_list.append(x[:, :, class_indices[:midpoint]])
            x_test_list.append(x[:, :, class_indices[midpoint:]])

            y_train_list.append(label.T[class_indices[:midpoint]])
            y_test_list.append(label.T[class_indices[midpoint:]])

        # Convert lists to numpy arrays
        x_train = np.concatenate(x_train_list, axis=0)
        x_test = np.concatenate(x_test_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)
        y_test = np.concatenate(y_test_list, axis=0)

    def data_prepare(self):
        self.data_mat()
        self.downsampling()
        self.bandpass()
        self.data_div()
        return self.seg_f_div, self.label_div



#create eeg pickle files
if __name__ == "__main__":
    x_eeg = np.full((16800, 15000), np.nan)
    y_eeg = np.full((16800), np.nan)
    
    for sub in range(1, 43):
        print('SUBJECT', str(sub))
        file_path = "/home/hice1/jliu3196/scratch/EAV_Dataset/Input_images/EEG/"
        file_name = f"subject_{sub:02d}_eeg.pkl"
        file_ = os.path.join(file_path, file_name)
        
        # You can directly work from here
        with open(file_, 'rb') as f:
            eeg_list = pickle.load(f)
        tr_x_eeg, tr_y_eeg, te_x_eeg, te_y_eeg = eeg_list
        
        # need to reconcatenate the data into one set
        x_eeg[80*(sub-1):80*sub] = np.concatenate((np.reshape(tr_x_eeg[0:56], (56, -1)), np.reshape(te_x_eeg[0:24], (24, -1))))
        y_eeg[80*(sub-1):80*sub] = np.concatenate((tr_y_eeg[0:56], te_y_eeg[0:24]))
        
        x_eeg[80*(sub-1)+3360:80*sub+3360] = np.concatenate((np.reshape(tr_x_eeg[56:112], (56, -1)), np.reshape(te_x_eeg[24:48], (24, -1))))
        y_eeg[80*(sub-1)+3360:80*sub+3360] = np.concatenate((tr_y_eeg[56:112], te_y_eeg[24:48]))
        
        x_eeg[80*(sub-1)+6720:80*sub+6720] = np.concatenate((np.reshape(tr_x_eeg[112:168], (56, -1)), np.reshape(te_x_eeg[48:72], (24, -1))))
        y_eeg[80*(sub-1)+6720:80*sub+6720] = np.concatenate((tr_y_eeg[112:168], te_y_eeg[48:72]))
        
        x_eeg[80*(sub-1)+10080:80*sub+10080] = np.concatenate((np.reshape(tr_x_eeg[168:224], (56, -1)), np.reshape(te_x_eeg[72:96], (24, -1))))
        y_eeg[80*(sub-1)+10080:80*sub+10080] = np.concatenate((tr_y_eeg[168:224], te_y_eeg[72:96]))
        
        x_eeg[80*(sub-1)+13440:80*sub+13440] = np.concatenate((np.reshape(tr_x_eeg[224:280], (56, -1)), np.reshape(te_x_eeg[96:120], (24, -1))))
        y_eeg[80*(sub-1)+13440:80*sub+13440] = np.concatenate((tr_y_eeg[224:280], te_y_eeg[96:120]))
        
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
    
    train_x_eeg = x_eeg[all_train_id]
    train_y_eeg = y_eeg[all_train_id]
    val_x_eeg = x_eeg[all_val_id]
    val_y_eeg = y_eeg[all_val_id]
    test_x_eeg = x_eeg[all_test_id]
    test_y_eeg = y_eeg[all_test_id]
    
    # hyperparameters to select from
    n_estimators_params = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250]
    max_depth_params = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    
    X_train = np.concatenate((train_x_eeg, val_x_eeg))
    y_train = np.concatenate((train_y_eeg, val_y_eeg))
    
    # initialize results vector
    cv_results = np.array([0.303486394557823,0.277380952380952,0.281632653061224,0.279591836734693,0.273299319727891,0.273044217687074,0.271343537414966,0.272108843537414,0.27797619047619,0.272278911564625,0.307653061224489,0.299404761904761,0.293962585034013,0.291071428571428,0.284863945578231,0.288010204081632,0.291921768707483,0.291581632653061,0.290986394557823,0.296173469387755,0.319472789115646,0.313860544217687,0.299234693877551,0.296853741496598,0.300510204081632,0.300680272108843,0.303316326530612,0.299319727891156,0.300085034013605,0.303231292517006,0.311054421768707,0.310884353741496,0.300680272108843,0.306037414965986,0.30875850340136,0.311649659863945,0.304931972789115,0.305612244897959,0.30484693877551,0.307312925170068,0.321513605442176,0.325,0.311394557823129,0.311819727891156,0.313095238095238,0.309438775510204,0.308928571428571,0.30875850340136,0.314200680272108,0.315051020408163,0.32610544217687,0.323214285714285,0.318027210884353,0.315306122448979,0.315051020408163,0.317772108843537,0.315646258503401,0.311649659863945,0.31156462585034,0.310544217687074,0.326785714285714,0.319642857142857,0.31360544217687,0.31828231292517,0.312159863945578,0.320068027210884,0.317942176870748,0.316071428571428,0.318792517006802,0.315986394557823,0.329676870748299,0.324489795918367,0.318877551020408,0.320918367346938,0.318792517006802,0.319132653061224,0.322874149659863,0.317517006802721,0.326700680272108,0.318622448979591,0.336989795918367,0.326870748299319,0.325085034013605,0.32219387755102,0.319217687074829,0.31938775510204,0.322108843537415,0.317431972789115,0.324574829931972,0.320408163265306,0.328656462585034,0.319217687074829,0.327380952380952,0.326615646258503,0.318622448979591,0.327721088435374,0.324234693877551,0.322363945578231,0.327465986394557,0.326190476190476])
    
    '''
    for i in n_estimators_params:
        for j in max_depth_params:
            if (i > 250) or ((i == 250) and (j > 40)):
                clf = RandomForestClassifier(n_estimators = i, max_depth = j)
    
                with open("EEG ALL SUBJECTS RESULTS.txt", "a") as f:
                    f.write('n_estimators: ' + str(i) + ', max_depth: ' + str(j) + '\n')
                cv_results[int(i/5*2 + j/10 - 11)] = np.mean(cross_val_score(clf, X_train, y_train, cv=5))
                with open("EEG ALL SUBJECTS RESULTS.txt", "a") as f:
                    f.write(str(cv_results[int(i/5*2 + j/10 - 11)]) + '\n')
    '''            
            
    # determine maximum score
    max_score_id = np.min(np.where(cv_results == np.max(cv_results)))
    best_n_estimators = int(np.ceil((max_score_id+1)/10)*25)
    best_max_depth = int(((max_score_id % 10) + 1) * 10)
    
    with open("EEG ALL SUBJECTS RESULTS.txt", "a") as f:
        f.write('BEST n_estimators: ' + str(best_n_estimators) + ', BEST max_depth: ' + str(best_max_depth) + '\n')
    
    # fit to training and validation sets
    clf = RandomForestClassifier(n_estimators = best_n_estimators, max_depth = best_max_depth)
    clf.fit(X_train, y_train)
    
    pred_y_eeg = clf.predict(test_x_eeg)
    conf_mat = confusion_matrix(test_y_eeg, pred_y_eeg)
    
    with open("EEG ALL SUBJECTS RESULTS.txt", "a") as f:
        f.write(str(conf_mat) + '\n' + str(clf.score(test_x_eeg, test_y_eeg)) + '\n\n')
