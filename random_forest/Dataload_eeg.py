import os
import scipy.io
from scipy.signal import butter
from scipy import signal

from EAV_datasplit import *

import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

'''
NEU_SPE = 108, 0
S_SPE = 1
A_SPE = 2
H_SPE = 3
R_SPE = 4  #####
'''

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
    for sub in range(1, 43):
        with open("EEG RESULTS.txt", "a") as f:
            f.write('SUBJECT ' + str(sub) + '\n')
        print('SUBJECT', str(sub))
        file_path = "/home/hice1/jliu3196/scratch/EAV_Dataset/Input_images/EEG/"
        file_name = f"subject_{sub:02d}_eeg.pkl"
        file_ = os.path.join(file_path, file_name)
        '''
        eeg_loader = DataLoadEEG(subject=sub, band=[0.5, 45], fs_orig=500, fs_target=100,
                                 parent_directory='C://Users//minho.lee//Dropbox//Datasets//EAV')
        data_eeg, data_eeg_y = eeg_loader.data_prepare()

        division_eeg = EAVDataSplit(data_eeg, data_eeg_y)
        [tr_x_eeg, tr_y_eeg, te_x_eeg, te_y_eeg] = division_eeg.get_split(h_idx=56)
        EEG_list = [tr_x_eeg, tr_y_eeg, te_x_eeg, te_y_eeg]
        
        
        # Here you can write / load vision features tr:{280}(30, 500), te:{120}(30, 500)
        import pickle
        with open(file_, 'wb') as f:
            pickle.dump(EEG_list, f)
        '''
        
        # You can directly work from here
        with open(file_, 'rb') as f:
            eeg_list = pickle.load(f)
        tr_x_eeg, tr_y_eeg, te_x_eeg, te_y_eeg = eeg_list
        # need to reconcatenate the data into one set
        x_eeg = np.concatenate((tr_x_eeg[0:56], te_x_eeg[0:24], tr_x_eeg[56:112], te_x_eeg[24:48], tr_x_eeg[112:168], te_x_eeg[48:72], tr_x_eeg[168:224], te_x_eeg[72:96], tr_x_eeg[224:280], te_x_eeg[96:120]))
        y_eeg = np.concatenate((tr_y_eeg[0:56], te_y_eeg[0:24], tr_y_eeg[56:112], te_y_eeg[24:48], tr_y_eeg[112:168], te_y_eeg[48:72], tr_y_eeg[168:224], te_y_eeg[72:96], tr_y_eeg[224:280], te_y_eeg[96:120]))
        
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
        
        train_x_eeg = np.reshape(x_eeg[all_train_id], (195,-1))
        train_y_eeg = y_eeg[all_train_id]
        val_x_eeg = np.reshape(x_eeg[all_val_id], (85,-1))
        val_y_eeg = y_eeg[all_val_id]
        test_x_eeg = np.reshape(x_eeg[all_test_id], (120,-1))
        test_y_eeg = y_eeg[all_test_id]

        '''
        # Transformer for EEG
        from Transformer_torch import Transformer_EEG
        model = Transformer_EEG.EEGClassificationModel(eeg_channel=30)
        trainer = Transformer_EEG.EEGModelTrainer(data, model = model, lr=0.001, batch_size = 64)
        trainer.train(epochs=100, lr=None, freeze=False)

        [accuracy, predictions] = trainer.evaluate()

        # CNN_tensorflow for EEG
        from CNN_tensorflow.CNN_EEG_tf import EEGNet
        from sklearn.metrics import accuracy_score, confusion_matrix

        model = EEGNet(nb_classes=5, D=8, F2=64, Chans=30, kernLength=300, Samples=500,
                       dropoutRate=0.5)
        model.compile(loss='categorical_crossentropy', optimizer='adam',
                      metrics=['accuracy'])
        y_train = np.zeros((tr_y_eeg.shape[0], 5))
        y_train[np.arange(tr_y_eeg.shape[0]), tr_y_eeg.flatten()] = 1
        y_test = np.zeros((te_y_eeg.shape[0], 5))
        y_test[np.arange(te_y_eeg.shape[0]), te_y_eeg.flatten()] = 1
        x_train = np.reshape(tr_x_eeg, (280, 30, 500, 1))
        x_test = np.reshape(te_x_eeg, (120, 30, 500, 1))
        model.fit(x_train, y_train, batch_size=32, epochs=200, shuffle=True, validation_data=(x_test, y_test))

        pred = model.predict(x_test)
        pred = np.argmax(pred, axis=1)

        y_test2 = np.argmax(y_test, axis=1)
        cm = confusion_matrix(pred, y_test2)
        accuracy = accuracy_score(pred, y_test2)

        # CNN_pytorch for EEG, fix the error, and make the accuracy same
        from CNN_torch.EEGNet_tor import EEGNet_tor, Trainer_uni
        import torch.nn as nn

        model = EEGNet_tor(nb_classes=5, D=8, F2=64, Chans=30, kernLength=300, Samples=500,
                           dropoutRate=0.5)
        trainer = Trainer_uni(model=model, data=data, lr=1e-5, batch_size=32, num_epochs=200)
        trainer.train()
        model.eval()

        criterion = nn.CrossEntropyLoss()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        te_x_eeg = torch.tensor(te_x_eeg, dtype=torch.float32).to(device)
        te_y_eeg = torch.tensor(te_y_eeg, dtype=torch.long).to(device)
        model.to(device)

        with torch.no_grad():
            scores = model(te_x_eeg)
            predictions = scores.argmax(dim=1)
            correct = (predictions == te_y_eeg).sum().item()
            total = te_y_eeg.size(0)
            accuracy = correct / total
        print(accuracy)
        '''
        
        # hyperparameters to select from
        n_estimators_params = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250]
        max_depth_params = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        
        X_train = np.concatenate((train_x_eeg, val_x_eeg))
        y_train = np.concatenate((train_y_eeg, val_y_eeg))
        
        # initialize results vector
        cv_results = np.zeros(len(n_estimators_params)*len(max_depth_params))
        
        for i in n_estimators_params:
            for j in max_depth_params:
                clf = RandomForestClassifier(n_estimators = i, max_depth = j)

                with open("EEG RESULTS.txt", "a") as f:
                    f.write('n_estimators: ' + str(i) + ', max_depth: ' + str(j) + '\n')
                cv_results[int(i/5*2 + j/10 - 11)] = np.mean(cross_val_score(clf, X_train, y_train, cv=5))
                with open("EEG RESULTS.txt", "a") as f:
                    f.write(str(cv_results[int(i/5*2 + j/10 - 11)]) + '\n')
                
                
        # determine maximum score
        max_score_id = np.min(np.where(cv_results == np.max(cv_results)))
        best_n_estimators = int(np.ceil((max_score_id+1)/10)*25)
        best_max_depth = int(((max_score_id % 10) + 1) * 10)
        
        with open("EEG RESULTS.txt", "a") as f:
            f.write('BEST n_estimators: ' + str(best_n_estimators) + ', BEST max_depth: ' + str(best_max_depth) + '\n')
        
        # fit to training and validation sets
        clf = RandomForestClassifier(n_estimators = best_n_estimators, max_depth = best_max_depth)
        clf.fit(X_train, y_train)
        
        pred_y_eeg = clf.predict(test_x_eeg)
        with open("EEG RESULTS.txt", "a") as f:
            f.write(str(pred_y_eeg) + '\n' + str(test_y_eeg) + '\n' + str(clf.score(test_x_eeg, test_y_eeg)) + '\n\n')


''' Direct evaluation
if __name__ == "__main__":
    eeg_loader = DataLoadEEG(subject=1, band=[0.5, 45], fs_orig=500, fs_target=100,
                             parent_directory='C://Users//minho.lee//Dropbox//Datasets//EAV')
    data_eeg, data_eeg_y = eeg_loader.data_prepare()

    division_eeg = EAVDataSplit(data_eeg, data_eeg_y)
    [tr_x_eeg, tr_y_eeg, te_x_eeg, te_y_eeg] = division_eeg.get_split()
    data = [tr_x_eeg, tr_y_eeg, te_x_eeg, te_y_eeg]

    trainer = Transformer_EEG.EEGModelTrainer(data, lr=0.001, batch_size = 64)
    trainer.train(epochs=200, lr=None, freeze=False)
'''
'''
from Transformer_EEG import EEGClassificationModel
accuracy_all = list()
prediction_all = list()
if __name__ == "__main__": # from pickle data
    import pickle
    for sub in range(1, 43):
        file_path = "C:/Users/minho.lee/Dropbox/Datasets/EAV/Input_images/EEG/"
        file_name = f"subject_{sub:02d}_eeg.pkl"
        file_ = os.path.join(file_path, file_name)

        with open(file_, 'rb') as f:
            eeg_list2 = pickle.load(f)
        tr_x_eeg, tr_y_eeg, te_x_eeg, te_y_eeg = eeg_list2
        data = [tr_x_eeg, tr_y_eeg, te_x_eeg, te_y_eeg]

        model = EEGClassificationModel(eeg_channel=30)
        trainer = Transformer_EEG.EEGModelTrainer(data, model = model, lr=0.001, batch_size = 64)
        trainer.train(epochs=100, lr=None, freeze=False)

        [accuracy, predictions] = trainer.evaluate()
        accuracy_all.append(accuracy)
        prediction_all.append(predictions)
'''

