import os
import torchaudio
from torchaudio.transforms import Resample
from transformers import ASTFeatureExtractor
import pickle

from EAV_datasplit import *
from Transformer_torch import Transformer_Audio


class DataLoadAudio:
    def __init__(self, subject='all', parent_directory=r'/opt/scratchspace/mbibars/eeg/data/EAV', target_sampling_rate=16000):
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

import numpy as np
import torch
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from collections import defaultdict

if __name__ == "__main__":
    log_file = "/home/mbibars/mbibars/coding/EAV/AllSubs_Final_Model_results.txt"
    all_train_x, all_train_y = [], []
    all_test_x, all_test_y = [], []

    mod_path = "/home/mbibars/mbibars/coding/EAV/Pre_trained_models/ast-finetuned-audioset"

    # STEP 1: Aggregate data from all subjects
    for sub in range(1, 43):
        file_path = "/opt/scratchspace/mbibars/eeg/data/Input_images/Audio"
        file_name = f"subject_{sub:02d}_aud.pkl"
        file_ = os.path.join(file_path, file_name)
        
        with open(file_, 'rb') as f:
            Aud_list = pickle.load(f)
        tr_x_aud, tr_y_aud, te_x_aud, te_y_aud = Aud_list

        all_train_x.extend(tr_x_aud)
        all_train_y.extend(tr_y_aud)
        all_test_x.extend(te_x_aud)
        all_test_y.extend(te_y_aud)

    all_train_x = np.array(all_train_x)
    all_train_y = np.array(all_train_y)
    all_test_x = np.array(all_test_x)
    all_test_y = np.array(all_test_y)
    print(f"Train shape: {all_train_x.shape}")
    print(f"Test shape: {all_test_x.shape}")
    with open(log_file, "w") as f:
        f.write(f"Train shape: {all_train_x.shape}\n")
        f.write(f"Test shape: {all_test_x.shape}\n")
        f.flush()
    best_epochs_1, best_epochs_2 = [39, 44, 36, 24, 30], [37, 23, 17, 39, 33]
    fold_accs, fold_f1s = [0.7466, 0.7134, 0.7045, 0.7283, 0.7028], [0.7467,0.7130, 0.7048, 0.7283, 0.7009]
    """
    # STEP 2: Cross-validation on entire train set
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    

    for fold, (train_idx, val_idx) in enumerate(skf.split(all_train_x, all_train_y)):
        print(f"\n=========== Fold {fold} ===========")
        if fold <= 1:
            continue

        tr_x_fold = all_train_x[train_idx]
        tr_y_fold = all_train_y[train_idx]
        val_x_fold = all_train_x[val_idx]
        val_y_fold = all_train_y[val_idx]
        print(f"Train fold shape: {tr_x_fold.shape}")
        print(f"Val fold shape: {val_x_fold.shape}")
        
        fold_data = [tr_x_fold, tr_y_fold, val_x_fold, val_y_fold]
        Trainer = Transformer_Audio.AudioModelTrainer(
            fold_data, model_path=mod_path, sub="AllSubjectsCV",
            num_classes=5, weight_decay=1e-5, lr=0.005, batch_size=8
        )
        print("Model instantiated!")

        n_epochs_1, acc1, f1_1 = Trainer.train_with_early_stopping(
            max_epochs=50, patience=10, lr=5e-4, freeze=True, fold=fold
        )
        print("Model training phase 1 complete")
        n_epochs_2, acc2, f1_2 = Trainer.train_with_early_stopping(
            max_epochs=50, patience=10, lr=5e-6, freeze=False, fold=fold
        )
        print("Model training phase 2 complete")
        
        best_epochs_1.append(n_epochs_1)
        best_epochs_2.append(n_epochs_2)
        fold_accs.append(acc2)
        fold_f1s.append(f1_2)
    """
    median_n_epochs_1 = int(np.median(best_epochs_1))
    median_n_epochs_2 = int(np.median(best_epochs_2))

    print(f"\nCross-Validation Results:")
    print(f"Median n_epochs_1: {median_n_epochs_1}")
    print(f"Median n_epochs_2: {median_n_epochs_2}")
    print(f"Mean Accuracy: {np.mean(fold_accs):.4f}")
    print(f"Mean F1: {np.mean(fold_f1s):.4f}")
    with open(log_file, "a") as f:
        f.write(f"================================================================================\n")
        f.write(f"Cross-Validation Accuracy: {np.mean(fold_accs):.4f}\n")
        f.write(f"Cross-Validation F1 Score: {np.mean(fold_f1s):.4f}\n")
        f.write(f"Median n_epochs_1: {median_n_epochs_1}\n")
        f.write(f"Median n_epochs_2: {median_n_epochs_2}\n")
        f.write(f"================================================================================\n")
        f.flush()
    # STEP 3: Train final model on full training set
    full_data = [all_train_x, all_train_y, all_test_x, all_test_y]
    print("Model training on full dataset...")
    final_trainer = Transformer_Audio.AudioModelTrainer(
        full_data, model_path=mod_path, sub="AllSubjectsFinal",
        num_classes=5, weight_decay=1e-5, lr=0.005, batch_size=8, log_file=log_file
    )
    print("Final model instantiated!")
    final_trainer.train(epochs=median_n_epochs_1, lr=5e-4, freeze=True)
    print("Final model training phase 1 complete")
    with open(log_file, "a") as f:
        f.write(f"Final model training phase 1 complete\n")
        f.flush()
    final_trainer.train(epochs=median_n_epochs_2, lr=5e-6, freeze=False)
    print("Final model training phase 2 complete")
    with open(log_file, "a") as f:
        f.write(f"Final model training phase 2 complete\n")
        f.flush()

    # STEP 4: Test on full test set
    with open(log_file, "a") as f:
        f.write(f"Final model testing phase\n")
        f.flush()
    y_true = all_test_y
    y_pred = np.argmax(final_trainer.outputs_test, axis=1)

    final_f1 = f1_score(y_true, y_pred, average="weighted")
    final_acc = accuracy_score(y_true, y_pred)
    final_cm = confusion_matrix(y_true, y_pred, labels=np.arange(5))

    print(f"\nFinal Test Results:")
    print(f"Accuracy: {final_acc:.4f}")
    print(f"F1 Score: {final_f1:.4f}")
    print("Confusion Matrix:")
    print(final_cm)

    # Save results
    with open(log_file, "a") as f:
        f.write(f"================================================================================\n")
        f.write(f"Test Accuracy: {final_acc:.4f}\n")
        f.write(f"Test F1 Score: {final_f1:.4f}\n")

    with open("AllSubs_Final_Model_results.pkl", "wb") as f:
        pickle.dump({
            "CV_Accuracies": fold_accs,
            "CV_F1s": fold_f1s,
            "Test_Accuracy": final_acc,
            "Test_F1": final_f1,
            "Confusion_Matrix": final_cm
        }, f)
    print("Results saved!")
    print("================================================================================")
    print("Saving the confusion matrix...")
    # Plot confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(final_cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.arange(5), yticklabels=np.arange(5))
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix - Combined Subjects")
    plt.savefig("confusion_matrix_all_subjects.png")
    plt.close()
    print("Confusion matrix saved!")
