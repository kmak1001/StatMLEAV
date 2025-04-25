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

if __name__ == "__main__":
    all_f1_scores = []
    all_accuracies = []
    aggregated_cm = np.zeros((5, 5))  # Assuming 5 classes
    results_dict = {}

    results_file = "cv_audio_classification_results.txt"
    results_pickle = "cv_audio_classification_results.pkl"

    with open(results_file, "w") as f:
        f.write("Per-subject results:\n")
        
    subject_cv_accs = []
    subject_cv_f1s = []
    
    for sub in range(1, 43):
        print("="*30, f"Subject_{sub:02d}", "="*30)
        file_path = "/opt/scratchspace/mbibars/eeg/data/Input_images/Audio"
        file_name = f"subject_{sub:02d}_aud.pkl"
        file_ = os.path.join(file_path, file_name)
        
        with open(file_, 'rb') as f:
            Aud_list = pickle.load(f)
        [tr_x_aud, tr_y_aud, te_x_aud, te_y_aud] = Aud_list
        
        data = [tr_x_aud, tr_y_aud, te_x_aud, te_y_aud]
        mod_path = "/home/mbibars/mbibars/coding/EAV/Pre_trained_models/ast-finetuned-audioset"

        # **STEP 1: 5-FOLD CROSS-VALIDATION ON TRAINING DATA**
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        best_epochs_1 = []
        best_epochs_2 = []
        fold_accs = []
        fold_f1s = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(tr_x_aud, tr_y_aud)):
            print(f"-----------Starting fold {fold}--------------")
            tr_x_fold, tr_y_fold = tr_x_aud[train_idx], tr_y_aud[train_idx]
            val_x_fold, val_y_fold = tr_x_aud[val_idx], tr_y_aud[val_idx]

            fold_data = [tr_x_fold, tr_y_fold, val_x_fold, val_y_fold]
            Trainer = Transformer_Audio.AudioModelTrainer(
                fold_data, model_path=mod_path, sub=f"subject_{sub:02d}",
                num_classes=5, weight_decay=1e-5, lr=0.005, batch_size=8
            )

            best_n_epochs_1, val_acc, val_f1= Trainer.train_with_early_stopping(
                max_epochs=50, patience=10, lr=5e-4, freeze=True, fold=fold
            )
            best_n_epochs_2, val_acc, val_f1= Trainer.train_with_early_stopping(
                max_epochs=50, patience=10,lr=5e-6, freeze=False, fold=fold
            )
            best_epochs_1.append(best_n_epochs_1)
            best_epochs_2.append(best_n_epochs_2)
            fold_accs.append(val_acc)
            fold_f1s.append(val_f1)

        # Save average CV scores for the subject
        with open(f'cv_training_logs/training_performance_subject_{sub:02d}.txt', 'a') as f:
            f.write(f"{'='*20}\nSubject {sub:02d} CV Results:\n")
            f.write(f"Median Epochs 1: {np.median(best_epochs_1):.2f}\n")
            f.write(f"Median Epochs 2: {np.median(best_epochs_2):.2f}\n")
            f.write(f"Mean ACC: {np.mean(fold_accs):.4f}, Mean F1: {np.mean(fold_f1s):.4f}\n")
            f.write(f"{'='*20}\n\n")
        
        # **STEP 2: Train on Full Training Set with Median(n_epochs)**
        median_n_epochs_1 = int(np.median(best_epochs_1))
        median_n_epochs_2 = int(np.median(best_epochs_2))

        mean_acc = np.mean(fold_accs)
        mean_f1 = np.mean(fold_f1s)

        subject_cv_accs.append(mean_acc)
        subject_cv_f1s.append(mean_f1)
        
        final_trainer = Transformer_Audio.AudioModelTrainer(
            data, model_path=mod_path, sub=f"subject_{sub:02d}",
            num_classes=5, weight_decay=1e-5, lr=0.005, batch_size=8
        )
        final_trainer.train(epochs=median_n_epochs_1, lr=5e-4, freeze=True)
        final_trainer.train(epochs=median_n_epochs_2, lr=5e-6, freeze=False)

        # **STEP 3: Test Model on Test Set**
        y_true = te_y_aud
        y_pred = np.argmax(final_trainer.outputs_test, axis=1)

        # Compute per-subject metrics
        f1 = f1_score(y_true, y_pred, average="weighted")
        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(5))  # Adjust labels if necessary

        all_f1_scores.append(f1)
        all_accuracies.append(acc)
        aggregated_cm += cm  # Sum confusion matrices across subjects

        results_dict[f"subject_{sub:02d}"] = {
            "F1-score": f1,
            "Accuracy": acc
        }

        with open(results_file, "a") as f:
            f.write(f"Subject {sub}: F1-score={f1:.4f}, Accuracy={acc:.4f}\n")

        print(f"Subject {sub}: F1-score={f1:.4f}, Accuracy={acc:.4f}")
        print(f"Confusion Matrix for Subject {sub}:\n{cm}\n")

    print(f"All folds done for this patient")
    # calculate average CV scores across patients
    final_cv_acc = np.mean(subject_cv_accs)
    final_cv_f1 = np.mean(subject_cv_f1s)
    
    with open('cv_training_logs/final_cv_results.txt', 'w') as f:
        f.write(f"Average Cross-Validation Results Across All Subjects:\n")
        f.write(f"Average CV Accuracy: {final_cv_acc:.4f}\n")
        f.write(f"Average CV F1 Score: {final_cv_f1:.4f}\n")
        
    # **STEP 4: Compute Final Aggregated Results**
    mean_f1 = np.mean(all_f1_scores)
    mean_acc = np.mean(all_accuracies)

    results_dict["Overall"] = {
        "Average F1-score": mean_f1,
        "Average Accuracy": mean_acc
    }

    with open(results_file, "a") as f:
        f.write("\nFinal Results:\n")
        f.write(f"Average F1-score across subjects: {mean_f1:.4f}\n")
        f.write(f"Average Accuracy across subjects: {mean_acc:.4f}\n")

    print("\nFinal Results:")
    print(f"Average F1-score across subjects: {mean_f1:.4f}")
    print(f"Average Accuracy across subjects: {mean_acc:.4f}")
    print(f"Aggregated Confusion Matrix:\n{aggregated_cm}")

    # **SAVE SUMMED CONFUSION MATRIX PLOT**
    plt.figure(figsize=(6, 5))
    sns.heatmap(aggregated_cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.arange(5), yticklabels=np.arange(5))
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Aggregated Confusion Matrix Across Subjects")
    plt.savefig("aggregated_confusion_matrix_ast_cv.png")
    plt.close()

    # **SAVE THE RESULTS DICTIONARY**
    with open(results_pickle, "wb") as f:
        pickle.dump(results_dict, f)
