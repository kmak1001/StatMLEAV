import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForAudioClassification
from torch.utils.data import DataLoader, TensorDataset
from transformers import ASTFeatureExtractor
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import copy
import os

class AudioModelTrainer:
    def __init__(self, DATA, model_path, sub = '', num_classes=5, weight_decay=1e-5, lr=0.001, batch_size=128, log_file=None):

        self.tr, self.tr_y, self.te, self.te_y = DATA
        self.tr_x = self._feature_extract(self.tr)
        self.te_x = self._feature_extract(self.te)

        self.sub = sub
        self.batch_size = batch_size
        self.log_file = log_file
        self.train_dataloader = self._prepare_dataloader(self.tr_x, self.tr_y, shuffle=True)
        self.test_dataloader = self._prepare_dataloader(self.te_x, self.te_y, shuffle=False)

        self.model = AutoModelForAudioClassification.from_pretrained(model_path)
        # Modify classifier to fit the number of classes
        self.model.classifier.dense = torch.nn.Linear(self.model.classifier.dense.in_features, num_classes)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        # Setup optimizer and loss function
        self.initial_lr = lr
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.initial_lr)
        self.loss_fn = nn.CrossEntropyLoss()

    def _prepare_dataloader(self, x, y, shuffle=False):
        dataset = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
        return dataloader

    def _feature_extract(self, x):
        feature_extractor = ASTFeatureExtractor()
        ft = feature_extractor(x, sampling_rate=16000, padding='max_length',
                               return_tensors='pt')
        return ft['input_values']



    def train_with_early_stopping(self, max_epochs=30, patience=5, freeze=True, lr=None, fold=0):
        lr = lr if lr is not None else self.initial_lr
        if lr is not None:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
                
        best_val_acc = 0.0
        best_model_state = None
        epochs_no_improve = 0
        best_epoch = 0
    
        #log_file = f'cv_training_logs/training_performance_{self.sub}.txt'
        log_file = 'mbibars/coding/EAV/cv_training_logs/training_performance_AllSubjectsCV.txt'
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
        # Set trainable parameters
        for param in self.model.parameters():
            param.requires_grad = not freeze
        for param in self.model.classifier.parameters():
            param.requires_grad = True
    
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
    
        all_preds, all_trues = [], []
    
        for epoch in range(max_epochs):
            self.model.train()
            train_correct, train_total = 0, 0
            for x, t in self.train_dataloader:
                x, t = x.to(self.device), t.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(x).logits
                loss = self.loss_fn(logits, t)
                loss.backward()
                self.optimizer.step()
    
                train_correct += (logits.argmax(dim=-1) == t).sum().item()
                train_total += t.size(0)
    
            train_acc = train_correct / train_total
    
            # Validation
            self.model.eval()
            val_preds, val_trues = [], []
            with torch.no_grad():
                for x, t in self.test_dataloader:
                    x = x.to(self.device)
                    logits = self.model(x).logits
                    val_preds.extend(logits.argmax(dim=-1).cpu().numpy())
                    val_trues.extend(t.numpy())
    
            val_acc = accuracy_score(val_trues, val_preds)
    
            # Save log
            with open(log_file, 'a') as f:
                f.write(f"{self.sub}, Fold {fold}, Epoch {epoch + 1}, Train ACC: {train_acc:.4f}, Val ACC: {val_acc:.4f}\n")
                f.flush()
            
            print(f"{self.sub}, Fold {fold}, Epoch {epoch + 1}, Train ACC: {train_acc:.4f}, Val ACC: {val_acc:.4f}\n")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = copy.deepcopy(self.model.state_dict())
                best_preds = val_preds.copy()
                best_trues = val_trues.copy()
                best_epoch = epoch + 1
                epochs_no_improve = 0
                print(f"!!!!!!!!!!!!! Val Accuracy improved to {best_val_acc}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"!!!!!!!!!!!!!! [{self.sub}] Early stopping at epoch {epoch + 1}")
                    break
    
        self.model.load_state_dict(best_model_state)
    
        # Final evaluation
        final_acc = accuracy_score(best_trues, best_preds)
        final_f1 = f1_score(best_trues, best_preds, average='weighted')
        with open(log_file, 'a') as f:
            f.write(f"{self.sub}, Fold {fold}, Best Epoch: {best_epoch}, Final Val ACC: {final_acc:.4f}, Final Val F1: {final_f1:.4f}\n\n")
            f.flush()
    
        return best_epoch, final_acc, final_f1



    def train(self, epochs=20, lr=None, freeze=True):
        print(f"Training model for {epochs} epochs...")
        with open(self.log_file, 'a') as f:
            f.write(f"Training model for {epochs} epochs...\n")
            f.flush()
        lr = lr if lr is not None else self.initial_lr
        if lr is not None:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

        if isinstance(self.model, nn.DataParallel):
            self.model = self.model.module
        # Freeze or unfreeze model parameters based on the freeze flag
        print(f"Freeze model parameters: {freeze}")
        with open(self.log_file, 'a') as f:
            f.write(f"Freeze model parameters: {freeze}\n")
            f.flush()
        for param in self.model.parameters():
            param.requires_grad = not freeze
        for param in self.model.classifier.parameters():
            param.requires_grad = True
        
        # Wrap the model with DataParallel
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        print(f"Using {torch.cuda.device_count()} GPUs for training.")
        print("Model training started...")

        with open(self.log_file, 'a') as f:
            f.write(f"Using {torch.cuda.device_count()} GPUs for training.\n")
            f.write("Model training started...\n")
            f.flush()
        ############## Training loop ##############
        for epoch in range(epochs):
            self.model.train()
            train_correct, train_total = 0, 0
            print(f"Epoch {epoch + 1}/{epochs}")
            with open(self.log_file, 'a') as f:
                f.write(f"Epoch {epoch + 1}/{epochs}\n")
                f.flush()

            total_batches = len(self.train_dataloader)
            for batch_idx, batch in enumerate(self.train_dataloader, start=1):
                #print(f'batch ({batch_idx}/{total_batches})')

                x, t = [b.to(self.device) for b in batch]
                self.optimizer.zero_grad()
                logits = self.model(x).logits
                loss = self.loss_fn(logits, t)
                if loss.dim() > 0:
                    loss = loss.mean()
                else:
                    loss = loss
                loss.backward()
                self.optimizer.step()

                train_correct += (logits.argmax(dim=-1) == t).sum().item()
                train_total += t.size(0)
            train_accuracy = train_correct / train_total
            print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
            with open(self.log_file, 'a') as f:
                f.write(f"Training Accuracy: {train_accuracy * 100:.2f}%\n")
                f.write(f"Testing model...\n")
                f.flush()
            
            print(f"Testing model...")
            ############## Testing loop ##############
            self.model.eval()
            correct, total = 0, 0
            outputs_batch = []
            with torch.no_grad():
                for x, t in self.test_dataloader:
                    x, t = x.to(self.device), t.long().to(self.device)
                    logits = self.model(x).logits
                    correct += (logits.argmax(dim=-1) == t).sum().item()
                    total += t.size(0)

                    logits_cpu = logits.detach().cpu().numpy()
                    outputs_batch.append(logits_cpu)
                test_accuracy = correct / total
            if epoch == epochs-1 and not freeze: # we saved test prediction only at last epoch, and finetuning
                self.outputs_test = np.concatenate(outputs_batch, axis=0)
                print(f"Test predictions shape: {self.outputs_test.shape}")
                with open(self.log_file, 'a') as f:
                    f.write(f"Test predictions shape: {self.outputs_test.shape}\n")
                    f.flush()
            print("Evaluation complete")
            with open(self.log_file, 'a') as f:
                f.write(f"Evaluation complete\n")
                f.write(f"{self.sub}, Epoch {epoch + 1}, Test Accuracy: {test_accuracy * 100:.2f}%\n")
                f.flush()
            print(f"Epoch {epoch + 1}/{epochs}, Training Accuracy: {train_accuracy * 100:.2f}%, Test Accuracy: {test_accuracy * 100:.2f}%")
                


