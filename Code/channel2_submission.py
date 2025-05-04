import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import matplotlib.pyplot as plt

hp = {
    'num_conv_layers': 4,
    'conv1_channels': 53,
    'conv2_channels': 90,
    'conv3_channels': 65,
    'conv4_channels': 65,
    'kernel_size': 3,
    'stride': 1,
    'padding': 'same',
    'use_batch_norm': True,
    'slope':0.01,
    'pooling_type': 'max',
    'pooling_kernel_size': 2,
    'pooling_stride': 2,
    'dropout_conv': 0.08755577559324501,
    'dropout_fc': 0.85,
    'num_fc_layers': 2,
    'fc_neurons_1': 414,
    'fc_neurons_2': 414,
    'fc_neurons_3': 414,
    'learning_rate': 0.002278056544827262,
    'weight_decay': 0.02,
    'batch_size': 100,
    #'epochs': 250,
    'epochs': 250
}

data_signals_ids = truncated_merged_ch2_4to11ms_new["signals_id"]
data_X = truncated_merged_ch2_4to11ms_new["X_data"]
data_y = truncated_merged_ch2_4to11ms_new["y_data"]

num_classes = data_X.shape[1]
num_targets = 6

class peaksDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.signals = [torch.tensor(x, dtype=torch.float32).unsqueeze(0) for x in X_data]
        self.targets = [torch.tensor(y, dtype=torch.long) for y in y_data]
        self.signals = torch.stack(self.signals)
        self.targets = torch.stack(self.targets)
        
    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, idx):
        return self.signals[idx], self.targets[idx]

class peaksCNN(nn.Module):
    def __init__(self, signal_length, hp, num_classes, num_targets):
        super().__init__()
        
        num_conv_layers = hp['num_conv_layers']
        conv1_channels = hp['conv1_channels']
        conv2_channels = hp['conv2_channels']
        conv3_channels = hp['conv3_channels']
        conv4_channels = hp['conv4_channels']
        fc_neurons_1 = hp['fc_neurons_1']
        fc_neurons_2 = hp['fc_neurons_2']
        kernel_size = hp['kernel_size']
        stride = hp['stride']
        padding_mode = hp['padding']
        use_batch_norm = hp['use_batch_norm']
        slope = hp['slope']
        pooling_type = hp['pooling_type']
        pooling_kernel_size = hp['pooling_kernel_size']
        pooling_stride = hp['pooling_stride']
        dropout_conv = hp['dropout_conv']
        dropout_fc = hp['dropout_fc']
        
        if padding_mode == 'same':
            padding = kernel_size // 2
            
        else:
            padding = 0
            
        conv_layers = []
        conv_layers.append(nn.Conv1d(1, conv1_channels, kernel_size=kernel_size, stride=stride, padding=padding))
        
        if use_batch_norm:
            conv_layers.append(nn.BatchNorm1d(conv1_channels))
        
        conv_layers.append(nn.LeakyReLU(negative_slope=slope))
        if pooling_type == 'max':
            conv_layers.append(nn.MaxPool1d(kernel_size=pooling_kernel_size, stride=pooling_stride))
        
        else:
            conv_layers.append(nn.AvgPool1d(kernel_size=pooling_kernel_size, stride=pooling_stride))
        
        if dropout_conv > 0:
            conv_layers.append(nn.Dropout(dropout_conv))
        
        if num_conv_layers >= 2:
            
            conv_layers.append(nn.Conv1d(conv1_channels, conv2_channels, kernel_size=kernel_size, stride=stride, padding=padding))
            if use_batch_norm:
                conv_layers.append(nn.BatchNorm1d(conv2_channels))
            
            conv_layers.append(nn.LeakyReLU(negative_slope=slope))
            if pooling_type == 'max':
                conv_layers.append(nn.MaxPool1d(kernel_size=pooling_kernel_size, stride=pooling_stride))
            
            else:
                conv_layers.append(nn.AvgPool1d(kernel_size=pooling_kernel_size, stride=pooling_stride))
            
            if dropout_conv > 0:
                conv_layers.append(nn.Dropout(dropout_conv))
        
        if num_conv_layers >= 3:
            
            conv_layers.append(nn.Conv1d(conv2_channels, conv3_channels, kernel_size=kernel_size, stride=stride, padding=padding))
            if use_batch_norm:
                conv_layers.append(nn.BatchNorm1d(conv3_channels))
            
            conv_layers.append(nn.LeakyReLU(negative_slope=slope))
            if pooling_type == 'max':
                conv_layers.append(nn.MaxPool1d(kernel_size=pooling_kernel_size, stride=pooling_stride))
            
            else:
                conv_layers.append(nn.AvgPool1d(kernel_size=pooling_kernel_size, stride=pooling_stride))
            
            if dropout_conv > 0:
                conv_layers.append(nn.Dropout(dropout_conv))
        
        if num_conv_layers == 4:
            
            conv_layers.append(nn.Conv1d(conv3_channels, conv4_channels, kernel_size=kernel_size, stride=stride, padding=padding))
            if use_batch_norm:
                conv_layers.append(nn.BatchNorm1d(conv4_channels))
            
            conv_layers.append(nn.LeakyReLU(negative_slope=slope))
            if pooling_type == 'max':
                conv_layers.append(nn.MaxPool1d(kernel_size=pooling_kernel_size, stride=pooling_stride))
            
            else:
                conv_layers.append(nn.AvgPool1d(kernel_size=pooling_kernel_size, stride=pooling_stride))
            
            if dropout_conv > 0:
                conv_layers.append(nn.Dropout(dropout_conv))
        
        self.conv = nn.Sequential(*conv_layers)
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, signal_length)
            dummy_output = self.conv(dummy_input)
            flattened_size = dummy_output.view(1, -1).size(1)
        
        print("Computed flattened size:", flattened_size)
        num_fc_layers = hp['num_fc_layers']
        final_out_dim = num_targets * num_classes
        
        if num_fc_layers == 1:
            fc_neurons_list = []
        elif num_fc_layers == 2:
            fc_neurons_list = [hp['fc_neurons_1']]
        elif num_fc_layers == 3:
            fc_neurons_list = [hp['fc_neurons_1'], hp['fc_neurons_2']]
        elif num_fc_layers == 4:
            fc_neurons_list = [hp['fc_neurons_1'], hp['fc_neurons_2'], hp['fc_neurons_3']]
        else:
            raise ValueError("Unsupported number of fc layers")
        
        fc_layers = []
        in_features = flattened_size
        
        for neurons in fc_neurons_list:
            fc_layers.append(nn.Linear(in_features, neurons))
            fc_layers.append(nn.LeakyReLU(negative_slope=slope))
            
            if dropout_fc > 0:
                fc_layers.append(nn.Dropout(dropout_fc))
            in_features = neurons
        fc_layers.append(nn.Linear(in_features, final_out_dim))
        
        self.fc = nn.Sequential(*fc_layers)
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x.view(x.size(0), num_targets, num_classes)
        return x

def get_loss_function(loss_name):
    if loss_name == 'CrossEntropyLoss':
        return nn.CrossEntropyLoss()
    else:
        return nn.CrossEntropyLoss()

def adjust_prediction_for_extreme(pred_idx, signal, extreme_type, window, target_idx, prev_extreme):

    if hasattr(pred_idx, 'item'):
        pred_idx = pred_idx.item()
    if hasattr(prev_extreme, 'item'):
        prev_extreme = prev_extreme.item()
    
    lower_bound = max(pred_idx - window, prev_extreme + 1, 1)
    upper_bound = min(pred_idx + window, len(signal) - 2)
    
    if extreme_type == 'max':
        if (pred_idx - 1 >= 0 and pred_idx + 1 < len(signal) and 
            signal[pred_idx] > signal[pred_idx - 1] and signal[pred_idx] > signal[pred_idx + 1]):
            return pred_idx
    elif extreme_type == 'min':
        if (pred_idx - 1 >= 0 and pred_idx + 1 < len(signal) and 
            signal[pred_idx] < signal[pred_idx - 1] and signal[pred_idx] < signal[pred_idx + 1]):
            return pred_idx
    
    valid_indices = []
    for i in range(lower_bound, upper_bound + 1):
        if i - 1 < 0 or i + 1 >= len(signal):
            continue
        if extreme_type == 'max' and signal[i] > signal[i - 1] and signal[i] > signal[i + 1]:
            valid_indices.append(i)
        elif extreme_type == 'min' and signal[i] < signal[i - 1] and signal[i] < signal[i + 1]:
            valid_indices.append(i)
    
    if valid_indices:
        new_idx = min(valid_indices, key=lambda i: abs(i - pred_idx))
        return new_idx
    else:
        return pred_idx



def train_model_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    
    for signals, targets in train_loader:
        signals, targets = signals.to(device), targets.to(device)
        optimizer.zero_grad()
        
        outputs = model(signals)
        loss = criterion(outputs.view(-1, num_classes), targets.view(-1))
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * signals.size(0)
    return running_loss / len(train_loader.dataset)

def validate_model(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for signals, targets in val_loader:
            signals, targets = signals.to(device), targets.to(device)
            
            outputs = model(signals)
            loss = criterion(outputs.view(-1, num_classes), targets.view(-1))
            
            running_loss += loss.item() * signals.size(0)
    return running_loss / len(val_loader.dataset)

dataset = peaksDataset(data_X, data_y)
print("Total Dataset size:", len(dataset))
num_samples = len(dataset)
indices = list(range(num_samples))
random.shuffle(indices)

k = 10
fold_size = num_samples // k
folds = []

for i in range(k):
    if i == k - 1:
        fold = indices[i * fold_size:]
    
    else:
        fold = indices[i * fold_size : (i + 1) * fold_size]
    folds.append(fold)

fold_results = []
all_fold_losses = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
signal_length = data_X.shape[1]
fold_number = 1

for i in range(k):
    print(f"\nStarting fold {fold_number}")
    test_indices = folds[i]
    train_indices = [idx for j, fold in enumerate(folds) if j != i for idx in fold]
    
    X_train = [data_X[idx] for idx in train_indices]
    y_train = [data_y[idx] for idx in train_indices]
    X_test= [data_X[idx] for idx in test_indices]
    y_test= [data_y[idx] for idx in test_indices]
    
    test_signal_ids = [data_signals_ids[idx] for idx in test_indices]
    
    train_dataset = peaksDataset(X_train, y_train)
    test_dataset = peaksDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=hp['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    model = peaksCNN(signal_length, hp, num_classes, num_targets).to(device)
    criterion = get_loss_function("CrossEntropyLoss")
    optimizer = optim.AdamW(model.parameters(), lr=hp['learning_rate'], weight_decay=hp['weight_decay'])
    
    
    train_losses = []
    for epoch in range(hp['epochs']):
        train_loss = train_model_epoch(model, train_loader, optimizer, criterion, device)
        
        train_losses.append(train_loss)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Fold {fold_number}, Epoch {epoch+1}/{hp['epochs']}, Training loss: {train_loss:.6f}")
    
    all_fold_losses.append(train_losses)
    
    #t=np.linspace(2.990078005464481, 12.00127199453552, 220)
    #t=np.linspace(3.9321573770491804,12.00127199453552,197)
    #t = np.linspace(3.9321573770491804, 11.05919262295082, 174)
    t = np.linspace(3.9321573770491804, 11.263954918032788, 179)
    model.eval()
    total_test_loss = 0.0
    num_batches = 0
    correct_targets = 0
    total_targets = 0
    
    with torch.no_grad():
        for signals, targets in test_loader:
            signals, targets = signals.to(device), targets.to(device)
            outputs = model(signals)
            
            loss = criterion(outputs.view(-1, num_classes), targets.view(-1))
            total_test_loss += loss.item()
            
            num_batches += 1
            predicted_indices = outputs.argmax(dim=2)
            signal_np = signals[0,0,:].cpu().numpy()
            
            adjusted_preds = []
            prev_extreme = 0  
            window_size = 10 
            
            pred_flat = predicted_indices.flatten()
            
            for j in range(num_targets):
                orig_pred = pred_flat[j].item()  
                if j in [0, 2, 4]:
                    adjusted = adjust_prediction_for_extreme(orig_pred, signal_np, 'max', window_size, j, prev_extreme)
                elif j in [1, 3, 5]:
                    adjusted = adjust_prediction_for_extreme(orig_pred, signal_np, 'min', window_size, j, prev_extreme)
                else:
                    adjusted = orig_pred
                adjusted_preds.append(adjusted)
                # prev_extreme = adjusted
            
            for j in range(targets.shape[1]):
                target_idx = targets[0, j].item()
                if abs(adjusted_preds[j] - target_idx) <= 2:
                    correct_targets += 1
                total_targets += 1

    avg_test_loss = total_test_loss / num_batches if num_batches > 0 else 0
    percent_accuracy = (correct_targets / total_targets) * 100 if total_targets > 0 else 0
    

    print(f"Fold {fold_number} evaluation:")
    print(f"Average test loss: {avg_test_loss:.6f}")
    print(f"Correct targets: {correct_targets} out of {total_targets} ({percent_accuracy:.2f}%)")
    
    fold_results.append((avg_test_loss, percent_accuracy))
    
    for idx in range(len(test_dataset)):
        signal, target = test_dataset[idx]
        signal_np = signal.squeeze().cpu().numpy()
        
        with torch.no_grad():
            output = model(signal.unsqueeze(0).to(device))
        predicted_indices = output.argmax(dim=2).squeeze(0).cpu().numpy()
        adjusted_preds = []
        prev_extreme = 0   
        window_size = 20 
        
        for j in range(num_targets):
            orig_pred = predicted_indices[j]
            if j in [0, 2, 4]:
                adjusted = adjust_prediction_for_extreme(orig_pred, signal_np, 'max', window_size, j, prev_extreme)
            elif j in [1, 3, 5]:
                adjusted = adjust_prediction_for_extreme(orig_pred, signal_np, 'min', window_size, j, prev_extreme)
            else:
                adjusted = orig_pred
            adjusted_preds.append(adjusted)
            prev_extreme = adjusted

        incorrect_points = []
        
        for j in range(num_targets):
            if abs(adjusted_preds[j] - target[j].item()) > 2:
                incorrect_points.append(j)
        
        if incorrect_points:
            plt.figure()
            plt.plot(t, signal_np)
           
            pred_x = [t[adjusted_preds[j]] for j in range(num_targets)]
            pred_y = [signal_np[adjusted_preds[j]] for j in range(num_targets)]
            targ_x = [t[target[j].item()] for j in range(num_targets)]
            targ_y = [signal_np[target[j].item()] for j in range(num_targets)]
            
            plt.scatter(pred_x, pred_y, color='blue', marker='o', label="Predicted", s=15)
            plt.scatter(targ_x, targ_y, color='red', marker='x', label="Target", s=15)
            plt.xlabel("Time")
            plt.ylabel("Amp")
            plt.legend()
            plt.title(f"{test_signal_ids[idx]} -- {test_indices[idx]} - incorrect: {incorrect_points}")
            plt.show()
    fold_number += 1

avg_loss = np.mean([fr[0] for fr in fold_results])
avg_accuracy = np.mean([fr[1] for fr in fold_results])
print("\nK-fold cross validation Results:")
print(f"Average test loss across folds: {avg_loss:.6f}")
print(f"Average accuracy across folds: {avg_accuracy:.2f}%")


# for idx, losses in enumerate(all_fold_losses, start=1):
#     plt.figure()
#     plt.plot(range(1, len(losses)+1), losses)
#     plt.title(f'Fold {idx} Training Loss per Epoch')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.grid(True)


# plt.figure()
# for idx, losses in enumerate(all_fold_losses, start=1):
#     plt.plot(range(1, len(losses)+1), losses, label=f'Fold {idx}')

# plt.title('Channel 2: Loss per Epoch')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid(True)
# plt.ylim(0.5,6)
# plt.show()


# percent_accuracies = [acc for (_, acc) in fold_results]
# fold_indices= list(range(1, len(percent_accuracies) + 1))

# plt.figure()
# plt.plot(fold_indices, percent_accuracies, marker='o', linestyle='-')
# plt.title('Channel 2: Accuracy per Fold')
# plt.xlabel('Fold')
# plt.ylabel('Accuracy (%)')
# plt.xticks(fold_indices) 
# plt.grid(True)
# plt.ylim(88,100)
# plt.show()




# #final training on entire dataset before saving state
# full_dataset = peaksDataset(data_X, data_y)
# full_loader = DataLoader(full_dataset, batch_size=hp['batch_size'], shuffle=True)

# final_model = peaksCNN(signal_length, hp, num_classes, num_targets).to(device)

# optimizer = optim.AdamW(final_model.parameters(), lr=hp['learning_rate'], weight_decay=hp['weight_decay'])
# criterion = get_loss_function("CrossEntropyLoss")

# for epoch in range(hp['epochs']):
#     train_loss = train_model_epoch(final_model, full_loader, optimizer, criterion, device)
#     if (epoch + 1) % 10 == 0 or epoch == 0:
#         print(f"Final Model, Epoch {epoch+1}/{hp['epochs']}, Training Loss: {train_loss:.6f}")


# torch.save(final_model.state_dict(), "ch2_final_92ish.pth")
# print("Final model saved as ch2_final_92ish.pth")


# #model testing bs:
    
    
# import torch
# import time
# from torch.utils.data import DataLoader

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# final_model = peaksCNN(signal_length, hp, num_classes, num_targets).to(device)

# model_path = "ch2_final_92ish.pth"
# final_model.load_state_dict(torch.load(model_path, map_location=device))
# final_model.eval()


# test_dataset = peaksDataset(data_X, data_y)
# test_loader = DataLoader(test_dataset, batch_size=hp['batch_size'], shuffle=False)

# total_inference_time = 0.0
# total_signals = 0
# all_predictions = [] 

# with torch.no_grad():
#     for batch_idx, (inputs, labels) in enumerate(test_loader):
#         inputs = inputs.to(device)
#         start_time = time.perf_counter()
#         outputs = final_model(inputs)
        
#         if device.type == "cuda":
#             torch.cuda.synchronize()
        
#         end_time = time.perf_counter()
#         batch_time = end_time - start_time

#         total_inference_time += batch_time
#         total_signals += inputs.size(0)
        
#         all_predictions.append(outputs.cpu())

# average_time_per_signal = total_inference_time / total_signals

# print(f"Total calculation time for {total_signals} signals: {total_inference_time:.6f} s")
# print(f"Average calculation time per signal: {average_time_per_signal:.6f} s")

    


