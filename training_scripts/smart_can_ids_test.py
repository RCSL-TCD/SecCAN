# Start working from the lowest bops models and work your way up to find the best model that we have achieved from our design space exploration.

import torch
import math
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from brevitas.nn import QuantLinear, QuantReLU
from brevitas.quant import SignedBinaryWeightPerTensorConst
import torch.nn as nn
import brevitas.nn as bnn
import itertools
import time
import os
from scipy.spatial.distance import hamming
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_curve

# ------Setting the training device-------------------------

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
device = torch.device('cpu')
print(device)


class canTestDataset(Dataset):
    def __init__(self):
        # will be mostly used for dataloading
        x_load = np.loadtxt('./dos_test_x.txt',
                            delimiter=",", dtype=np.float32)
        y_load = np.loadtxt('./dos_test_y.txt',
                            delimiter=",", dtype=np.float32)
        self.x = torch.from_numpy(x_load)
        self.y = torch.from_numpy(y_load)
        self.n_samples = x_load.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        # Will allow us to get the length of our dataset
        return self.n_samples


testDataset = canTestDataset()
first_data = testDataset[0]
features, labels = first_data
print(features, labels)
samples = len(testDataset)
test_batch_size = 1000

# Works for a batch size of 1.
testloader = DataLoader(dataset=testDataset, batch_size=test_batch_size, shuffle=False)
# Working with a large batch size makes the process very fast.
epochs = 200
torch.manual_seed(0)
a1 = 4
a2 = 4
a3 = 4
a4 = 4
a5 = 4
# This version was trained without dropout for 200 epoch and the config was 2,2,2,2,2 and the BCE loss function.
model_folder = "./smart_can_4bit_200/"
model = nn.Sequential(
    QuantLinear(10, 64, bias=True, weight_bit_width=int(a1)),
    nn.BatchNorm1d(64),
    #nn.Dropout(0.2),
    QuantReLU(bit_width=int(a1)),
    QuantLinear(64, 32, bias=True, weight_bit_width=int(a2)),
    nn.BatchNorm1d(32),
    #nn.Dropout(0.4),
    QuantReLU(bit_width=int(a2)),
    QuantLinear(32, 1, bias=True, weight_bit_width=int(a3)),
    nn.Sigmoid()
 )

model = model.float()

max_accuracy = 0
max_index = 200
acc = 0
count_n = 0
count_n_acc = 0
count_d = 0
count_d_acc = 0
count_f = 0
count_f_acc = 0
count_r = 0
count_r_acc = 0
full_test_label = []
full_pred_label = []
roc_true = []
roc_predict = []
box_normal = np.zeros(103012)
box_dos = np.zeros(23555)
box_fuzzy = np.zeros(27859)
box_rpm = np.zeros(25040)
for j in range(156,157):
    path = model_folder+'/modelModel_'+str(j)+'.pt'
    # When trying to load on the 'cpu' which is trained on a GPU use the map_location arguement with the torch.load function.
    model.load_state_dict(torch.load(path, map_location=device),strict=False)
    count = np.zeros(7)
    model.eval()
    t1 = 0
    t2 = 0
    t3 = 0
    acc = 0
    count_n = 0
    count_n_acc = 0
    count_d = 0
    count_d_acc = 0
    count_f = 0
    count_f_acc = 0
    count_r = 0
    count_r_acc = 0
    print_count = 0
    accuracy = 0  # We report the accuracy of the model here.
    for l, (test_inputs, test_labels) in enumerate(testloader):
        t1 = t1 + time.time()
        outputs = model(test_inputs.float())
        a = outputs.detach().numpy()
        a = np.round(a)
        # In this and the above steps the tensors are converted into a numpy array to allow us to process stuff.
        b = test_labels.detach().numpy()
        a = a.reshape(test_batch_size)
        b = b.reshape(test_batch_size)
        roc_true.append(b)
        roc_predict.append(a)
        max_idx_pred = a
        # print(max_idx_pred.shape)
        max_idx_test = b
        full_test_label.append(max_idx_test)
        full_pred_label.append(max_idx_pred)
        for i in range(test_batch_size):
            if (max_idx_test[i] == max_idx_pred[i]):
                acc = acc+1
            if (max_idx_test[i] == 0):
                count_n = count_n+1
                if (max_idx_pred[i] == 0):
                    count_n_acc = count_n_acc + 1
            if (max_idx_test[i] == 1):
                count_d = count_d+1
                if (max_idx_pred[i] == 1):
                    count_d_acc = count_d_acc + 1
            if (max_idx_test[i] == 2):
                count_f = count_f+1
                if (max_idx_pred[i] == 2):
                    count_f_acc = count_f_acc + 1
                    print_count += 1
            if (max_idx_test[i] == 3):
                count_r = count_r+1
                if (max_idx_pred[i] == 3):
                    count_r_acc = count_r_acc + 1
        # We then calculate the hamming distance of the output and the actual output arrays..subtract it from the batch size and add it to the final accuracy of the model.
        accuracy = accuracy + (test_batch_size-hamming(max_idx_pred, max_idx_test)*len(max_idx_test))
        t2 = t2 + time.time()
        # This 'if' statement notes the max accuracy and the index for which we get that accuracy so that we know which is the best model among the top 50 for a given variation
    if (accuracy > max_accuracy):
        max_accuracy = accuracy
        max_index = j
    t3 = t3+t2-t1
    print("Accuracy = ", acc, "% accuracy = ", (acc/75000)*100)
    print('Total messages =', samples, 'Overall accuracy =', accuracy, 'Misclassifications = ', (samples-accuracy), 'Percentage accuracy =',
          (accuracy/samples)*100, 'Epoch =', int(j), '\n')  # Print the accuracy and the percentage accuracy here in this statement.
    print('Total Normal =', count_n, ' Correct normal =', count_n_acc,
          'Misclassifications = ', (count_n-count_n_acc), '\n')
    print('Total DoS =', count_d, ' Correct DoS =', count_d_acc,
          'Misclassifications = ', (count_d-count_d_acc), '\n')
    print('Total Fuzzing =', count_f, ' Correct Fuzzy =', count_f_acc,
          'Misclassifications = ', (count_f-count_f_acc), '\n')
    print('Total RPM =', count_r, ' Correct RPM =', count_r_acc,
          'Misclassifications = ', (count_r-count_r_acc), '\n')
    print('---------------------------')
print("Maximum accuracy index = ", max_index,
      "Max accuracy = ", (max_accuracy/samples)*100)
