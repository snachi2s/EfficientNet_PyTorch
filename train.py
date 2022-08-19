from distutils.log import error
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from effnet_b7 import modified_EfficientNet
from LR_reducer import LRScheduler
from Early_stopper import EarlyStopping
from dataset import Unclearimage
import numpy as np
CUDA_LAUNCH_BLOCKING = "1"


#splitting of datasets into train, validation and test
# https://stackoverflow.com/questions/52697875/split-number-into-rounded-numbers

def parts(data):
    return [int((len(data)+1)*.80), int((len(data)+1)*.10), int((len(data)+1)*.10)]


class RMSELoss(nn.Module):

    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, inputs, targets):
        tmp = (inputs-targets)**2
        loss = torch.mean(tmp)
        # rmse
        loss = torch.sqrt(torch.mean(tmp))
        return loss


print("Starting training")

device = ("cuda" if torch.cuda.is_available() else "cpu")
print("using : ", device)

#transforming the image to the correspomding to the model input size
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

# parameters
num_epochs = 100
learning_rate = 0.0001
batch_size = 32
pin_memory = True
num_workers = 32
#modified_dataset_size = 1000

dataset = Unclearimage("train", "30_removed_files.csv", transform=transform)
print("Loading data...")
#dataset.annotations = dataset.annotations[0:modified_dataset_size]
print("dataset shape", dataset.annotations.shape)

# preparing the data loaders
train_set, validation_set, test_set = torch.utils.data.random_split(dataset, parts(dataset))

print("train_set shape", len(train_set))
print("validation_set shape", len(validation_set))
print("test_set_shape", len(test_set))

train_loader = DataLoader(dataset=train_set, shuffle=True,
                          batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
validation_loader = DataLoader(dataset=validation_set, shuffle=False,
                               batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
test_loader = DataLoader(dataset=test_set, shuffle=False,
                         batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

print("<...Data loading completed...>")
print("train_loader length: ", len(train_loader))
print("validation_loader length: ", len(validation_loader))
print("test_loader length: ", len(test_loader))

print("-----------------------------------------------------")
model = modified_EfficientNet().to(device)

#freezing the top layers 
for param in model.parameters():
    param.requires_grad = False

#training only the fully connected layers
for param in model.model._fc.parameters():
    param.requires_grad = True

for param in model.model._fc_2.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
lr_scheduler = LRScheduler(optimizer)

patience = 15
early_stopping = EarlyStopping(patience=patience)

## loss function
# mse = RMSELoss() 
mse = nn.MSELoss()

min_valid_loss = np.inf
epoch_array = []
train_loss_array = []
valid_loss_array = []

training_accuracy_array = []
validation_accuracy_array = []
test_accuracy_array = []


def training_loop():
    model.train()
    train_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        #normalizing labels
        labels = labels/1466
        labels = labels.to(device)
        target = model(images)
        loss = mse(target, labels)
        training_accuracy = accuracy(target, labels)
        training_accuracy_array.append(training_accuracy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0) #to compensate for batch size in last iteration

    return train_loss, training_accuracy_array


def validation_loop():
    valid_loss = 0.0
    model.eval()
    print("validating.....")
    with torch.no_grad():
        for i, (data, labels) in enumerate(validation_loader):
            data = data.to(device)
            labels = labels/1466
            labels = labels.to(device)
            target = model(data)
            loss = mse(target, labels)
            validation_accuracy = accuracy(target, labels)
            validation_accuracy_array.append(validation_accuracy)
            valid_loss += loss.item() * data.size(0)
        return valid_loss, validation_accuracy_array


def accuracy(target, labels):
    total = 0
    correct = 0
    total += labels.size(0)
    target_m = target.tolist()
    labels_m = labels.tolist()
    for (i, j) in zip(target_m, labels_m):
        if (i-j) < 0.010:
            correct += 1
    return correct/total


def test_loop(model, num_epochs, test_loader):
    model.eval()
    print("testing.....")
    with torch.no_grad():
        for i, (data, labels) in enumerate(test_loader):
            data = data.to(device)
            labels = labels/1466
            labels = labels.to(device)
            target = model(data)
            test_accuracy = accuracy(target, labels)
            test_accuracy_array.append(test_accuracy)
        return test_accuracy_array


for e in range(num_epochs):
    epoch_array.append(e)
    train_loss, accuracy_1 = training_loop()
    valid_loss, accuracy_2 = validation_loop()

    train_loss_array.append(train_loss/len(train_loader))
    valid_loss_array.append(valid_loss/len(validation_loader))

    print("Training accuracy: ", sum(accuracy_1)/len(accuracy_1))
    print("Validation accuracy: ", sum(accuracy_2)/len(accuracy_2))
    #earling stopping and learning rate reducer
    early_stopping(valid_loss)
    lr_scheduler(valid_loss)

    if early_stopping.early_stop:
        print("Early stopping")
        break
    print(f'Epoch {e+1} \t\t Training Loss: {train_loss / len(train_loader)} \t\t Validation Loss: {valid_loss / len(validation_loader)}')
    if min_valid_loss > valid_loss:
        print(
            f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f})')
        min_valid_loss = valid_loss
        #for finding the best model (!!!)
        best_accuracy = 0.0
        if (sum(accuracy_2)/len(accuracy_2)) > best_accuracy:
            torch.save(model.state_dict(),
                       'effnet_b7_best_model.pt')
            print("Saved the best model")
            best_accuracy = accuracy_2

# run the model on test set
model.load_state_dict(torch.load(
    'effnet_b7_best_model.pt'))
accuracy_3 = test_loop(model, num_epochs, test_loader)
print("Test accuracy: ", sum(accuracy_3)/len(accuracy_3))

# print(epoch_array)
# print(train_loss_array)
# print(valid_loss_array)
