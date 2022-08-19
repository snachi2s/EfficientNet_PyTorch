## Model: EfficientNet (EfficientNet - B7 pre-trained model from pytorch)

- EfficientNet is a convolutional neural network architecture and scaling method that uniformly scales all dimensions of depth/width/resolution using a compound coefficient [(paper)](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwio1IeIx9P5AhVt_rsIHWZwDMEQFnoECAgQAQ&url=https%3A%2F%2Farxiv.org%2Fabs%2F1905.11946%3Fcontext%3Dstat.ML&usg=AOvVaw2GZtOwMQfRIIXUBpOkj7LZ)
- Unlike conventional practice that arbitrary scales these factors, the EfficientNet scaling method uniformly scales network width, depth, and resolution with a set of fixed scaling coefficients

- for importing efficientnet pre-trained model run

`pip install efficientnet-pytorch`


## Description
  - In this repository, efficientnet-b7 pre-trained (CNN) model is modified to output a **single value (num_classes = 1)** for given images. It can be modified for multi-class outputs by changing the last layer in self.model.\_fc2 layer into desired number of classes **(i.e) nn.Linear(128,<desired_number_of_classes>)**

  - efficientnet-b7 model is modified to give a single value output is defined in `effnet_b7.py`
  - `Early_stopping.py` implements the early stopper function which stops the training of the network if the validation loss is keep on increasing for multiple epochs. `patience (p)` is passed while calling early_stopping. During training, if validation loss increases for p number of times, training will be stopped.  
  - `LR_reducer.py` - implements the learning rate reducer accordingly with respect to the validation losses of each epoch
  - `train.py` - training and evaluation of the model
  - `datset.py` - it fetches the image using the filepath from .csv file and transforms them to required resolution using transform function from pytorch


## Model and Training

### EfficientNet-B7
 - Existing pre-trained EfficientNet CNN model is modified and used for training and validation   
 - **Modification:** Prior layers to the fc layers are freezed. Two fully connected layers are added to the existing model and only those fc layers of the network are trained (i.e) by

```python
model = modified_EfficientNet().to(device)
for param in model.parameters():
    param.requires_grad = False

for param in model.model._fc.parameters():
    param.requires_grad = True

for param in model.model._fc_2.parameters():
    param.requires_grad = True

```

