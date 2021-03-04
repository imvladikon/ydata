#!/usr/bin/env python
# coding: utf-8

# # Transfer Learning for Computer Vision Tutorial
# **Extended from [code](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) by [Sasank Chilamkurthy](https://chsasank.github.io).**  
# License: BSD.
# 
# In this tutorial, you will learn how to train a convolutional neural network for
# image classification using transfer learning. You can read more about the transfer
# learning at the [CS231N notes](https://cs231n.github.io/transfer-learning/).
# 
# Quoting these notes:
# 
#     In practice, very few people train an entire Convolutional Network
#     from scratch (with random initialization), because it is relatively
#     rare to have a dataset of sufficient size. Instead, it is common to
#     pretrain a ConvNet on a very large dataset (e.g. ImageNet, which
#     contains 1.2 million images with 1000 categories), and then use the
#     ConvNet either as an initialization or a fixed feature extractor for
#     the task of interest.
# 
# These two major transfer learning scenarios look as follows:
# 
# -  **Finetuning the convnet**: Instead of random initializaion, we
#    initialize the network with a pretrained network, like the one that is
#    trained on imagenet 1000 dataset. Rest of the training looks as
#    usual.
# -  **ConvNet as fixed feature extractor**: Here, we will freeze the weights
#    for all of the network except that of the final fully connected
#    layer. This last fully connected layer is replaced with a new one
#    with random weights and only this layer is trained.
# 
# 
# 

# ### Running notes
# If you are running this in Google Colab, be sure to change the runtime to GPU by clicking `Runtime > Change runtime type` and selecting "GPU" from the dropdown menu.

# ### Import relevant packages

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import warnings
# warnings.filterwarnings("ignore", category=UserWarning)
# we want to find exif problematic images


# In[ ]:


from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

plt.ion()   # interactive mode


# ## Load Data
# 
# We will use `torchvision` and `torch.utils.data` packages for loading the
# data.
# 
# The problem we're going to solve today is to train a model to classify
# present and future **Israeli politicians**. We have about ~100 training images for each of our 9 politicians, scraped from the first page of the Google Images Search.
# Usually, this is a very small dataset to generalize upon, if trained from scratch (or is it? We'll soon test this ourselves!). Since we are using transfer learning, we should be able to generalize reasonably well.

# In[ ]:


# Create a folder for our data
get_ipython().system('mkdir data')
get_ipython().system('mkdir data/israeli_politicians')


# In[ ]:


# Download our dataset and extract it
import requests
from zipfile import ZipFile

url = 'https://github.com/omriallouche/ydata_deep_learning_2021/blob/main/data/israeli_politicians.zip?raw=true'
r = requests.get(url, allow_redirects=True)
open('./data/israeli_politicians.zip', 'wb').write(r.content)

with ZipFile('./data/israeli_politicians.zip', 'r') as zipObj:
   # Extract all the contents of zip file in current directory
   zipObj.extractall(path='./data/israeli_politicians/')


# In[ ]:





# In[ ]:


#searching problem files and delete them
import glob, os
import PIL.Image

def get_files(folder, extension):
  for filename in glob.iglob('{}/**/*{}'.format(folder, extension), recursive=True):
    if os.path.isfile(filename):
        yield filename

for f in get_files("./data/israeli_politicians", "jpg"):
  with warnings.catch_warnings(record=True) as w:
     img = PIL.Image.open(f)
     exif_data = img._getexif()
     try:
       print(str(w[-1].message))
       print(f)
       os.remove(f)
     except:
       pass


# In[ ]:


# On Linux machines, the following simpler code can be used instead
# !wget https://github.com/omriallouche/ydata_deep_learning_2021/blob/main/data/israeli_politicians.zip?raw=true
# !unzip israeli_politicians.zip -d data/israeli_politicians/


# ### Datasets definition
# PyTorch uses DataLoaders to define datasets. We'll create 2 data loaders, `train` and `val` (for validation).  
# Our dataset was already split into different folders for these - as you can see under the "Files" menu on the left of the Colab.

# In[ ]:


means = [0.485, 0.456, 0.406]
stds = [0.229, 0.224, 0.225]

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(means, stds)
    ]),
    'val': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(means, stds)
    ]),
}


# In[ ]:


data_dir = r'./data/israeli_politicians/'

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {
    'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=16,
                                             shuffle=True, num_workers=4),
    'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=16,
                                          shuffle=False, num_workers=4)
  }
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
print('dataset_sizes: ', dataset_sizes)

class_names = image_datasets['train'].classes
print('class_names:', class_names)


# In[ ]:


# Check for the availability of a GPU, and use CPU otherwise
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device


# ### Datasets and Dataloaders
# Let's examine the dataloaders and datasets and learn more about their attributes and functions.  

# In[ ]:


train_dataloader = dataloaders['train']


# In Colab or Jupyter notebook, if we type  `train_dataloader.` and wait, we'd see a drop-down with the object attributes and functions.  
# 
# `train_dataloader.dataset.samples` contains the filenames + true labels (0 to 8 for our 9 classes).  
# `train_dataloader.dataset.classes` contains the class names in order.  
# `train_dataloader.dataset.class_to_idx` contains a map from a class name to the integer that represents it.

# In[ ]:


train_dataloader.dataset.class_to_idx


# ### Visualize a few images
# 
# Let's visualize a few training images so as to understand the data
# augmentations.
# 
# 

# In[ ]:


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array(means)
    std = np.array(stds)
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    fig = plt.figure(figsize=(5,3), dpi=300)
    plt.imshow(inp)
    if title is not None:
       plt.title(title, fontsize=5)
    plt.pause(0.001)  # pause a bit so that plots are updated


# In[ ]:


# Get a batch of training data - the data loader is a generator
inputs, classes = next(iter(dataloaders['train']))


# `inputs` is a tensor of shape `[16, 3, 256, 256]` - there are 16 images in the batch, each with 3 color channels (RGB), a width of 256 pixels, and a height of 256 pixels:

# In[ ]:


inputs.shape


# `classes` is a tensor of size 16, containing numbers matching the true class of each image, from our 9 classes.

# In[ ]:


classes


# To map it to the class names, we can run:

# In[ ]:


[class_names[c] for c in classes]


# In[ ]:


# Make a grid from batch
out = torchvision.utils.make_grid(inputs, nrow=8)

imshow(out, title='\n'.join([class_names[x] for x in classes]))


# ### Using a pretrained model
# Let's load a model pretrained on ImageNet, and use it for our task.  
# The first time we load the model, it will be downloaded locally, and then cached for the future.

# In[ ]:


# We load a pretrain model with its weights. Alternatively, one might want to only load the model architecture.
model = models.vgg16(pretrained=True)


# We can print the model to learn about its structure:

# In[ ]:


model


# Keras has a useful model summary, that we can also use for PyTorch models:

# In[ ]:


get_ipython().system('pip install torchsummary ')
from torchsummary import summary
summary(model.to(device), (3, 256, 256))


# We can see above that the final layer is a fully connected layer, from 4096 neurons to 1000 neurons, with a total of `4096 * 1000 + 1000 = 4,097,000` parameters. We add 1000 weights for the bias term of each output neuron.  
# 
# So in total, VGG-16 has 138M parameters, and we've set them all to be trainable.
# 
# However, VGG-16 and other networks created for ImageNet all have 1,000 neurons in their output, since they classify images to one of the 1,000 categories in the ImageNet challenge.  
# To use the network for our task, we need to replace the final fully-connected layer with one mapping to 9 neurons instead of 1000:

# In[ ]:


# This code should output the number of input neurons to the final layer. We'll use it to create a new layer instead of it.
last_layer = list(model.children())[-1]
if hasattr(last_layer, 'in_features'):
  num_ftrs = last_layer.in_features
else:
  num_ftrs = last_layer[-1].in_features

num_ftrs


# To access and set a layer in the network, we reference it by name. In the example above, the final layer is `model.classifier[6]`.

# In[ ]:


# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model.classifier[6]


# Let's now replace it with a linear layer with 4096 input features and 9 output features:

# In[ ]:


model.classifier[6] = nn.Linear(in_features=4096, out_features=9)


# Let's review our change in the model:

# In[ ]:


model


# In[ ]:


summary(model.to(device), (3, 256, 256))


# Next, we define the loss, optimizer and LR scheduler.

# In[ ]:


# If a GPU is available, make the model use it
model = model.to(device)

# For a multi-class problem, you'd usually prefer CrossEntropyLoss()
criterion = nn.CrossEntropyLoss()

# Use Stochastic Gradient Descent as the optimizer, with a learning rate of 0.001 and momentum
optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

num_epochs = 10


# Training the model
# ------------------
# 
# Now, let's write a general function to train a model. Here, we will
# illustrate:
# 
# -  Scheduling the learning rate
# -  Saving the best model
# 
# In the following, parameter ``scheduler`` is an LR scheduler object from
# ``torch.optim.lr_scheduler``.
# 
# 

# In[ ]:


def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    # Init variables that will save info about the best model
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                # Set model to training mode. 
                model.train()  
            else:
                # Set model to evaluate mode. In evaluate mode, we don't perform backprop and don't need to keep the gradients
                model.eval()   

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                # Prepare the inputs for GPU/CPU
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # ===== forward pass ======
                with torch.set_grad_enabled(phase=='train'):
                    # If we're in train mode, we'll track the gradients to allow back-propagation
                    outputs = model(inputs) # apply the model to the inputs. The output is the softmax probability of each class
                    _, preds = torch.max(outputs, 1) # 
                    loss = criterion(outputs, labels)

                    # ==== backward pass + optimizer step ====
                    # This runs only in the training phase
                    if phase == 'train':
                        loss.backward() # Perform a step in the opposite direction of the gradient
                        optimizer.step() # Adapt the optimizer

                # Collect statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            if phase == 'train':
                # Adjust the learning rate based on the scheduler
                scheduler.step()  

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Keep the results of the best model so far
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                # deepcopy the model
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# ## Finetuning the network
# 
# 
# 

# ### Train and evaluate
# It should take around 15-25 min on CPU. On GPU though, it takes less than a minute.

# In[ ]:


model = train_model(model, 
                    dataloaders,
                       criterion, 
                       optimizer_ft, 
                       exp_lr_scheduler,
                       num_epochs=num_epochs)


# ### Visualizing the model predictions
# 
# Let's define a generic function to display our model's predictions for a few images:
# 
# 
# 

# In[ ]:


def visualize_model(model, num_images=6):    
    # Record the train/evaluate mode of the model, to restore it after we're done
    was_training = model.training
    # Set the model mode to evaluate
    model.eval()
    
    images_so_far = 0
    plt.figure(figsize=(2,1), dpi=300)

    with torch.no_grad(): # No need to collect gradients when generating predictions
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1

                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
                
        # Restore the model's train/evaluate mode
        model.train(mode=was_training)


# In[ ]:


visualize_model(model)


# # Additional Tasks
# In the sections below, you'll experiment with PyTorch and NN yourself.  
# 
# Before you start having fun on your own, let's review together how to get some predictions from our trained model.  
# 
# As we've done in both our training and visualization code, we apply the model to input with the command:  
# ```python
# outputs = model(inputs)
# ```
# 
# The relevant code:

# In[ ]:


inputs, labels = next(iter(dataloaders['train']))
inputs = inputs.to(device)
print(inputs)

labels = labels.to(device)
print(labels)


# Note that we got tensors. We can convert them into numpy objects. Since we've previously set them to use GPU, We'll need to set them to use CPU first:

# In[ ]:


labels.cpu().numpy()


# In[ ]:


outputs = model(inputs)

print(outputs.shape)
print(outputs)


# Note that `outputs` have 16 "rows", one for each of the images in our batch, and each "row" has 9 values, for each of our classes.  
# The values are the activations of the neurons, before applying the softmax function to them. To apply the softmax function to them, we can run:

# In[ ]:


torch.nn.functional.softmax(outputs, dim=1).cpu().detach().numpy()


# ## Task 1: Plot model convergence
# Adapt the code above to plot the loss and the accuracy every epoch. Show both the training and the validation performance.  
# Does our model overfit?  
# Do you have suggestions following these graphs? 

# we implemented couple helpers and abstract model class, that plot learning curves on each epoch, it's easier to check training process (we didn't learn tensorboard yet)

# ## Helper functions

# In[ ]:


import pandas as pd
import gc
from IPython.display import clear_output
import PIL
from torch import nn

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
plt.style.use('seaborn')
get_ipython().run_line_magic('config', 'InlineBackend.print_figure_kwargs={\'facecolor\' : "w"}')

def clear_cache():
  gc.collect()
  if torch.cuda.is_available():
    torch.cuda.empty_cache()

def setup_seed(rnd_state=42):
  import torch
  import random
  random.seed(rnd_state)
  np.random.seed(rnd_state)
  torch.manual_seed(rnd_state)
  if torch.cuda.is_available():
      torch.cuda.manual_seed_all(rnd_state)

setup_seed()
torch.backends.cudnn.benchmark = True


# In[ ]:


class AbstractModel(nn.Module):
  def __init__(self, cls2idx, pretrained=True, feature_extracting=False,  *args, **kwargs):
    super(AbstractModel, self).__init__()
    self.metric_history = {"train":[], "val":[]} 
    self.cls2idx = cls2idx
    self.num_classes= len(cls2idx)
    self.idx2cls = {v:k for k,v in self.cls2idx.items()}
    self.pretrained = pretrained
    self.feature_extracting=feature_extracting
    

  @torch.no_grad()
  def predict_image(self, img, return_class=False):
    self.eval()
    if isinstance(img, str):
      img = PIL.Image.open(img)
    if not torch.is_tensor(img):
      img = data_transforms["val"](img)
    img = img.unsqueeze(0).to(device)
    outputs = self(img) 
    if return_class:
      _, preds = torch.max(outputs, 1) 
      return self.idx2cls[preds.item()]
    else:
      return outputs

  def __freeze_pretrained__(self):
    for layer in list(self.children()):
      for param in layer.parameters():
        param.requires_grad = False
  
  def init_weights(self, modules=None):
    if not hasattr(self, "detector") or self.detector is None: return
    if self.pretrained: return
    modules = self.detector.modules() if modules is None else modules
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight.data)
            # nn.init.kaiming_normal_(m.weight.data, mode='fan_out')
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()

  
  @torch.no_grad()
  def predict(self, test_dataloaders):
    self.eval()
    clear_cache()
    ret = np.zeros((len(test_dataloaders), 1))
    for idx, inputs in enumerate(test_dataloaders):
        if isinstance(inputs, list):
          inputs = inputs[0]
        inputs = inputs.to(device)
        outputs = self(inputs) 
        _, preds = torch.max(outputs, 1) 
        ret[idx] = preds.item()
    return ret
 
  def fit(self, dataloaders, criterion, optimizer, scheduler, num_epochs=25, do_plot=True):
    self.metric_history = {"train":[], "val":[]} 
    model = self
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        clear_cache()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  
            else:
                model.eval()   
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(inputs) 
                    _, preds = torch.max(outputs, 1) 
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward() 
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            if phase == 'train':
                if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                  scheduler.step(epoch_loss)
                else:
                  scheduler.step()  
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            self.metric_history[phase].append({"epoch": epoch,"loss":epoch_loss, "acc":epoch_acc.item()})
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        print()
        if do_plot:
          self.plot_learning_curves(title='Epoch {}/{}'.format(epoch, num_epochs - 1))
        
    time_elapsed = time.time() - since
    print(f'Training complete in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    model.load_state_dict(best_model_wts)
    if do_plot:
      self.plot_learning_curves()
    return self.metric_history

  def forward(self, in_data):
    x = self.detector(in_data)
    return x

  def summary(self):
    return summary(self, (3, 256, 256))

  def plot_learning_curves(self, do_clear=True, title=""):
    if do_clear:
      clear_output(wait=True)
    with plt.style.context('seaborn-whitegrid'):
      fig,ax = plt.subplots(1,2, figsize=(16, 6))
      train_history = pd.DataFrame(self.metric_history["train"])
      val_history = pd.DataFrame(self.metric_history["val"])
      train_history.plot(x="epoch", y="acc", ax=ax[0], color="r", label="acc_train") 
      val_history.plot(x="epoch", y="acc", ax=ax[0], color="b", label="acc_val")
      train_history.plot(x="epoch", y="loss", color="r", ax=ax[1], label="loss_train")
      val_history.plot(x="epoch", y="loss", color="b", ax=ax[1], label="loss_val")
      ax[0].set_title(f'Train Acc: {train_history.iloc[-1]["acc"]:.4f} Val Acc: {val_history.iloc[-1]["acc"]:.4f}')
      ax[1].set_title(f'Train Loss: {train_history.iloc[-1]["loss"]:.4f} Val Loss: {val_history.iloc[-1]["loss"]:.4f}')
      if not title:
        fig.suptitle(title)
      plt.show();


# In[ ]:


class BaselineVGG16(AbstractModel):
  def __init__(self, cls2idx, pretrained=True, feature_extracting=False, *args, **kwargs):
    super(BaselineVGG16, self).__init__(cls2idx, pretrained=pretrained, feature_extracting=feature_extracting, *args, **kwargs)
    self.detector = models.vgg16(pretrained=self.pretrained)
    last_layer = list(self.detector.children())[-1]
    if hasattr(last_layer, 'in_features'):
        num_ftrs = last_layer.in_features
    else:
        num_ftrs = last_layer[-1].in_features
    if self.feature_extracting:
      self.__freeze_pretrained__()
    # we change classifier after freezing, hence last layer should be trainable, it happens in abstract class
    self.detector.classifier[6] = nn.Linear(in_features=num_ftrs, out_features=self.num_classes)


# In[ ]:


cls2idx = train_dataloader.dataset.class_to_idx
model = BaselineVGG16(cls2idx=cls2idx, pretrained=True, feature_extracting=False).to(device)
optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
criterion = nn.CrossEntropyLoss()
num_epochs = 10
history = model.fit(dataloaders, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs)


# As we see, after 4 epoch our model is overfitted, validation loss is flat while training loss continues to decrease (same with on accuracy comparisons).
# 
# What could we do? first of all try check what will happen if we try to handle the problem of the imbalanced classes

# In[ ]:


import glob2 as glob
from pathlib import Path
import seaborn as sns

samples=[]
for f in ["./data/israeli_politicians/train", "./data/israeli_politicians/val"]:
  for img in glob.glob(f+"/**/*.jpg", recursive=True):
    label = os.path.basename(Path(img).parent)
    samples.append({"filename":img, "label":label, "phase": os.path.basename(f)})
samples = pd.DataFrame(samples)


# In[ ]:


sns.countplot(y=samples[samples["phase"]=="train"]["label"])


# In[ ]:


sns.countplot(y=samples[samples["phase"]=="val"]["label"])


# let's try to sample according weights of the samples per classes

# In[ ]:


from torch.utils.data import WeightedRandomSampler

def check_weights_experiment():
    """
    we do not want change global vars, let's compute it separately
    """
    def get_weights_to_fix(data, num_classes=9):
        amounts = [0] * num_classes
        for _, lable in data:
            amounts[lable] += 1
        tot_samples = sum(amounts)
        ratios = [tot_samples / amounts[i] for i in range(num_classes)]
        weights = [ratios[j] for _, j in data]
        return weights

    data_dir = r'./data/israeli_politicians/'

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}

    weights = get_weights_to_fix(image_datasets['train'])
    train_sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights))

    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=16,
                                            num_workers=4, sampler=train_sampler),
        'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=16,
                                              shuffle=False, num_workers=4)
      }
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    print('dataset_sizes: ', dataset_sizes)

    class_names = image_datasets['train'].classes
    print('class_names:', class_names)
    train_dataloader = dataloaders['train']

    model = BaselineVGG16(cls2idx=cls2idx).to(device)
    optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    num_epochs = 10
    history = model.fit(dataloaders, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs)
    return model, history


# In[ ]:


check_weights_experiment();


# It doesn't seem to improve the results by much, and it's reasonable as we use weight sampler on training set but, not on validation set. After looking at the images themselves, we found out we have not just small amount of the data, but also different distributions / nature of the pictures train vs val sets, - multiple faces per class, like common photos - Bibi with Gantz, where model could not infer what is difference between them and on small val data, and cost of error per sample on validation is too high.

# besides lr scheduler, which reduces the learning rate over times, there is another way that we can reduce the learning rate by some factors (say 10) if the validation loss stops improving in some epochs (say 5) and stop the training process if the validation loss stops improving in some epochs (say 10). this can be done easily by using ReduceLROnPlateau

# In[ ]:


def check_reduceplateu_sheduler_experiment():
    model = BaselineVGG16(cls2idx=cls2idx).to(device)
    optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # we tried also this one:
    # exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=20)
    exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft)
    criterion = nn.CrossEntropyLoss()
    num_epochs = 10
    history = model.fit(dataloaders, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs)
    return model, history


# In[ ]:


check_reduceplateu_sheduler_experiment();


# Model using changes of LR doesn't significantly improve the results, but we see instability in training and learning curves. It seems that we already have limitation of the architecture on this small data. We should mention that VGG16 doesn't have batch normalizations, so we also could try use just small batch_size like = 1. Another technics of the avoiding small data problem is augmentation like horisontals flipping, small rotations - some changes that could not impact on face recognition in a bad way (we will use it later)

# ### Discussion of the results
# 
# As expected for a very complex model with 134M of parameters trained on a very small dataset, the model overfits. We can easily see it by the fact that training loss is very small and training accuracy is close to 100%, while validation loss and accuracy are quite far from these results.
# 
# Our suggesions following the charts is either to provide more training examples (by getting more labeled images or applying augmentations) or try a simpler model with less parameters, that might have less overfitting and might generalize better. We could also try to use a model that was pre-trained on human faces and not on generic objects.
# 
# It also seems that training the model for much more epochs would not be helpful, as training loss is already very small and has almost stabilized.

# ## Task 2: Evaluate the model performance
# Write code that shows the performance of our model.  
# Show the classification report and the confusion matrix.  
# 
# What can you say about the model so far?  
# Can you suggest a few ideas to improve it?

# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix


# > for simplicity we will use val as test data for evaluating perfomance

# In[ ]:


test_dataloaders = torch.utils.data.DataLoader(datasets.ImageFolder(os.path.join(data_dir, "val"),
                                          data_transforms["val"]), batch_size=1,
                                          shuffle=False, num_workers=4)
y_pred = model.predict(test_dataloaders)
y_true = np.array([cls.item() for _, cls in test_dataloaders])


# In[ ]:


# let's do it in a simple way without heatmap
# sns.heatmap(confusion_matrix(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))


# In[ ]:


y_pred.shape, y_true.shape


# In[ ]:


y_pred = y_pred[:, 0]


# In[ ]:


from sklearn.metrics import ConfusionMatrixDisplay

def plot_confusion_matrix(labels, pred_labels, classes):
    fig = plt.figure(figsize = (16, 16));
    ax = fig.add_subplot(1, 1, 1);
    cm = confusion_matrix(labels, pred_labels);
    cm = ConfusionMatrixDisplay(cm, display_labels = classes);
    cm.plot(values_format = 'd', cmap = 'Blues', ax = ax)
    fig.delaxes(fig.axes[1]) #delete colorbar
    plt.xticks(rotation = 90)
    plt.xlabel('Predicted Label', fontsize = 50)
    plt.ylabel('True Label', fontsize = 50)


# In[ ]:


plot_confusion_matrix(y_true, y_pred, list(cls2idx.keys()))


# In[ ]:


print(classification_report(y_true, y_pred, target_names=list(cls2idx.keys())))


# * as we see on 3 samples of kostya_kilimnik we got bad accuracy and high recall - because we have small amount of the images.
# * Benny Gantz is often classified as Bibi, because they have common photos. 
# * Ayelet Shaked has 100% precision, we assume it's because she has good features for distinguishing her (long hair). It makes sense also for Danny Danon (he has glasses on the train pictures) 
# 
# * what we could improve. first of all clean data:
#    1. remove photos with wrong labels (e.g. there is photo with label Bibi, but it's - Jonathan Pollard )
#    2. remove photos with more than 1 faces (Bibi with Gantz. what is right prediction of it?)
#    3. remove low resolution photos and raw quality photo
#    4. try to crop faces as part of the preprocessing 

# ## Task 3: Perform Error Analysis 
# Error Analysis is an extremely important practice in Machine Learning research. It is rare that our base model works great out of the box. Proper error analysis helps us detect and fix issues in our DL code, data preprocessing and even in the data itself.

# In[ ]:


from torch.nn import functional as F


# In[ ]:


predictions_vs_labels = []
for i, row in samples[samples["phase"]=="val"].iterrows():
   outputs = F.softmax(model.predict_image(row["filename"]), dim=1)
   proba, preds = torch.max(outputs, 1) 
   predicted =  model.idx2cls[preds.item()]
   proba = proba.item()
   predictions_vs_labels.append({"label":row["label"], "proba":proba, "predicted":predicted, "filename":row["filename"]})


# In[ ]:


predictions_vs_labels = pd.DataFrame(predictions_vs_labels)


# In[ ]:


incorrect_predictions = predictions_vs_labels[predictions_vs_labels["label"]!=predictions_vs_labels["predicted"]]
incorrect_predictions.sample(5)


# In[ ]:


import PIL


# In[ ]:


def plot_most_incorrect(incorrect, classes, n_images, for_class=""):
    if for_class:
      incorrect = incorrect[:][incorrect["label"]==for_class].reset_index()

    rows = int(np.sqrt(n_images))
    cols = int(np.sqrt(n_images))
    fig = plt.figure(figsize = (25, 20))
    for i in range(rows*cols):
        if len(incorrect)<=i: break
        ax = fig.add_subplot(rows, cols, i+1)
        
        row = incorrect.iloc[i]
        filename, true_class, incorrect_class, incorrect_prob = row["filename"], row["label"], row["predicted"], row["proba"]
        image = data_transforms["val"](PIL.Image.open(filename))
        filename = os.path.basename(filename)
        true_prob = 0.0

        inp = image.numpy().transpose((1, 2, 0))
        mean = np.array(means)
        std = np.array(stds)
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        
        ax.imshow(inp)
        ax.set_title(f'{filename}\n'                      f'true label: {true_class} \n'                      f'pred label: {incorrect_class} ({incorrect_prob:.3f})')
        ax.axis('off');
        
    fig.subplots_adjust(hspace=0.4);


# In[ ]:


plot_most_incorrect(incorrect_predictions, list(cls2idx.keys()), n_images=20, for_class="benjamin_netanyahu")


# In[ ]:


plot_most_incorrect(incorrect_predictions, list(cls2idx.keys()), n_images=20, for_class="gideon_saar")


# In[ ]:


plot_most_incorrect(incorrect_predictions, list(cls2idx.keys()), n_images=20, for_class="danny_danon")


# In[ ]:


plot_most_incorrect(incorrect_predictions, list(cls2idx.keys()), n_images=20, for_class="benny_gantz")


# we found incorrect label - image191.jpg - it's not Bibi

# let's check common photo of Bibi and Gantz together (predicted: Bibi , label:Gantz) and visualize the activation of the last feature layer

# In[ ]:


get_ipython().run_cell_magic('capture', '', '\n!pip install pytorch-gradcam')


# In[ ]:


from gradcam.utils import visualize_cam
from gradcam import GradCAM, GradCAMpp
layer_name='features_29' # last CNN layer

img_path = "./data/israeli_politicians/val/benny_gantz/image384.jpg"
pil_img = PIL.Image.open(img_path)

torch_img = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])(pil_img).to(device)
normed_torch_img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(torch_img)[None].to(device)

vgg = model.detector.to(device)
configs = [dict(model_type='vgg', arch=vgg, layer_name='features_29')]
cams = [
    [cls.from_config(**config) for cls in (GradCAM, GradCAMpp)]
    for config in configs
]
images = []
for gradcam, gradcam_pp in cams:
    mask, _ = gradcam(normed_torch_img)
    heatmap, result = visualize_cam(mask, torch_img)

    mask_pp, _ = gradcam_pp(normed_torch_img)
    heatmap_pp, result_pp = visualize_cam(mask_pp, torch_img)
    
    images.extend([torch_img.cpu(), heatmap, heatmap_pp, result, result_pp])
    
grid_image = torchvision.utils.make_grid(images, nrow=1)

transforms.ToPILImage()(grid_image)


# as we can see, our model concentrates mostly on Bibi features, that's why it's predicted as Bibi. 

# #### Review examples of top errors
# One of the basic techniques of Error Analysis is manually reviewing the top errors of the model - samples where the model was most confident about one class, but the true label was different.  
# Plot the top 10 errors of the model for each true class.  
# Do you spot any issue or pattern?  
# Try to see if you can improve the performance of the model following your insights.

# *  as we see, our model indeed make wrong predictions sometimes because of common photo, we see that attention of our model on Bibi instead of Gantz. One of the solution is using detector, like - https://github.com/timesler/facenet-pytorch.
# *  also make sense check acc@top_k
# *  we tried to remove photos with several faces and it slightly improved the model accuracy

# ## Task 4: NN as a fixed feature extractor
# 
# Here, we need to freeze all the network except the final layer. We need
# to set ``requires_grad == False`` to freeze the parameters so that the
# gradients are not computed in ``backward()``.
# 
# You can read more about this in the documentation [here](https://pytorch.org/docs/notes/autograd.html#excluding-subgraphs-from-backward).
# 
# 
# 

# In[ ]:


model = BaselineVGG16(cls2idx=cls2idx, feature_extracting=True).to(device)
optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
criterion = nn.CrossEntropyLoss()
num_epochs = 10
history = model.fit(dataloaders, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs)


# let's check how many trainable params we had for to be sure, that it was indeed feature extractions model where everything except last one layer is freezing up

# In[ ]:


model.summary()


# as we see indeed Trainable params: 36,873 and Linear-39 has 36,873 params

# we got worse results, but the training / valication curves are more stable, which makes sense, since we use a pre-trained model as feature extractions and train only last linear layer. This layer is not sufficiently complex or deep  to learn effectively.

# also we tried catboost and embeddings from model and got similiar results
# 
# ![image.png](data:image/png;base64,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)

# but problem here, that embeddings and features are not good enough, as we could see only Aeled Shaked has good cluster according to UMAP on embeddings
# 
# ![image.png](data:image/png;base64,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)

# ### Train and evaluate
# 
# On CPU this will take about half the time compared to previous scenario. This is expected as gradients don't need to be computed for most of the network. However, forward does need to be computed.

# In[ ]:


visualize_model(model.detector)

plt.ioff()
plt.show()


# ## Task 5: Comparison with a model trained from scratch
# In this section, compare the performance of the pretrained model with those of a model trained from scratch.  
# Plot the train and validation loss and accuracy with time.
# 
# Note: To do this, you will need to modify `train_model()` to return the validation accuracy.

# In[ ]:


model = BaselineVGG16(cls2idx=cls2idx, pretrained=False).to(device)
optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
criterion = nn.CrossEntropyLoss()
num_epochs = 10
history = model.fit(dataloaders, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs)


# ### Discussion of the results
# 
# * We can see that the model starts with poor results (23-25% vs 36%-46% in pre-trained model). That is reasonable, as pre-trained weights have a lot of helpful information (that's what transfer learning is all about).
# * We can see that model results after 10 training epochs are better (27%), but are still very far even from initial results of pre-trained model, let along its final results (82-83% on validation set).
# * We can see that the model is not yet in over-fitted state and we should contiune to train it for large number of epochs (it is not stabilized yet and both training and validation loss are still improving).

# ## Task 6: Data Augmentation
# Manually labeling can be expensive, both in terms of money and of time. Data augmentations serve to increase the amount of data available for the classifier without requiring labeling more images.  
# 
# The torch vision package allows easy augmentation of images using the data transforms.  
# Use and adapt the code below to try different augmentations, and discuss the results and the model improvements you got from these augmentations.
# 
# [This guide](https://www.analyticsvidhya.com/blog/2019/12/image-augmentation-deep-learning-pytorch/) might help you along the way.

# In[ ]:


# redefine the transform to inlcude augmentations, but only for trainset
import PIL

data_transforms_augmentations = {
    'train': transforms.Compose([
        transforms.Resize((256, 256)),
        torchvision.transforms.ColorJitter(hue=.05, saturation=.05),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomRotation(15, resample=PIL.Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(means, stds)
    ]),
    'val': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(means, stds)
    ]),
}


# In[ ]:


# redefine the dataloades using the new trasnform

image_datasets_aug = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms_augmentations[x]) for x in ['train', 'val']}
dataloaders_aug = {
    'train': torch.utils.data.DataLoader(image_datasets_aug['train'], batch_size=16, shuffle=True, num_workers=4),
    'val': torch.utils.data.DataLoader(image_datasets_aug['val'], batch_size=16, shuffle=False, num_workers=4)
  }


# In[ ]:


model = BaselineVGG16(cls2idx=cls2idx, pretrained=True).to(device)
optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
criterion = nn.CrossEntropyLoss()
num_epochs = 10
history = model.fit(dataloaders_aug, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs)


# As expected, these augmentations reduced training accuracy, as due to augmentations each epoch received slightly different samples, making it harder for the model to "memorize" them and overfit. Unfortunately, it still didn't improve validation accuracy (best accuracy was 82.5% in original model and in this model with augmented data). We might see improvement in future epochs, but in both models validation acuracy / loss plots look almost flat in the last epochs, so I wouldn't count on it.
# 
# Let's try another augmentation transformation - perhaps this time we will have more luck.

# In[ ]:


data_transforms_augmentations = {
    'train': transforms.Compose([
        transforms.Resize((350, 350)),
        transforms.CenterCrop((300, 300)),
        transforms.RandomCrop((256, 256)),
        torchvision.transforms.ColorJitter(hue=.05, saturation=.05),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomRotation(10, resample=PIL.Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(means, stds)
    ]),
    'val': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(means, stds)
    ]),
}


# In[ ]:


# redefine the dataloades using the new trasnform

image_datasets_aug = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms_augmentations[x]) for x in ['train', 'val']}
dataloaders_aug = {
    'train': torch.utils.data.DataLoader(image_datasets_aug['train'], batch_size=16, shuffle=True, num_workers=4),
    'val': torch.utils.data.DataLoader(image_datasets_aug['val'], batch_size=16, shuffle=False, num_workers=4)
  }


# In[ ]:


inputs, classes = next(iter(dataloaders_aug['train']))
out = torchvision.utils.make_grid(inputs, nrow=8)

imshow(out)


# In[ ]:


model = BaselineVGG16(cls2idx=cls2idx, pretrained=True).to(device)
optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
criterion = nn.CrossEntropyLoss()
num_epochs = 10
history = model.fit(dataloaders_aug, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs)


# This, more complex, augmentation makes it even harder for the training accuracy to increase, but unfortunately, it also didn't improve the validation accuracy, resulting in ~79% vs original ~83%.

# ## Task 7: Model Architectures
# In the course we'll cover in depth various DL architectures suggested to perform image classification. Among other things, these networks differ in depth (the number of layers), the number of weights (the network power), the composing layers, and more.  
# In the figure below, you can see the performance of different network architectures on the ImageNet Image Classification task, and the number of flops (atomic computations) required for them. 
# 
# ![CNN performance/flops graph](https://miro.medium.com/max/1838/1*n16lj3lSkz2miMc_5cvkrA.jpeg)

# In our code above we've used the `mobilenet` architecture.  
# See if you can increase model performance by using alternative architectures from [torchvision.models](https://pytorch.org/vision/0.8/models.html).  
# Pay attention to the input dimensions that each network architecture expects.

# In[ ]:


# desired input of Resnet18 is 224x224
data_transforms_224 = {
    'train': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(means, stds)
    ]),
    'val': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(means, stds)
    ]),
}
image_datasets_224 = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms_224[x]) for x in ['train', 'val']}
dataloaders_224 = {
    'train': torch.utils.data.DataLoader(image_datasets_224['train'], batch_size=16, shuffle=True, num_workers=4),
    'val': torch.utils.data.DataLoader(image_datasets_224['val'], batch_size=16, shuffle=False, num_workers=4)
  }


# In[ ]:


class MobileNetv2Model(AbstractModel):
  def __init__(self, cls2idx, *args, **kwargs):
    super(MobileNetv2Model, self).__init__(cls2idx, *args, **kwargs)
    self.detector = models.mobilenet_v2(pretrained=True)
    last_layer = list(self.detector.children())[-1]
    if hasattr(last_layer, 'in_features'):
        num_ftrs = last_layer.in_features
    else:
        num_ftrs = last_layer[-1].in_features
    if self.feature_extracting:
      self.__freeze_pretrained__()
    self.detector.classifier[1] = nn.Linear(num_ftrs, len(class_names))


# In[ ]:


model = MobileNetv2Model(cls2idx=cls2idx).to(device)
optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
criterion = nn.CrossEntropyLoss()
num_epochs = 10
history = model.fit(dataloaders_224, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs)


# In[ ]:


class Resenet50Model(AbstractModel):
  def __init__(self, cls2idx, *args, **kwargs):
    super(Resenet50Model, self).__init__(cls2idx, *args, **kwargs)
    self.detector = models.wide_resnet50_2(pretrained=True)
    last_layer = list(self.detector.children())[-1]
    if hasattr(last_layer, 'in_features'):
        num_ftrs = last_layer.in_features
    else:
        num_ftrs = last_layer[-1].in_features
    if self.feature_extracting:
      self.__freeze_pretrained__()
    self.detector.fc = nn.Linear(num_ftrs, len(class_names))


# In[ ]:


model = Resenet50Model(cls2idx=cls2idx).to(device)
optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
criterion = nn.CrossEntropyLoss()
num_epochs = 10
history = model.fit(dataloaders, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs)


# In[ ]:


class Inception3Model(AbstractModel):
  def __init__(self, cls2idx, *args, **kwargs):
    super(Inception3Model, self).__init__(cls2idx, *args, **kwargs)
    self.detector = models.inception_v3(pretrained=True)
    self.detector.aux_logits=False  # we don't process aux_loss on the train stage
    last_layer = list(self.detector.children())[-1]
    if hasattr(last_layer, 'in_features'):
        num_ftrs = last_layer.in_features
    else:
        num_ftrs = last_layer[-1].in_features
    if self.feature_extracting:
      self.__freeze_pretrained__()
    # we change classifier after freezing, hence last layer should be trainable
    self.detector.fc = nn.Linear(num_ftrs, len(cls2idx))


# In[ ]:


def run_inception_experiment():
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(means, stds)
        ]),
        'val': transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(means, stds)
        ]),
    }

    data_dir = r'./data/israeli_politicians/'

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=16,
                                                shuffle=True, num_workers=4),
        'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=16,
                                              shuffle=False, num_workers=4)
      }
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    print('dataset_sizes: ', dataset_sizes)

    class_names = image_datasets['train'].classes
    print('class_names:', class_names)


    model = Inception3Model(cls2idx=cls2idx).to(device)
    optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    num_epochs = 10
    history = model.fit(dataloaders, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs)
    return model, history

run_inception_experiment();


# Using much smaller models witl less parameters, we have managed to reach similiar validation accuracy

# ## Task 8: Design your own Neural Network Architecture
# Take a stab at building your own NN architecture.  
# To allow you to experiment quickly, we'll limit it to 8 layers max, and up to 10 million parameters. Use [this PyTorch guide](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html) for reference.  
# Train it only on our provided images - we'll present the winner with the best results on the validation set!

# In[ ]:


class RakBibiNet(AbstractModel):
    def __init__(self, cls2idx, *args, **kwargs):
        super(RakBibiNet, self).__init__(cls2idx, *args, **kwargs)
        self.cls2idx = cls2idx
        self.num_classes = len(cls2idx)
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.num_classes),
        )
        self.init_weights(self.features.modules())
        self.init_weights(self.classifier.modules())
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# In[ ]:


model = RakBibiNet(cls2idx=cls2idx).to(device)
optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
criterion = nn.CrossEntropyLoss()
num_epochs = 20
history = model.fit(dataloaders_aug, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs)


# In[ ]:


model.summary()


# Unfortunately, our model is "stuck" and does not improve over ~25%

# # Task 9: Experiments

# 1. new without batch-normalization model, without pretrained weights

# In[ ]:


get_ipython().run_cell_magic('capture', '', '!pip install git+https://github.com/rwightman/pytorch-image-models')


# In[ ]:


import timm

class NFNetModel(AbstractModel):
    def __init__(self, cls2idx, *args, **kwargs):
        super(NFNetModel, self).__init__(cls2idx, *args, **kwargs)
        self.cls2idx = cls2idx
        self.detector = timm.create_model('nfnet_f1', pretrained=self.pretrained)
        self.detector.head.fc = nn.Linear(self.detector.head.fc.in_features, len(cls2idx))

model = NFNetModel(cls2idx=cls2idx).to(device)
optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
criterion = nn.CrossEntropyLoss()
num_epochs = 10
history = model.fit(dataloaders, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs)


# it's a bit better then our rakbibi model. everything is better then rakbibi

# 2. Label smoothing

# In[ ]:


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self):
        super(LabelSmoothingCrossEntropy, self).__init__()
    def forward(self, x, target, smoothing=0.1):
        confidence = 1. - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss.mean()


# In[ ]:


model = NFNetModel(cls2idx=cls2idx).to(device)
optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
criterion = LabelSmoothingCrossEntropy()
num_epochs = 10
history = model.fit(dataloaders, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs)


# 3. mixup

# In[ ]:


@torch.no_grad()
def mixup(alpha, data, target):
      bs = data.size(0)
      c = np.random.beta(alpha, alpha)

      perm = torch.randperm(bs).cuda()

      md = c * data + (1 - c) * data[perm, :]
      mt = c * target + (1 - c) * target[perm]
      return md, mt


class MixUpWrapper(object):
    def __init__(self, alpha, dataloader):
        self.alpha = alpha
        self.dataloader = dataloader

    def mixup_loader(self, loader):
        for input, target in loader:
            i, t = mixup(self.alpha, input, target)
            yield i, t.long()

    def __iter__(self):
        return self.mixup_loader(self.dataloader)

    def __len__(self):
        return len(self.dataloader)


# In[ ]:


dataloaders["train"] = MixUpWrapper(0.1, dataloaders["train"])


# In[ ]:


model = NFNetModel(cls2idx=cls2idx).to(device)
optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
criterion = LabelSmoothingCrossEntropy()
num_epochs = 10
history = model.fit(dataloaders, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs)


# 4. using only faces, we preprocessed images and extrated only faces (there are a lot of way to do it, using face detector pretrained model or using opencv cascade classifier, we use first way, but is out of scope notebook and we decided do not provide a code for that)

# In[ ]:


get_ipython().system('gdown https://drive.google.com/uc?id=1Wl9l9I4xGAQSsf8cUSJAkJgaOK8Rt9d6')
get_ipython().system('unzip /content/data_cropped.zip')


# In[ ]:


data_dir = r'./data_cropped/'

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {
    'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=16,
                                             shuffle=True, num_workers=4),
    'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=16,
                                          shuffle=False, num_workers=4)
  }
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
print('dataset_sizes: ', dataset_sizes)

class_names = image_datasets['train'].classes
print('class_names:', class_names)


# In[ ]:


inputs, classes = next(iter(dataloaders['train']))
out = torchvision.utils.make_grid(inputs, nrow=8)

imshow(out, title='\n'.join([class_names[x] for x in classes]))


# In[ ]:


model = NFNetModel(cls2idx=cls2idx).to(device)
optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
criterion = LabelSmoothingCrossEntropy()
num_epochs = 10
history = model.fit(dataloaders, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs)


# as we see, we improved our results a bit, almost 0.3;)

# 5. also we tried - images from google search with good quality, removing photos where several faces. 
