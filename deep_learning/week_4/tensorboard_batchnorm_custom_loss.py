#!/usr/bin/env python
# coding: utf-8

# # Image Classification - Tensorboard, Batch Norm and Custom Loss Functions
# In this exercise, you'll continue to work with our neural network for classifying Israeli Politicians.  
# We will use tensorboard to monitor the training process and model performance.  
# 
# For the questions below, please use the network architecture you suggested in Q8 of HW1.  
# This time, we provide you with a clean dataset of Israeli Politicians, that doesn't include multiple politicians in the same image, in the folder `data/israeli_politicians_cleaned/`.

# ## Tensorboard
# TensorBoard provides visualization and tooling for machine learning experimentation:
# - Tracking and visualizing metrics such as loss and accuracy
# - Visualizing the model graph (ops and layers)
# - Viewing histograms of weights, biases, or other tensors as they change over time
# - Projecting embeddings to a lower dimensional space
# - Displaying images, text, and audio data
# - Profiling programs
# 
# Tensorboard worked originally with Tensorflow but can now be used with PyTorch as well.  
# You can embed a tensorboard widget in a Jupyter Notebook, although if you're not using Google Colab we recommend that you open tensorboard separately.

# To get started with Tensorboard, please read the following pages:
# 
# PyTorch related:
# 1. https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html
# 1. https://becominghuman.ai/logging-in-tensorboard-with-pytorch-or-any-other-library-c549163dee9e
# 1. https://towardsdatascience.com/https-medium-com-dinber19-take-a-deeper-look-at-your-pytorch-model-with-the-new-tensorboard-built-in-513969cf6a72
# 1. https://pytorch.org/docs/stable/tensorboard.html
# 1. https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/04-utils/tensorboard
# 
# Tensorflow related:
# 1. https://itnext.io/how-to-use-tensorboard-5d82f8654496
# 1. https://www.datacamp.com/community/tutorials/tensorboard-tutorial
# 1. https://medium.com/@anthony_sarkis/tensorboard-quick-start-in-5-minutes-e3ec69f673af
# 1. https://www.guru99.com/tensorboard-tutorial.html
# 1. https://www.youtube.com/watch?time_continue=1&v=s-lHP8v9qzY&feature=emb_logo
# 1. https://www.youtube.com/watch?v=pSexXMdruFM
# 

# In[ ]:


from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('logs/RegularNet') # Create Tensorboard event writer will output to the relevant folder


# ### Imports

# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy


# ### Starting Tensorboard
# Jupyter Notebook has extensions for displaying TensorBoard inside the notebook. Still, I recommend that you run it separately, as it tends to get stuck in notebooks.
# 
# The syntax to load TensorBoard in a notebook is this:
# ```python
# # Load the TensorBoard notebook extension
# %load_ext tensorboard
# %tensorboard --logdir ./logs
# ```

# In the shell, you can instead run:
# ```
# tensorboard --logdir ./logs
# ```

# In[1]:


# Load the TensorBoard notebook extension
get_ipython().run_line_magic('load_ext', 'tensorboard')

# Start TensorBoard within the notebook:
#%tensorboard --logdir ./logs


# ### Load Images

# In[ ]:


# Create a folder for our data
get_ipython().system('mkdir data')
get_ipython().system('mkdir data/israeli_politicians')


# In[ ]:


# Download our dataset and extract it
import requests
from zipfile import ZipFile

url = 'https://github.com/omriallouche/ydata_deep_learning_2021/blob/main/data/israeli_politicians_cleaned.zip?raw=true'
r = requests.get(url, allow_redirects=True)
open('./data/israeli_politicians_cleaned.zip', 'wb').write(r.content)

with ZipFile('./data/israeli_politicians_cleaned.zip', 'r') as zipObj:
   # Extract all the contents of zip file in current directory
   zipObj.extractall(path='./data/israeli_politicians/')


# In[ ]:


#searching files with currupt exif data and deleting them
import glob, os
import PIL.Image
import warnings

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


def setup_seed(rnd_state=42):
  import torch
  import random
  random.seed(rnd_state)
  np.random.seed(rnd_state)
  torch.manual_seed(rnd_state)
  if torch.cuda.is_available():
      torch.cuda.manual_seed_all(rnd_state)

setup_seed()


# In[ ]:


means = np.array([0.485, 0.456, 0.406])
stds = np.array([0.229, 0.224, 0.225])

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

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
dataloaders = {
    'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=16, shuffle=True, num_workers=2),
    'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=16, shuffle=False, num_workers=2)
  }
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
print('dataset_sizes: ', dataset_sizes)

class_names = image_datasets['train'].classes
print('class_names:', class_names)


# ### Show images using TensorBoard

# In[ ]:


# here we show images after transform, which normalizes them
images, labels = next(iter(dataloaders['train'])) # get one batch of images
grid = torchvision.utils.make_grid(images) # create a grid of these images
writer.add_image('normalized images', grid, 0)
writer.close()


# In[ ]:


def denormalize_image(img):
    img = img.numpy().transpose((1, 2, 0)) # change from 3x256x256 to 256x256x3
    img = stds * img + means # apply std&mean reverse transform on each channel
    img = np.clip(img, 0, 1) # clip the results between 0 and 1
    img = img.transpose((2, 0, 1)) # return the impage to the original form 3x256x256
    return img


# In[ ]:


def denormalize_images(images):
  result = []
  for img in images:
    result.append(denormalize_image(img)) # apply transform on one image and save it
  return torch.tensor(result)


# In[ ]:


# here we perform denormalization, showing the original images
original_images = denormalize_images(images)
grid = torchvision.utils.make_grid(original_images) # create a grid of these images
writer.add_image('original images', grid, 0)
writer.close()


# ### Define the Network

# In[ ]:


# define abstract model class with fit, predict and forward methods
class AbstractModel(nn.Module):
  def __init__(self, cls2idx, *args, **kwargs):
    super(AbstractModel, self).__init__()
    self.cls2idx = cls2idx
    self.num_classes= len(cls2idx)
    self.idx2cls = {v:k for k,v in self.cls2idx.items()}

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

  @torch.no_grad()
  def predict(self, test_dataloaders):
    self.eval()
    # clear_cache()
    preds = np.zeros((len(test_dataloaders), 1))
    probs = np.zeros((len(test_dataloaders), self.num_classes))
    for idx, inputs in enumerate(test_dataloaders):
        if isinstance(inputs, list):
          inputs = inputs[0]
        inputs = inputs.to(device)
        outputs = self(inputs) 
        _, current_pred = torch.max(outputs, 1) 
        current_probs = nn.functional.softmax(outputs, dim=1)
        preds[idx] = current_pred.item() # save the predicted class index
        probs[idx] = current_probs.cpu() # save probs of each class
    return preds, probs
  
  def init_weights(self):
      for m in self.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()

  def fit(self, dataloaders, criterion, optimizer, scheduler, num_epochs=25):
    model = self
    model.init_weights()
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
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
            
            # log stats in Tensorboard
            writer.add_scalar('Loss/'+phase, epoch_loss, epoch) # log loss in tensorboard
            writer.add_scalar('Accuracy/'+phase, epoch_acc, epoch) # log acc in tensorboard
            layer_0_weights = self.features[0].weight
            writer.add_scalar('Mean layer 0 weight', layer_0_weights.mean().item(), epoch) # log mean value of first layer weights
            writer.add_histogram('Layer 0 weights', layer_0_weights.flatten(), epoch)    # add a historgram of first layer weights
            writer.flush()

            # save the best model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        
    time_elapsed = time.time() - since
    print(f'Training complete in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    model.load_state_dict(best_model_wts) # return model to the best state


  def summary(self):
    return summary(self, (3, 256, 256))


# In[ ]:


class Net(AbstractModel):
    def __init__(self, cls2idx, batchnorm=False, *args, **kwargs):
        super(Net, self).__init__(cls2idx, *args, **kwargs)
        self.cls2idx = cls2idx
        self.num_classes = len(cls2idx)
        if batchnorm:
          self.features = nn.Sequential(
              nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
              nn.BatchNorm2d(64),
              nn.ReLU(inplace=True),
              nn.MaxPool2d(kernel_size=3, stride=2),
              nn.Conv2d(64, 192, kernel_size=5, padding=2),
              nn.BatchNorm2d(192),
              nn.ReLU(inplace=True),
              nn.MaxPool2d(kernel_size=3, stride=2),
              nn.Conv2d(192, 384, kernel_size=3, padding=1),
              nn.BatchNorm2d(384),
              nn.ReLU(inplace=True),
              nn.Conv2d(384, 256, kernel_size=3, padding=1),
              nn.BatchNorm2d(256),
              nn.ReLU(inplace=True),
              nn.Conv2d(256, 256, kernel_size=3, padding=1),
              nn.BatchNorm2d(256),
              nn.ReLU(inplace=True),
              nn.MaxPool2d(kernel_size=3, stride=2)
          )
        else:
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
              nn.MaxPool2d(kernel_size=3, stride=2)
            )
          
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, self.num_classes)
        )
        self.init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# In[ ]:


# Check for the availability of a GPU, and use CPU otherwise
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device


# In[ ]:


cls2idx = dataloaders['train'].dataset.class_to_idx
net = Net(cls2idx=cls2idx).to(device)
print(net)


# ### Inspect the model graph
# You can print a network object to find useful information about it:

# TensorBoard can help visualize the network graph. It takes practice to read these.  
# 
# Write the graph to TensorBoard and review it.

# In[ ]:


writer.add_graph(net, images.to(device))
writer.flush()


# You can also use the package `torchsummary` for a fuller info on the model:

# In[ ]:


get_ipython().system('pip install torchsummary')


# In[ ]:


channels=3; H=256; W=256
from torchsummary import summary
summary(net, input_size=(channels, H, W))


# ## Train the network
# Next, we'll train the network. In the training loop, log relevant metrics that would allow you to plot in TensorBoard:

# 1. The network loss
# 1. Train and test error
# 1. Average weight in the first layer
# 1. Histogram of weights in the first layer

# In[ ]:


setup_seed()
optimizer_ft = optim.Adam(net.parameters(), lr=0.0001)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)
criterion = nn.CrossEntropyLoss()
num_epochs = 15
net.fit(dataloaders, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs)


# ### Precision-Recall Curve
# Use TensorBoard to plot the precision-recall curve:

# In[ ]:


test_dataloader = torch.utils.data.DataLoader(datasets.ImageFolder(os.path.join(data_dir, "val"), data_transforms["val"]), batch_size=1, shuffle=False, num_workers=2)
y_pred, y_prob = net.predict(test_dataloader)
y_true_idx = np.array(test_dataloader.dataset.targets)


# In[ ]:


y_pred = y_pred.flatten() # convert to 1d vector
# plotting pr curve for each class
for class_index in range(len(class_names)):
    preds = (y_pred == class_index) # boolean vector identifying if the sample was predicted to belong to the current class
    probs = y_prob[:, class_index] # take predicted probability of the current class
    writer.add_pr_curve(class_names[class_index], preds, probs) # log a PR curve for the current class


# ### Display Model Errors
# A valuable practice is to review errors made by the model in the test set. These might reveal cases of bad preprocessing or lead to come up with improvements to your original model.
# 
# Show 12 images of errors made by the model. For each, display the true and predicted classes, and the model confidence in its answer.

# In[ ]:


error_indices = np.argwhere(y_pred.flatten() != y_true_idx) # get indexies of wrong predictions
selected_12_errors = np.random.choice(error_indices.flatten(), size=12, replace=False) # get 12 random errors
error_images = np.array(test_dataloader.dataset.samples) # load paths of all images

for i, err_idx in enumerate(selected_12_errors):
      true_class = y_true_idx[err_idx]
      true_label = class_names[true_class]
      incorrect_class = int(y_pred[err_idx])
      incorrect_class_label = class_names[incorrect_class]
      incorrect_prob = y_prob[err_idx, incorrect_class]
      error_image_filename = error_images[err_idx][0] # take path of the current error image
      error_image = data_transforms["val"](PIL.Image.open(error_image_filename)) # load the image as array and transform it
      error_image = denormalize_image(error_image) # return the image to original color pallette
      title = f'true label: {true_label}, pred label: {incorrect_class_label} ({incorrect_prob:.3f})'
      writer.add_image(title, error_image, 0)
writer.close()


# ## Batch Normalization
# In this section, we'll add a Batch Norm layer to your network.  
# Use TensorBoard to compare the network's convergence (train and validation loss) with and without Batch Normalization.

# In[ ]:


netBN = Net(cls2idx=cls2idx, batchnorm=True).to(device)
print(netBN)


# In[ ]:


setup_seed()
writer = SummaryWriter('logs/BatchNorm') # Create Tensorboard event writer will output to the relevant folder
optimizer_BN = optim.Adam(netBN.parameters(), lr=0.0001)
exp_lr_scheduler_BN = lr_scheduler.StepLR(optimizer_BN, step_size=5, gamma=0.1)
criterion = nn.CrossEntropyLoss()
netBN.fit(dataloaders, criterion, optimizer_BN, exp_lr_scheduler_BN, num_epochs=15)


# We can see that los on both training and validation is much lower, and accuracy on both is much higher: **67%** validation accuracy with BatchNorm and **46%** without BN!
# 
# Go BatchNorm!!! :-) 

# Use TensorBoard to plot the distribution of activations with and without Batch Normalization.

# In[ ]:


def activation_hook_regular(layer, input, out):
  writer.add_histogram('Layer 0 activations without BatchNorm', out.flatten())

def activation_hook_BN(layer, input, out):
  writer.add_histogram('Layer 0 activations with BatchNorm', out.flatten())

net.features[1].register_forward_hook(activation_hook_regular) # register activation hook for logging activations of layer 0 (ReLU output) in a regular net without Batchnorm
netBN.features[2].register_forward_hook(activation_hook_BN) # register activation hook for logging activations of layer 0 (ReLU output) - for comparison with regular model without BatchNorm


# In[ ]:


images = images.to(device) # moves a batch of images to GPU
_ = net(images) # apply the regular model on one batch of images and log the activations
_ = netBN(images) # apply the BatchNorm model on one batch of images and log the activations


# From the histogram we can see that in a model without BatchNorm, most of the the activations are between 0.05-0.16, while in BatchNormn model they are larger (scaled up) between 0.16 and 0.48

# ## Custom Loss Function
# Manually labeled datasets often contain labeling errors. These can have a large effect on the trained model.  
# In this task we’ll work on a highly noisy dataset. Take our cleaned Israeli Politicians dataset and randomly replace 10% of the true labels.
# Compare the performance of the original model to a similar model trained on the noisy labels. 
# 
# Suggest a loss function that might help with noisy labels. Following this guide, implement your own custom loss function in PyTorch and compare the model performance using it:  
# https://discuss.pytorch.org/t/solved-what-is-the-correct-way-to-implement-custom-loss-function/3568/9
# 

# In[ ]:


# Create a dataloader that replaces 10% of the labels in the training set

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

dataloaders["train"] = MixUpWrapper(0.1, dataloaders["train"])


# In[ ]:


# Let's replace 10% of the labels in the training set and see how it affects accuracy without changing the loss function
writer = SummaryWriter('logs/NoisyLabels') # Create Tensorboard event writer will output to the relevant folder
netCL = Net(cls2idx=cls2idx, batchnorm=True).to(device)
setup_seed()
optimizer_CL = optim.Adam(netCL.parameters(), lr=0.0001)
exp_lr_scheduler_CL = lr_scheduler.StepLR(optimizer_CL, step_size=5, gamma=0.1)
criterion = nn.CrossEntropyLoss()
netCL.fit(dataloaders, criterion, optimizer_CL, exp_lr_scheduler_CL, num_epochs=15) 


# We can see that mixing up 10% of the training labels reduced validation accuracy from 67% to 49%

# In[ ]:


# Now, let's create a class with custom loss function (LabelSmoothing), that is supposted to help in cases on noisy labels

import torch.nn.functional as F

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


writer = SummaryWriter('logs/CustomLoss') # Create Tensorboard event writer will output to the relevant folder
netCL = Net(cls2idx=cls2idx, batchnorm=True).to(device)
setup_seed()
optimizer_CL = optim.Adam(netCL.parameters(), lr=0.0001)
exp_lr_scheduler_CL = lr_scheduler.StepLR(optimizer_CL, step_size=5, gamma=0.1)
criterion = LabelSmoothingCrossEntropy() 
netCL.fit(dataloaders, criterion, optimizer_CL, exp_lr_scheduler_CL, num_epochs=15) 

