#!/usr/bin/env python
# coding: utf-8

# # Image Classification - Tensorboard, Batch Norm and Custom Loss Functions
# In this exercise, you'll continue to work with our neural network for classifying Israeli Politicians.  
# We will use tensorboard to monitor the training process and model performance.  
# 
# For the questions below, please use the network architecture you suggested in Q8 of HW1.  
# This time, we provide you with a clean dataset of Israeli Politicians, that doesn't include multiple politicians in the same image, in the folder `data/israeli_politicians_cleaned.zip`.

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

# In[8]:





# ### Show images using TensorBoard

# In[9]:





# ### Inspect the model graph
# You can print a network object to find useful information about it:

# In[ ]:


print(net)


# TensorBoard can help visualize the network graph. It takes practice to read these.  
# 
# Write the graph to TensorBoard and review it.

# In[10]:





# You can also use the package `torchsummary` for a fuller info on the model:

# In[13]:


get_ipython().system('pip install torchsummary')


# In[ ]:


channels=3; H=32; W=32
from torchsummary import summary
summary(net, input_size=(channels, H, W))


# ## Train the network
# Next, we'll train the network. In the training loop, log relevant metrics that would allow you to plot in TensorBoard:

# 1. The network loss
# 1. Train and test error
# 1. Average weight in the first layer
# 1. Histogram of weights in the first layer

# In[ ]:





# ### Precision-Recall Curve
# Use TensorBoard to plot the precision-recall curve:

# In[ ]:





# ### Display Model Errors
# A valuable practice is to review errors made by the model in the test set. These might reveal cases of bad preprocessing or lead to come up with improvements to your original model.
# 
# Show 12 images of errors made by the model. For each, display the true and predicted classes, and the model confidence in its answer.

# In[ ]:





# ## Batch Normalization
# In this section, we'll add a Batch Norm layer to your network.  
# Use TensorBoard to compare the network's convergence (train and validation loss) with and without Batch Normalization.

# In[18]:





# Use TensorBoard to plot the distribution of activations with and without Batch Normalization.

# In[ ]:





# ## Custom Loss Function
# Manually labeled datasets often contain labeling errors. These can have a large effect on the trained model.  
# In this task we’ll work on a highly noisy dataset. Take our cleaned Israeli Politicians dataset and randomly replace 10% of the true labels.
# Compare the performance of the original model to a similar model trained on the noisy labels. 
# 
# Suggest a loss function that might help with noisy labels. Following this guide, implement your own custom loss function in PyTorch and compare the model performance using it:  
# https://discuss.pytorch.org/t/solved-what-is-the-correct-way-to-implement-custom-loss-function/3568/9
# 

# In[ ]:




