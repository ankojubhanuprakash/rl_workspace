import torch
import numpy as np
import random
import matplotlib.pyplot as plt
#from torch._C import T
from torch.nn.modules.conv import ConvTranspose2d
import torchvision
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def set_device():
  device = "cuda" if torch.cuda.is_available() else "cpu"
  if device != "cuda":
    print("WARNING: For this notebook to perform best, "
        "if possible, in the menu under `Runtime` -> "
        "`Change runtime type.`  select `GPU` ")
  else:
    print("GPU is enabled in this notebook.")

  return device
  
def set_seed(seed=None, seed_torch=True):
  if seed is None:
    seed = np.random.choice(2 ** 32)
  random.seed(seed)
  np.random.seed(seed)
  if seed_torch:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

  print(f'Random seed {seed} has been set.')


# In case that `DataLoader` is used
def seed_worker(worker_id):
  worker_seed = torch.initial_seed() % 2**32
  np.random.seed(worker_seed)
  random.seed(worker_seed)

SEED = 2021
set_seed(seed=SEED)
DEVICE = set_device()


# dowlonald Data
mnist = MNIST(root='data',train=True,download=True,transform=Compose([ToTensor()]))
batch_size = 100
data_loader = DataLoader(mnist,batch_size=batch_size,shuffle=True)
for img_batch,label_batch in data_loader:
  print('first'  )
  print(img_batch.shape)
  plt.imshow(img_batch[0][0])
  print(label_batch.shape)
  print(label_batch[0])
  break

image_size=28*28
hidden_size = 256


class disc_Network(nn.Module):
  def __init__(self,input_shape):
    super(disc_Network,self).__init__()
    self.conv = nn.Sequential(
                    nn.Conv2d(in_channels=input_shape[0],out_channels=64,stride=2,kernel_size=4,padding=1),
                    nn.Tanh(),  
                    nn.Conv2d(in_channels=64,out_channels=128,stride=2,kernel_size=4,padding=1) ,
                    nn.Tanh(),
                    nn.Conv2d(in_channels=128,out_channels=256,stride=2,kernel_size=4,padding=1) ,
                    nn.Tanh(),
                    nn.Conv2d(in_channels=256,out_channels=512,stride=2,kernel_size=4,padding=1) ,
                    nn.Tanh()
                    ,nn.Flatten())
    conv_out=self.get_conv_out (input_shape)
    self.fc=nn.Sequential(nn.Linear(in_features=conv_out,out_features=1),nn.ReLU())
    #nn.Relu()
  
  def get_conv_out(self,shape):
    return int(np.prod(self.conv(torch.zeros(1,*shape)).size()))

  def forward(self,x):
    conv_o = self.conv(x)
    return self.fc(conv_o)

class gen_Network(nn.Module):
  def __init__(self,latentsize=128):

    super(gen_Network,self).__init__()   
    self.conv =nn.Sequential(
    # in: latent_size x 1 x 1

    nn.ConvTranspose2d(100, 128, kernel_size=1, stride=1, padding=0, bias=False),
    nn.BatchNorm2d(512),
    nn.ReLU(True),
    # out: 512 x 4 x 4

    nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.ReLU(True),
    # out: 256 x 8 x 8

    nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(True),
    # out: 128 x 16 x 16

    nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(True),
    # out: 64 x 32 x 32

    nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=False),
    nn.Tanh()
    # out: 3 x 64 x 64
)

  def forward(self,x):
    return self.conv(x)

    

def to_device(data,device):
  if isinstance(data,(list,tuple)):
    return [to_device(x,device) for x in data]
  return data.to(device,non_blocking=True)  

class DeviceDataLoader():
  def __init__(self,data_loader_obj ,device):
    self.dl = data_loader_obj
    self.device = device
  def __iter__(self):
    #yield a batch of data  
    for b in self.dl: 
      yield to_device(b,self.device)
train_dl = DeviceDataLoader(data_loader,DEVICE)

generator = gen_Network(100)
generator.to(DEVICE)
discriminator=disc_Network([1,28,28])
discriminator.to(DEVICE)
discriminator=disc_Network([1,28,28])
discriminator.to(DEVICE)
def train_discriminator(real_images, opt_d):
    # Clear discriminator gradients
    opt_d.zero_grad()

    # Pass real images through discriminator
    real_preds = discriminator(real_images)
    real_targets = torch.ones(real_images.size(0), 1, device=device)
    real_loss = F.binary_cross_entropy(real_preds, real_targets)
    real_score = torch.mean(real_preds).item()
    
    # Generate fake images
    latent = torch.randn(batch_size, latent_size, 1, 1, device=device)
    fake_images = generator(latent)

    # Pass fake images through discriminator
    fake_targets = torch.zeros(fake_images.size(0), 1, device=device)
    fake_preds = discriminator(fake_images)
    fake_loss = F.binary_cross_entropy(fake_preds, fake_targets)
    fake_score = torch.mean(fake_preds).item()

    # Update discriminator weights
    loss = real_loss + fake_loss
    loss.backward()
    opt_d.step()
    return loss.item(), real_score, fake_score  
def train_generator(opt_g):
    # Clear generator gradients
    opt_g.zero_grad()
    
    # Generate fake images
    latent = torch.randn(batch_size, latent_size, 1, 1, device=device)
    fake_images = generator(latent)
    
    # Try to fool the discriminator
    preds = discriminator(fake_images)
    targets = torch.ones(batch_size, 1, device=device)
    loss = F.binary_cross_entropy(preds, targets)
    
    # Update generator weights
    loss.backward()
    opt_g.step()
    
    return loss.item()

from torchvision.utils import save_image
import os
sample_dir = 'generated'
os.makedirs(sample_dir, exist_ok=True)    
from tqdm.notebook import tqdm
import torch.nn.functional as F
def fit(epochs, lr, start_idx=1):
    torch.cuda.empty_cache()
    
    # Losses & scores
    losses_g = []
    losses_d = []
    real_scores = []
    fake_scores = []
    
    # Create optimizers
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    for epoch in range(epochs):
        for real_images, _ in tqdm(train_dl):
            # Train discriminator
            loss_d, real_score, fake_score = train_discriminator(real_images, opt_d)
            # Train generator
            loss_g = train_generator(opt_g)
            
        # Record losses & scores
        losses_g.append(loss_g)
        losses_d.append(loss_d)
        real_scores.append(real_score)
        fake_scores.append(fake_score)
        
        # Log losses & scores (last batch)
        print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}".format(
            epoch+1, epochs, loss_g, loss_d, real_score, fake_score))
    
        # Save generated images
        save_samples(epoch+start_idx, fixed_latent, show=False)
    
    return losses_g, losses_d, real_scores, fake_scores
device=DEVICE
lr = 0.0002
epochs = 25
history = fit(epochs, lr)
def save_samples(index, latent_tensors, show=True):
    fake_images = generator(latent_tensors)
    fake_fname = 'generated-images-{0:0=4d}.png'.format(index)
    save_image(denorm(fake_images), os.path.join(sample_dir, fake_fname), nrow=8)
    print('Saving', fake_fname)
    if show:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(fake_images.cpu().detach(), nrow=8).permute(1, 2, 0))