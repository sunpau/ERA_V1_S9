import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import GetCorrectPredCount
from tqdm.notebook import tqdm

    
#-----------------------S8 Assignment----------------
class Block(nn.Module):
    """
    This class defines a convolution layer followed by
    normalization and activation function. Relu is used as activation function.
    """
    def __init__(self, input_size, output_size, dropout = 0.0):
        """Initialize Block

        Args:
            input_size (int): Input Channel Size
            output_size (int): Output Channel Size
            padding (int, optional): Padding to be used for convolution layer. Defaults to 1.
            norm (str, optional): Type of normalization to be used. Allowed values ['bn', 'gn', 'ln']. Defaults to 'bn'.
        """
        super(Block, self).__init__()
        
        # self.norm = nn.BatchNorm2d(output_size)
        # self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(dropout)

        self.conv1 = nn.Sequential(nn.Conv2d(input_size, output_size, 3, padding=1, bias=False),
                     nn.ReLU(),
                     nn.BatchNorm2d(output_size),
                     nn.Dropout(dropout))
        self.conv2 = nn.Sequential(nn.Conv2d(output_size, output_size, 3, padding=1, bias=False),
                     nn.ReLU(),
                     nn.BatchNorm2d(output_size),
                     nn.Dropout(dropout))
        #3rd layer
        self.conv3 = nn.Sequential(nn.Conv2d(output_size, output_size, 2, padding=1, stride=2, dilation=2, bias=False),
                     nn.ReLU(),
                     nn.BatchNorm2d(output_size),
                     nn.Dropout(dropout))
        



    def __call__(self, x):
        """
        Args:
            x (tensor): Input tensor to this block
        Returns:
            tensor: Return processed tensor
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        return x
    
class DepthwiseSeparable(nn.Module):
    """
    This class implements depwise separable convolution.
    
    """
    def __init__(self, input_size, output_size):
        """Initialize Block

        Args:
            input_size (int): Input Channel Size
            output_size (int): Output Channel Size
        """
        super(DepthwiseSeparable, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(input_size, input_size, 3,  groups=input_size, bias=False),
                     nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(input_size, output_size, 1, bias=False),
                     nn.ReLU(),
                     nn.BatchNorm2d(output_size))
        

    def __call__(self, x):
        """
        Args:
            x (tensor): Input tensor to this block

        Returns:
            tensor: Return processed tensor
        """
        x = self.conv1(x)
        x = self.conv2(x)


        return x
    
class Net(nn.Module):
    """ Network Class

    Args:
        nn (nn.Module): Instance of pytorch Module
    """

    def __init__(self, drop = 0.0):
        """Initialize Network
        """
        super(Net, self).__init__()
        self.drop = drop
        # Conv
        self.block1 = Block(3, 32, dropout=self.drop)
        self.block2 = Block(32, 64, dropout=self.drop)
        #the 3rd block has 3 layers and all implemented as Depthwise-separable
        self.dropout = nn.Dropout(self.drop)
        self.block3_l1 = DepthwiseSeparable(64,128)
        self.block3_l2 = DepthwiseSeparable(128,128)
        self.block3_l3 = DepthwiseSeparable(128,128)
        
        #output layer
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flat = nn.Conv2d(128, 10, 1)

    def forward(self, x):

        # Conv Layer
        x = self.block1(x)
        x = self.block2(x)

        x = self.block3_l1(x)
        x = self.dropout(x)
        x = self.block3_l2(x)
        x = self.dropout(x)
        x = self.block3_l3(x)
        
        # Output Layer
        x = self.gap(x)
        x = self.flat(x)
        x = x.view(-1, 10)

        # Output Layer
        return F.log_softmax(x, dim=1)
        


def train_model(model, device, train_loader, optimizer,criterion):
  model.train()
#   pbar = tqdm(train_loader)

  train_loss = 0
  correct = 0
  processed = 0

  for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()  # zero the gradients- not to use perious gradients

    # Predict
    pred = model(data)

    # Calculate loss
    loss = criterion(pred, target)
    train_loss+=loss.item()

    # Backpropagation
    loss.backward()
    optimizer.step()   #updates the parameter - gradient descent
    
    correct += GetCorrectPredCount(pred, target)
    processed += len(data)

    # pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
  train_acc = 100*correct/processed
  train_loss = train_loss/len(train_loader)
  return train_acc, train_loss
  

def test_model(model, device, test_loader, criterion):
    model.eval() #set model in test (inference) mode

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss

            correct += GetCorrectPredCount(output, target)


    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct / len(test_loader.dataset)
    

    # print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))
    return test_acc, test_loss


