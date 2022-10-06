import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset,DataLoader


tests = 5                                    #number of validation samples per epoch

short = np.loadtxt('short.dat')              #file with x,y,z coodinates of event

x = np.array(np.ravel(short)[7::10])        
y = np.array(np.ravel(short)[8::10])
z = np.array(np.ravel(short)[9::10])

x = x/1000                                  #normalization
y = y/1000 
z = z/1000 

test_Y = np.array([x,y,z])
test_Y = np.reshape(test_Y.T,(1958,3))
Y = test_Y[:-tests]
Y = torch.tensor(Y)               #Y is the x,y,z coordinates (ie, the answer)
                                           
test_Y = torch.tensor(test_Y[1958-tests:])


test_X = np.loadtxt('long.dat')          #file with 255 detectors' firing strength                           
test_X = np.reshape(test_X,(1958,256))
X = test_X[:-tests]
X = torch.tensor(X)                     #X is the output from the DEAP detectors, containing firing strength of each of the 255 detectors per event

test_X = torch.tensor(test_X[1958-tests:])    #testX and testY are validation data


train_set = (X,Y)                       #data
val_set = (test_X,test_Y)     

class SetOfData(Dataset):               #dataset class
    
    def __init__(self, dSet):
        super().__init__()
        self.X = dSet[0]
        self.Y = dSet[1]
#        print('a')

    def __len__(self):
#        print('b')
        return len(self.X)
    
    def __getitem__(self, idx):
#        print('c')
        return (X[idx],Y[idx])


class LitDataMod(pl.LightningDataModule):   #datamodule class
    
    def __init__(self, tSet, vSet):
#        print('d')
        super().__init__()
        self.tSet = tSet
        self.vSet = vSet
        
    def setup(self, stage=None):          
        self.train_data = self.tSet
        self.val_data = self.vSet
#        print('e')
        
    def train_dataloader(self):
#        print('f')
        return DataLoader(self.train_data)
    
    def val_dataloader(self):               #unused for now
        return DataLoader(self.val_data)
        

class LitModel(pl.LightningModule):     #lightningmodule, containing model
    
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(256, 400), nn.ReLU(), nn.Linear(400, 3))           #!!experiment with layers, activation function, and neurons per layer!!
        self.model = self.model.double()
#        print('g')
        
    def forward(self, x):
        l = self.model(x)
#        print('h')
        return l
    
    def training_step(self, dSet):          #!!batches to be added!!                  
        X,Y = dSet
        l = self.model(X)
        cost = nn.MSELoss()(l, Y)
        loss = cost.mean()
        self.log('train_loss', loss)
#        print('i')
        return loss
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)    #!!learning rate to be adjusted!!
#        print('j')
        return optimizer
    
'''
    def validation_step(self, dSet, batch_idx, dataset_idx):
        X,Y = dSet                                              #self note: error is here???
        l = self.model(X)
        cost = nn.MSELoss()(l, Y)
        v_loss = cost.mean()
        self.log('val_loss', v_loss)
        
'''

#print('k')
trainload = SetOfData(train_set)
valload = SetOfData(val_set)              

dmodule = LitDataMod(trainload, valload)       #valload (validation dataset) unused, since no validation step

#print('l')
model = LitModel()
trainer = pl.Trainer(accelerator = "gpu", max_epochs=5)          #self note: accelerator = "gpu"
#print('m')
trainer.fit(model, dmodule)  

#print('n')                              #print statements used for debugging