from transformers import ViltProcessor, ViltModel
from PIL import Image
import requests

import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from transformers import ViltProcessor, ViltModel
from torch import nn

from dataloader import VEDataset

from torch.cuda.amp import autocast as autocast
from tqdm import tqdm

import torchvision.transforms as transforms
from torch.cuda.amp import GradScaler
CUDA_VISIBLE_DEVICES=1

import time  

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"






processor = ViltProcessor.from_pretrained("./models")





def collate_fn(data):

    text_list = []
    image_list = []
    label_list = []
    
    for item in data:
        text_list.append(item["text"])
        image_list.append(item["image"])
        label_list.append(item["label"])

    image_list = [Image.open(jpgimage) for jpgimage in image_list]

    inputs = processor(image_list, text_list, padding=True, truncation=True, return_tensors="pt")
        
    return {"inputs":inputs, "label":torch.tensor(label_list)}


NUM_CLASSES = 3
LEARNING_RATE = 5e-5


train_ve_dataset = VEDataset()

train_loader = DataLoader(
        train_ve_dataset,
        batch_size=64,
        num_workers=4, 
        collate_fn= collate_fn
    )




test_ve_dataset = VEDataset(dataset_type="test")

test_loader = DataLoader(
        test_ve_dataset,
        batch_size=64,
        num_workers=4,
        collate_fn= collate_fn
    )


class ModalModel(nn.Module):

    def __init__(
            self,
            num_classes=NUM_CLASSES,
            loss_fn=torch.nn.CrossEntropyLoss(),
            dropout_p=0.2
            
        ):
        super(ModalModel, self).__init__()

        self.dropout_p = dropout_p

        self.loss_fn = loss_fn


  
       
        self.backbone = ViltModel.from_pretrained("./models", num_hidden_layers=12)

        self.class_project = torch.nn.Linear(768, 768)
        self.pred = torch.nn.Linear(768, NUM_CLASSES)
            
        

    def forward(self, inputs, label):

        label = label.cuda()

        for key in inputs:
            inputs[key] = inputs[key].cuda()
        


        outputs = self.backbone(**inputs)
        last_hidden_first_token = outputs.last_hidden_state[:, 0, :]

       
        x = self.class_project(last_hidden_first_token)

        x = torch.nn.functional.relu(x)

        x = nn.Dropout(self.dropout_p)(x)

        pred = self.pred(x)

        loss = self.loss_fn(pred, label)
        

        return (pred, loss)


model = ModalModel().cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


scaler = GradScaler()






@torch.no_grad()
def validation():

    pred_labels = []
    labels = []
    count = 0.
    logs = {}

    model.eval()

    for index, batch in enumerate(test_loader):


        label = batch["label"].cuda()
        inputs = batch["inputs"]

        for key in inputs:
            inputs[key] = inputs[key].cuda()

        pred, _ = model(inputs, label)
        count += torch.sum(pred.argmax(-1) == label.cuda()).item()
        pred = pred.argmax(-1).tolist()
        pred_labels.extend(pred)
        labels.extend(label.tolist())

    
    logs["acc"] = count/len(labels)


    if NUM_CLASSES == 3:
        from sklearn import metrics
        from sklearn.metrics import roc_auc_score

        logs["micro"] = metrics.f1_score(labels, pred_labels, average="micro")
        logs["macro"] = metrics.f1_score(labels, pred_labels, average="macro")

    with  open("./e_log.txt", "a+") as f:
        print(logs, file=f)


    print("done val !")

    model.train()

    return logs["acc"]

best = 0.0

def train(num_epoch = 300):
    global best
    for epoch in range(num_epoch):

        losses = []
        

        for index, batch in enumerate(tqdm(train_loader)):

       
            
            inputs, label = batch["inputs"], batch["label"]

            optimizer.zero_grad()

            with autocast():
                _, loss = model(inputs, label)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.cpu().item())
                
            

            if index % 3000 == 0 and index != 0:

                with  open("./e_log.txt", "a+") as f:
                    print("epoch "+str(epoch)+" loss: " + str(np.array(losses).mean()), file=f)
                losses = []
                acc=validation()
                
                if acc>best:
                    best=acc
                    torch.save(model,"./best.pth")
                

        with  open("./e_log.txt", "a+") as f:
            print("epoch "+str(epoch)+" loss: " + str(np.array(losses).mean()), file=f)
        losses = []
        acc=validation()
        
        if acc>best:
            best=acc
            torch.save(model,"./best.pth")
                

train()
