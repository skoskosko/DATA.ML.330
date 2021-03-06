import numpy as np
import torch
import collections
import torch.utils.data
from PIL import Image
import pandas as pd
import os
import torchvision 
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import time
import torch.distributed as dist
import math
import datetime
import engine
import transforms
import utils
# from engine import train_one_epoch, evaluate


# https://towardsdatascience.com/building-your-own-object-detector-pytorch-vs-tensorflow-and-how-to-even-get-started-1d314691d4ae
# https://github.com/pytorch/vision/blob/master/references/detection/utils.py

def parse_one_annot(path_to_data_file, filename):
   data = pd.read_csv(path_to_data_file)
   boxes_array = data[data["filename"] == filename][["xmin", "ymin", "xmax", "ymax"]].values
   
   return boxes_array


class BranchDataset(torch.utils.data.Dataset):
    def __init__(self, root, data_file, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = sorted(os.listdir(os.path.join(root, "images")))
        self.path_to_data_file = data_file
        
    def __getitem__(self, idx):
        # load images and bounding boxes
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        box_list = parse_one_annot(self.path_to_data_file, 
        self.imgs[idx])
        boxes = torch.as_tensor(box_list, dtype=torch.float32)
        
        num_objs = len(box_list)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:,0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        
        if self.transforms is not None:
            img = self.transforms(img)
            # target = self.transforms(target)
        return img, target
        
    def __len__(self):
        return len(self.imgs)

# dataset = RaccoonDataset(root= "./raccoon_dataset", data_file= "./raccoon_dataset/data/raccoon_labels.csv")

# print( dataset.__getitem__(0) )


def get_model(num_classes):
   # load an object detection model pre-trained on COCO
   model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get the number of input features for the classifier
   in_features = model.roi_heads.box_predictor.cls_score.in_features
   # replace the pre-trained head with a new on
   model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
   
   return model

def get_transform(train):
   transforms = []
   # converts the image, a PIL image, into a PyTorch Tensor
   transforms.append(torchvision.transforms.ToTensor())
   if train:
      # during training, randomly flip the training images
      # and ground-truth for data augmentation
      transforms.append(torchvision.transforms.RandomHorizontalFlip(0.5))
   return torchvision.transforms.Compose(transforms)


# use our dataset and defined transformations
# 
dataset = BranchDataset(root= "./Dry", data_file= "./Dry/labels.csv", transforms = get_transform(train=True))
dataset_test = BranchDataset(root= "./Dry", data_file= "./Dry/labels.csv", transforms = get_transform(train=False))

# split the dataset in train and test set

torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()
slice_size = int(len(indices)*0.1) # size of test_dataset
dataset = torch.utils.data.Subset(dataset, indices[:-slice_size])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-slice_size:])

# define training and validation data loaders
def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=utils.collate_fn)
data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)

print("We have: {} examples, {} are training and {} testing".format(len(indices), len(dataset), len(dataset_test)))

torch.cuda.is_available()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')# our dataset has two classes only - raccoon and not racoon

num_classes = 2
# get the model using our helper function
model = get_model(num_classes)
# move model to the right device
model.to(device)# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler which decreases the learning rate by 
# # 10x every 3 epochs

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

num_epochs = 10
for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    engine.train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)# update the learning rate

    print("Test")

    lr_scheduler.step()

    print("Test2")
    # evaluate on the test dataset
    # engine.evaluate(model, data_loader_test, device=device)
    print("Test3")

try:
    os.mkdir("./slice_model/")
except:
    pass

torch.save(model.state_dict(), "./slice_model/slice.pth")