import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision
import os
from PIL import Image, ImageDraw
import pandas as pd
import numpy as np

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

def parse_one_annot(path_to_data_file, filename):
   data = pd.read_csv(path_to_data_file)
   boxes_array = data[data["filename"] == filename][["xmin", "ymin", "xmax", "ymax"]].values
   
   return boxes_array

class RaccoonDataset(torch.utils.data.Dataset):
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


loaded_model = get_model(num_classes = 2)

loaded_model.load_state_dict(torch.load("./slice_model/slice.pth"))


dataset_test = RaccoonDataset(root= "./Dry", data_file= "./Dry/labels.csv", transforms = get_transform(train=False))


def test(idx):
    img, _ = dataset_test[idx]
    label_boxes = np.array(dataset_test[idx][1]["boxes"])

    #put the model in evaluation mode
    loaded_model.eval()

    with torch.no_grad():
        prediction = loaded_model([img])
    
    image = Image.fromarray(img.mul(255).permute(1, 2,0).byte().numpy())
    draw = ImageDraw.Draw(image)

    # draw groundtruth

    for elem in range(len(label_boxes)):
        draw.rectangle([(label_boxes[elem][0], label_boxes[elem][1]), (label_boxes[elem][2], label_boxes[elem][3])], outline ="green", width =3)
    
    for element in range(len(prediction[0]["boxes"])):
        boxes = prediction[0]["boxes"][element].cpu().numpy()
        score = np.round(prediction[0]["scores"][element].cpu().numpy(),
                            decimals= 4)
        if score > 0.8:
            draw.rectangle([(boxes[0], boxes[1]), (boxes[2], boxes[3])], 
            outline ="red", width =3)
            draw.text((boxes[0], boxes[1]), text = str(score))

    image.save("./output/slices/test_slice_%s.png" % str(idx))

for i in range(10):
    test(i)
