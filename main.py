import rasterio
from torch.utils.data import Dataset,DataLoader
import numpy as np
import os
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from PIL import Image
from unet import UNet
from torch.cuda import amp
import torchvision
from torch.utils.tensorboard import SummaryWriter
writer =SummaryWriter("runs/greenCover")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained U-Net with ResNet encoder
model = smp.Unet(
    encoder_name="resnet34",        # Choose encoder (e.g., resnet34, efficientnet-b0, etc.)
    encoder_weights="imagenet",    # Use pre-trained weights on ImageNet
    in_channels=3,                 # Number of input channels (e.g., 3 for RGB images)
    classes=1                      # Number of output classes
).to(device)

# model = UNet().to(device)

# print(model(image).shape)
# print(model.parameters())



# Dataloading



class DataGenerator(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        
        try:
            # print("Image:",img_path)
            img = rasterio.open(img_path).read()
            img = np.array(img,dtype=np.float32)
            img = img[0:3,:,:]
            # print("mask:",mask_path)
            mask = rasterio.open(mask_path).read()
            mask = np.array(mask, dtype=np.float32)
            mask = mask
            
            return torch.tensor(img), torch.tensor(mask)
        except Exception as e:
            # print(f"Skipping corrupted image/mask")
            return self.__getitem__(index+1)
    

dataset = DataGenerator("/home/sijan/Downloads/AMAZON/Training/image","/home/sijan/Downloads/AMAZON/Training/label")
dataset_validation = DataGenerator("/home/sijan/Downloads/AMAZON/Validation/images", "/home/sijan/Downloads/AMAZON/Validation/masks")

train = DataLoader(dataset, batch_size=3, shuffle=True,pin_memory=True, num_workers=3)
val = DataLoader(dataset_validation, batch_size=5, shuffle=True)


def iou_score(pred, target, epsilon=1e-6):

    # Ensure the inputs are floats and binary
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()

    intersection = torch.sum(pred * target)
    union = torch.sum(pred) + torch.sum(target) - intersection
    return intersection / (union + epsilon)



loss_func = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

@torch.no_grad()
def validation(batch):
    model.eval()
    images, labels = batch
    im_grid = torchvision.utils.make_grid(images, nrow=3)
    im_grid = (im_grid - im_grid.min()) / (im_grid.max() - im_grid.min())
    writer.add_image("greenCover_images",im_grid)
    images = images.to(device)
    labels = labels.to(device)
    out = model(images)
    ms_grid = torchvision.utils.make_grid(out, nrow=3)
    # ms_grid = (ms_grid - ms_grid.min()) / (image.max() - image.min())
    writer.add_image("greenCover_masks",ms_grid)
    writer.close()
    loss = loss_func(out, labels)
    acc = iou_score(out,labels)
    return loss, acc
    
num_epochs = 2
n_total_steps = len(train)
scaler = amp.GradScaler()

history = []
total_loss = []
total_acc = []
total_val_loss = []
total_val_acc = []
running_loss = 0.0
running_acc = 0.0

for epoch in range(num_epochs):
    train_loss = []
    train_acc = []
    for i, (images, masks) in enumerate(train):
        images = images.to(device)
        masks = masks.to(device)
        # print(images.shape)

        with amp.autocast():
            out = model(images)
            loss = loss_func(out, masks)
            if torch.isnan(loss).any():
                print("------------------------")
                print("Nan found, Skipping")
                print("------------------------")
                continue
            acc = iou_score(out, masks)

        train_loss.append(loss)
        train_acc.append(acc)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        running_acc += acc

        if (i+1) % 10 == 0:
            print(f"Epoch: {epoch+1}, step [{i+1}/{n_total_steps}] loss:{loss.item()}, acc:{acc}")
            writer.add_scalar("Training loss", running_loss/10, epoch*n_total_steps+i)
            writer.add_scalar("Training Accuracy", running_acc/10, epoch*n_total_steps+i)
            running_loss = 0.0
            running_acc = 0.0

    # print("List:loss:",train_loss)   
    e_loss = torch.stack(train_loss).mean().item()
    total_loss.append(e_loss)
    print("EPOCH_LOSS:",e_loss)
    e_acc = torch.stack(train_acc).mean().item()
    total_acc.append(e_acc)
    print("EPOCH_ACC:",e_acc)

    # Validation
    print("Starting validation")
    val_loss = []
    val_acc = []
    val_len = len(val)
    for i,batch in enumerate(val):
        print(f"Val Step [{i}/{val_len}]")
        loss, acc = validation(batch)
        val_loss.append(loss)
        val_acc.append(acc)

    # print("Validation loss list:",val_loss)
    e_val_loss = torch.stack(val_loss).mean().item()
    e_val_acc = torch.stack(val_acc).mean().item()
    total_val_loss.append(e_val_loss)
    total_val_acc.append(e_val_acc)

    writer.add_scalar("Val loss", e_val_loss, epoch*n_total_steps)
    writer.add_scalar("Val Accuracy", e_val_acc, epoch*n_total_steps)

# print(total_val_acc, total_acc)

torch.save(obj=model.state_dict(),
           f="models/greenCover_detection.pth")

with torch.no_grad():
    test = rasterio.open("test.tif").read()[0:3,:,:]
    test = np.expand_dims(test, 0)
    test = torch.from_numpy(test.astype(np.float32)).to(device)

    writer.add_graph(model, test)
    writer.close()

    test_img = model(test)
    img = test_img.to(torch.device("cpu"))
    # plt.imshow(img)
    # plt.show()
    img = torch.squeeze(img)
    img = np.array(img, dtype=np.uint8)
    img_s = Image.fromarray(img*255)
    img_s.save("test_mask.png")


