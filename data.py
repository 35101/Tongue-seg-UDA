import torch
import glob
import torchvision
from torchvision import transforms
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset,DataLoader
from FDA import FDA_source_to_target_np

transforms = transforms.Compose([
    transforms.ToTensor()
])

class FDA_dataset(Dataset):
    def __init__(self,src_path,tar_path=None,transform=None):
        self.src_path = src_path
        self.tar_path = tar_path
        self.srcs_path = glob.glob(os.path.join(src_path, "image/*.png"))
        if tar_path is not None:
            self.tars_path = glob.glob(os.path.join(tar_path, "image/*.png"))
        self.transform = transforms

    def __getitem__(self, index):
        src_path = self.srcs_path[index]
        label_path = src_path.replace("image","label")
        src = np.array(Image.open(src_path).convert("RGB"), dtype=np.float32)
        if self.tar_path is not None:
            tar_path = self.tars_path[index]
            tar = np.array(Image.open(tar_path).convert("RGB"), dtype=np.float32)
        label = np.array(Image.open(label_path).convert("L"), dtype=np.float32)
        label[label > 0] = 1.0
        #print("label:",label.shape)
        if self.tar_path is not None:
            src = src.transpose((2, 0, 1))
            tar = tar.transpose((2, 0, 1))
            src_in_trg = FDA_source_to_target_np(src, tar, L=0.09)
            image = src_in_trg.transpose((1, 2, 0))
        else:
            image = src
        #print("image:",image.shape)
        if self.transform is not None:
            image = transforms(image)
            label = transforms(label)
        image = image.type(torch.FloatTensor)
        return image, label

    def __len__(self):
        return len(self.srcs_path)

#if __name__ == "__main__":
    #tongedataset = FDA_dataset("D:/table/hecheng/train","D:/table/hecheng/test",transform = transforms)
    #print("The number of data:", len(tongedataset))
    #train_loader = DataLoader(dataset=tongedataset,batch_size=4,num_workers=0,pin_memory=True,shuffle=True)
    #for image, label in train_loader:
        #print("image shape:", image.shape)
        #print("label shape:", label.shape)

