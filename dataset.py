from enum import Enum
from typing import Tuple

import PIL
import torch.utils.data
from PIL import Image
from torch import Tensor

import numpy as np
import h5py 
import os
import os.path
from torchvision import transforms

class Dataset(torch.utils.data.Dataset):

    class Mode(Enum):
        TRAIN = 'train'
        TEST = 'test'

    def __init__(self, path_to_data_dir: str, mode: Mode):
        super().__init__()
        is_train = mode == Dataset.Mode.TRAIN
        
        # TODO: CODE BEGIN
        self._mode = mode
        datasets = Dataset
        self.data = datasets.getData(datasets, path_to_data_dir, is_train)
        #raise NotImplementedError
        # TODO: CODE END

    def __len__(self) -> int:
        # TODO: CODE BEGIN
        return len(self.data[0])
        #raise NotImplementedError
        # TODO: CODE END

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        # TODO: CODE BEGIN
        labels = torch.tensor([self.data[1][0][index], self.data[1][1][index], self.data[1][2][index], self.data[1][3][index], self.data[1][4][index], self.data[1][5][index]], dtype=torch.int64)
        #image = transforms.ToPILImage()(image).convert('RGB')
        #image = image.resize([64,64])
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop([54, 54]),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        image = transform(self.data[0][index])
        return image, labels
        #raise NotImplementedError
        # TODO: CODE END

    @staticmethod
    def preprocess(image: PIL.Image.Image) -> Tensor:
        # TODO: CODE BEGIN
        transform = transforms.Compose([
            transforms.RandomCrop([54, 54]),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        image = transform(image)
        return image
        #raise NotImplementedError
        # TODO: CODE END
    
    def getName(dataset, index):
        names = dataset["digitStruct"]["name"]
        return ''.join([chr(c[0]) for c in dataset[names[index][0]].value])    
    
    def bboxHelper(datasets, dataset, attr):
        if (len(attr) > 1):
            attr = [dataset[attr.value[j].item()].value[0][0] for j in range(len(attr))]
        else:
            attr = [attr.value[0][0]]
            
        return attr
    
    def getBbox(datasets, dataset, index):
        item = dataset[dataset["digitStruct"]["bbox"][index].item()]
        
        return {
            "height": datasets.bboxHelper(datasets, dataset, item["height"]),
            "label": datasets.bboxHelper(datasets, dataset, item["label"]),
            "left": datasets.bboxHelper(datasets, dataset, item["left"]),
            "top": datasets.bboxHelper(datasets, dataset, item["top"]),
            "width": datasets.bboxHelper(datasets, dataset, item["width"]),
        }
    
    def getWholeBox(datasets, dataset, index, im):
        bbox = datasets.getBbox(datasets, dataset, index)

        im_left = min(bbox["left"])
        im_top = min(bbox["top"])
        im_height = max(bbox["top"]) + max(bbox["height"]) - im_top
        im_width = max(bbox["left"]) + max(bbox["width"]) - im_left
        
        im_top = im_top - im_height * 0.05 # a bit higher
        im_left = im_left - im_width * 0.05 # a bit wider
        im_bottom = min(im.size[1], im_top + im_height * 1.05)
        im_right = min(im.size[0], im_left + im_width * 1.05)
        
        return {
            "label": bbox["label"],
            "left": im_left,
            "top": im_top,
            "right": im_right,
            "bottom": im_bottom
        }
    
    def getData(datasets, path_to_data_dir, is_train):
        if is_train:
            path_to_digitStruct_mat = os.path.join(path_to_data_dir, 'digitStruct.mat')
            path_to_images_dir = path_to_data_dir
        else:
            path_to_digitStruct_mat = os.path.join(path_to_data_dir, 'test/digitStruct.mat')
            path_to_images_dir = os.path.join(path_to_data_dir, 'test/')
        
        digitstruct_mat = h5py.File(path_to_digitStruct_mat, "r")
        data_count = digitstruct_mat["digitStruct"]["name"].shape[0]
        #data_count = 1024
        #imgs = np.ndarray(shape=(data_count, 3, 64, 64), dtype='float32')
        imgs = torch.FloatTensor(data_count, 3, 64, 64)
        y = {
            0: np.zeros(data_count),
            1: np.ones(data_count) * 10,
            2: np.ones(data_count) * 10,
            3: np.ones(data_count) * 10,
            4: np.ones(data_count) * 10,
            5: np.ones(data_count) * 10
        }
        
        for i in range(data_count):
            image = Image.open(path_to_images_dir + datasets.getName(digitstruct_mat, i))
            box = datasets.getWholeBox(datasets, digitstruct_mat, i, image)
            if len(box["label"]) > 5:
                continue
            image = image.crop((box["left"], box["top"], box["right"], box["bottom"])).resize([64, 64])
            transform1 = transforms.Compose([
                transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
                ]
            )
            image = transform1(image)
            
            imgs[i,:,:,:] = image
            #image = transforms.ToPILImage()(image).convert('RGB')
            #image = image.resize([64,64])
            #image = image.reshape([3, 54, 54])
            
            #image = np.array(image, dtype='float32')
            #imgs[i,:,:,:] = np.array(image, dtype='float32')
            
            labels = box["label"]
            y[0][i] = len(labels)
            
            for j in range(0, 5):
                if j < len(labels):
                    if labels[j] == 10:
                        y[j+1][i] = 10
                    else:
                        y[j+1][i] = int(labels[j])
                else:
                    y[j+1][i] = 10
            
            if i % 500 == 0:
                print(i, len(y[0]))
        #y = [
        #    np.array(y[0]).reshape(data_count, 1),
        #    np.array(y[1]).reshape(data_count, 1),
        #    np.array(y[2]).reshape(data_count, 1),
        #    np.array(y[3]).reshape(data_count, 1),
        #    np.array(y[4]).reshape(data_count, 1),
        #    np.array(y[5]).reshape(data_count, 1)
        #]
        return imgs, y
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    