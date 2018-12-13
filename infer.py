import argparse

import torch
from PIL import Image
import os

from dataset import Dataset
from model import Model


def _infer(path_to_image: str, path_to_checkpoint: str):
    image = Image.open(path_to_image)
    image = Dataset.preprocess(image)
    prediction = ""
    model = Model().cuda()
    model.load(path_to_checkpoint)
    
    print('Start inferring')

    with torch.no_grad():
        
        images = image.unsqueeze(dim=0).cuda()
        
        # TODO: CODE BEGIN
        # prediction = xxx
        output = model.eval().forward(images)
        length_predictions = output[0].data.max(1)[1].item()
        predictions = [output.data.max(1)[1].item() for output in output]
        print(length_predictions, predictions)
        for _ in predictions:
            prediction += str(_)
        
        #raise NotImplementedError
        # TODO: CODE END

        print(f'Prediction = {prediction:s}')

    filename = path_to_image.split('/')[-1]  # would be xxx.png
    with open(os.path.join("/home/bruce/SVHN-TODO.pytorch/results/", f'{filename}-prediction.txt'), 'w') as fp:
        fp.write(f'{prediction:s}')

    print('Done')


if __name__ == '__main__':
    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument('image', type=str, help='path to image')
        parser.add_argument('-c', '--checkpoint', help='path to checkpoint')
        args = parser.parse_args()

        path_to_image = args.image
        path_to_checkpoint = args.checkpoint

        _infer(path_to_image, path_to_checkpoint)

    main()
