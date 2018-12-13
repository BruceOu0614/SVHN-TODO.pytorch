import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import Dataset
from model import Model

def _eval(path_to_checkpoint: str, path_to_data_dir: str, path_to_results_dir: str):
    os.makedirs(path_to_results_dir, exist_ok=True)

    # TODO: CODE BEGIN
    model = Model().cuda()
    model.load(path_to_checkpoint)
    model2 = Model().cuda()
    model2.load("/home/bruce/SVHN-TODO.pytorch/checkpoints/model-201812031644-100000.pth")
    model3 = Model().cuda()
    model3.load("/home/bruce/SVHN-TODO.pytorch/checkpoints/model-201812031725-120000.pth")
    model4 = Model().cuda()
    model4.load("/home/bruce/SVHN-TODO.pytorch/checkpoints/model-201812031805-140000.pth")
    model5 = Model().cuda()
    model5.load("/home/bruce/SVHN-TODO.pytorch/checkpoints/model-201812031846-160000.pth")
    model6 = Model().cuda()
    model6.load("/home/bruce/SVHN-TODO.pytorch/checkpoints/model-201812031929-180000.pth")
    model7 = Model().cuda()
    model7.load("/home/bruce/SVHN-TODO.pytorch/checkpoints/model-201812031950-190000.pth")
    model8 = Model().cuda()
    model8.load("/home/bruce/SVHN-TODO.pytorch/checkpoints/model-201812032012-200000.pth")
    model9 = Model().cuda()
    model9.load("/home/bruce/SVHN-TODO.pytorch/checkpoints/model-201812032032-210000.pth")
    model10 = Model().cuda()
    model10.load(path_to_checkpoint)
    model11 = Model().cuda()
    model11.load(path_to_checkpoint)
    
    dataset = Dataset(path_to_data_dir, Dataset.Mode.TEST)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    num_correct = 0
    #raise NotImplementedError
    # TODO: CODE END

    print('Start evaluating')

    with torch.no_grad():
        # TODO: CODE BEGIN
        model.eval()
        model2.eval()
        model3.eval()
        model4.eval()
        model5.eval()
        model6.eval()
        model7.eval()
        model8.eval()
        model9.eval()
        model10.eval()
        model11.eval()
        for batch_index, (images, labels) in enumerate(tqdm(dataloader)):
            images = images.cuda()
            labels = labels.cuda()
            
            output = model.eval()(images)
            length_predictions = output[0].data.max(1)[1]
            predictions = [output.data.max(1)[1] for output in output]
            labels = torch.chunk(labels, 6, dim=1)
            length = torch.squeeze(labels[0])
            dig1 = torch.squeeze(labels[1])
            dig2 = torch.squeeze(labels[2])
            dig3 = torch.squeeze(labels[3])
            dig4 = torch.squeeze(labels[4])
            dig5 = torch.squeeze(labels[5])
            num_correct += (predictions[1].eq(dig1.data) &
                                predictions[2].eq(dig2.data) &
                                predictions[3].eq(dig3.data) &
                                predictions[4].eq(dig4.data) &
                                predictions[5].eq(dig5.data)).cpu().sum()
        #raise NotImplementedError
        # TODO: CODE END

        accuracy = num_correct.item() / len(dataset)
        print(f'Accuracy = {accuracy:.4f}')
        num_correct = 0
        
        for batch_index, (images, labels) in enumerate(tqdm(dataloader)):
            images = images.cuda()
            labels = labels.cuda()
            
            output = model2(images)
            length_predictions = output[0].data.max(1)[1]
            predictions = [output.data.max(1)[1] for output in output]
            labels = torch.chunk(labels, 6, dim=1)
            length = torch.squeeze(labels[0])
            dig1 = torch.squeeze(labels[1])
            dig2 = torch.squeeze(labels[2])
            dig3 = torch.squeeze(labels[3])
            dig4 = torch.squeeze(labels[4])
            dig5 = torch.squeeze(labels[5])
            num_correct += (predictions[1].eq(dig1.data) &
                                predictions[2].eq(dig2.data) &
                                predictions[3].eq(dig3.data) &
                                predictions[4].eq(dig4.data) &
                                predictions[5].eq(dig5.data)).cpu().sum()
        #raise NotImplementedError
        # TODO: CODE END

        accuracy = num_correct.item() / len(dataset)
        print(f'Accuracy = {accuracy:.4f}')
        num_correct = 0
        
        for batch_index, (images, labels) in enumerate(tqdm(dataloader)):
            images = images.cuda()
            labels = labels.cuda()
            
            output = model3(images)
            length_predictions = output[0].data.max(1)[1]
            predictions = [output.data.max(1)[1] for output in output]
            #print(predictions)
            labels = torch.chunk(labels, 6, dim=1)
            length = torch.squeeze(labels[0])
            dig1 = torch.squeeze(labels[1])
            dig2 = torch.squeeze(labels[2])
            dig3 = torch.squeeze(labels[3])
            dig4 = torch.squeeze(labels[4])
            dig5 = torch.squeeze(labels[5])
            num_correct += (predictions[1].eq(dig1.data) &
                                predictions[2].eq(dig2.data) &
                                predictions[3].eq(dig3.data) &
                                predictions[4].eq(dig4.data) &
                                predictions[5].eq(dig5.data)).cpu().sum()
        #raise NotImplementedError
        # TODO: CODE END

        accuracy = num_correct.item() / len(dataset)
        print(f'Accuracy = {accuracy:.4f}')
        num_correct = 0
        
        for batch_index, (images, labels) in enumerate(tqdm(dataloader)):
            images = images.cuda()
            labels = labels.cuda()
            
            output = model4(images)
            length_predictions = output[0].data.max(1)[1]
            predictions = [output.data.max(1)[1] for output in output]
            labels = torch.chunk(labels, 6, dim=1)
            length = torch.squeeze(labels[0])
            dig1 = torch.squeeze(labels[1])
            dig2 = torch.squeeze(labels[2])
            dig3 = torch.squeeze(labels[3])
            dig4 = torch.squeeze(labels[4])
            dig5 = torch.squeeze(labels[5])
            num_correct += (predictions[1].eq(dig1.data) &
                                predictions[2].eq(dig2.data) &
                                predictions[3].eq(dig3.data) &
                                predictions[4].eq(dig4.data) &
                                predictions[5].eq(dig5.data)).cpu().sum()
        #raise NotImplementedError
        # TODO: CODE END

        accuracy = num_correct.item() / len(dataset)
        print(f'Accuracy = {accuracy:.4f}')
        num_correct = 0
        
        for batch_index, (images, labels) in enumerate(tqdm(dataloader)):
            images = images.cuda()
            labels = labels.cuda()
            
            output = model5(images)
            length_predictions = output[0].data.max(1)[1]
            predictions = [output.data.max(1)[1] for output in output]
            labels = torch.chunk(labels, 6, dim=1)
            length = torch.squeeze(labels[0])
            dig1 = torch.squeeze(labels[1])
            dig2 = torch.squeeze(labels[2])
            dig3 = torch.squeeze(labels[3])
            dig4 = torch.squeeze(labels[4])
            dig5 = torch.squeeze(labels[5])
            num_correct += (predictions[1].eq(dig1.data) &
                                predictions[2].eq(dig2.data) &
                                predictions[3].eq(dig3.data) &
                                predictions[4].eq(dig4.data) &
                                predictions[5].eq(dig5.data)).cpu().sum()
        #raise NotImplementedError
        # TODO: CODE END

        accuracy = num_correct.item() / len(dataset)
        print(f'Accuracy = {accuracy:.4f}')
        num_correct = 0
        
        for batch_index, (images, labels) in enumerate(tqdm(dataloader)):
            images = images.cuda()
            labels = labels.cuda()
            
            output = model6(images)
            length_predictions = output[0].data.max(1)[1]
            predictions = [output.data.max(1)[1] for output in output]
            labels = torch.chunk(labels, 6, dim=1)
            length = torch.squeeze(labels[0])
            dig1 = torch.squeeze(labels[1])
            dig2 = torch.squeeze(labels[2])
            dig3 = torch.squeeze(labels[3])
            dig4 = torch.squeeze(labels[4])
            dig5 = torch.squeeze(labels[5])
            num_correct += (predictions[1].eq(dig1.data) &
                                predictions[2].eq(dig2.data) &
                                predictions[3].eq(dig3.data) &
                                predictions[4].eq(dig4.data) &
                                predictions[5].eq(dig5.data)).cpu().sum()
        #raise NotImplementedError
        # TODO: CODE END

        accuracy = num_correct.item() / len(dataset)
        print(f'Accuracy = {accuracy:.4f}')
        num_correct = 0

        for batch_index, (images, labels) in enumerate(tqdm(dataloader)):
            images = images.cuda()
            labels = labels.cuda()
            
            output = model7(images)
            length_predictions = output[0].data.max(1)[1]
            predictions = [output.data.max(1)[1] for output in output]
            labels = torch.chunk(labels, 6, dim=1)
            length = torch.squeeze(labels[0])
            dig1 = torch.squeeze(labels[1])
            dig2 = torch.squeeze(labels[2])
            dig3 = torch.squeeze(labels[3])
            dig4 = torch.squeeze(labels[4])
            dig5 = torch.squeeze(labels[5])
            num_correct += (predictions[1].eq(dig1.data) &
                                predictions[2].eq(dig2.data) &
                                predictions[3].eq(dig3.data) &
                                predictions[4].eq(dig4.data) &
                                predictions[5].eq(dig5.data)).cpu().sum()
        #raise NotImplementedError
        # TODO: CODE END

        accuracy = num_correct.item() / len(dataset)
        print(f'Accuracy = {accuracy:.4f}')
        num_correct = 0
        
        for batch_index, (images, labels) in enumerate(tqdm(dataloader)):
            images = images.cuda()
            labels = labels.cuda()
            
            output = model8(images)
            length_predictions = output[0].data.max(1)[1]
            predictions = [output.data.max(1)[1] for output in output]
            labels = torch.chunk(labels, 6, dim=1)
            length = torch.squeeze(labels[0])
            dig1 = torch.squeeze(labels[1])
            dig2 = torch.squeeze(labels[2])
            dig3 = torch.squeeze(labels[3])
            dig4 = torch.squeeze(labels[4])
            dig5 = torch.squeeze(labels[5])
            num_correct += (predictions[1].eq(dig1.data) &
                                predictions[2].eq(dig2.data) &
                                predictions[3].eq(dig3.data) &
                                predictions[4].eq(dig4.data) &
                                predictions[5].eq(dig5.data)).cpu().sum()
        #raise NotImplementedError
        # TODO: CODE END

        accuracy = num_correct.item() / len(dataset)
        print(f'Accuracy = {accuracy:.4f}')
        num_correct = 0
        
        for batch_index, (images, labels) in enumerate(tqdm(dataloader)):
            images = images.cuda()
            labels = labels.cuda()
            
            output = model9(images)
            length_predictions = output[0].data.max(1)[1]
            predictions = [output.data.max(1)[1] for output in output]
            labels = torch.chunk(labels, 6, dim=1)
            length = torch.squeeze(labels[0])
            dig1 = torch.squeeze(labels[1])
            dig2 = torch.squeeze(labels[2])
            dig3 = torch.squeeze(labels[3])
            dig4 = torch.squeeze(labels[4])
            dig5 = torch.squeeze(labels[5])
            num_correct += (predictions[1].eq(dig1.data) &
                                predictions[2].eq(dig2.data) &
                                predictions[3].eq(dig3.data) &
                                predictions[4].eq(dig4.data) &
                                predictions[5].eq(dig5.data)).cpu().sum()
        #raise NotImplementedError
        # TODO: CODE END

        accuracy = num_correct.item() / len(dataset)
        print(f'Accuracy = {accuracy:.4f}')
        num_correct = 0

    with open(os.path.join(path_to_results_dir, 'accuracy.txt'), 'w') as fp:
        fp.write(f'{accuracy:.4f}')

    print('Done')


if __name__ == '__main__':
    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument('checkpoint', type=str, help='path to evaluate checkpoint, e.g.: ./checkpoints/model-100.pth')
        parser.add_argument('-d', '--data_dir', default='./data', help='path to data directory')
        parser.add_argument('-r', '--results_dir', default='./results', help='path to results directory')
        args = parser.parse_args()

        path_to_checkpoint = args.checkpoint
        path_to_data_dir = args.data_dir
        path_to_results_dir = args.results_dir

        _eval(path_to_checkpoint, path_to_data_dir, path_to_results_dir)

    main()
