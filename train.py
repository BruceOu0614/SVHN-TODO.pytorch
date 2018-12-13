import argparse
import os
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import Dataset
from model import Model
from collections import deque

def _train(path_to_data_dir: str, path_to_checkpoints_dir: str):
    os.makedirs(path_to_checkpoints_dir, exist_ok=True)

    # TODO: CODE BEGIN
    path_to_data_train_dir = os.path.join(path_to_data_dir, 'train/')
    path_to_data_extra_dir = os.path.join(path_to_data_dir, 'extra/')
    path_to_data_test_dir = os.path.join(path_to_data_dir, 'test/')
    
    dataset_train = Dataset(path_to_data_train_dir, mode=Dataset.Mode.TRAIN)
    dataset_extra = Dataset(path_to_data_extra_dir, mode=Dataset.Mode.TRAIN)
    dataset_test = Dataset(path_to_data_test_dir, mode=Dataset.Mode.TRAIN)
    
    dataset = dataset_train+dataset_extra
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=64, shuffle=True)
    
    model = Model().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    
    num_steps_to_display = 200
    num_steps_to_snapshot = 1000
    num_steps_to_finish = 30000000
    
    num_correct = 0
    best_accuracy = 0
    
    step = 0
    time_checkpoint = time.time()
    losses = deque(maxlen=100)
    should_stop = False
    #raise NotImplementedError
    # TODO: CODE END

    print('Start training')

    while not should_stop:
        # TODO: CODE BEGIN
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.cuda()
            labels = labels.cuda()
            output = model.train()(images)
            loss = model.loss(output, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            step += 1
            
            if step % num_steps_to_display == 0:
                elapsed_time = time.time() - time_checkpoint
                time_checkpoint = time.time()
                steps_per_sec = num_steps_to_display / elapsed_time
                avg_loss = sum(losses) / len(losses)
                print(f'[Step {step}] Avg. Loss = {avg_loss:.6f} ({steps_per_sec:.2f} steps/sec)')

            if step % num_steps_to_snapshot == 0:
                path_to_checkpoint = model.save(path_to_checkpoints_dir, step)
                print(f'Model saved to {path_to_checkpoint}')
                
                model.eval()
                for batch_index, (images, labels) in enumerate(tqdm(dataloader_test)):
                    images = images.cuda()
                    labels = labels.cuda()
                    output = model(images)
                    
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
                accuracy = num_correct.item() / len(dataset)
                print(f'Accuracy = {accuracy:.4f}')
                num_correct = 0

            if step == num_steps_to_finish:
                should_stop = True
                break

        #raise NotImplementedError
        # TODO: CODE END

    print('Done')


if __name__ == '__main__':
    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument('-d', '--data_dir', default='./data', help='path to data directory')
        parser.add_argument('-c', '--checkpoints_dir', default='./checkpoints', help='path to checkpoints directory')
        args = parser.parse_args()

        path_to_data_dir = args.data_dir
        path_to_checkpoints_dir = args.checkpoints_dir

        _train(path_to_data_dir, path_to_checkpoints_dir)

    main()
