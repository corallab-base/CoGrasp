from model import PruningNetwork
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from dataset_builder import dataset_builder
import sys
import time
import gc
import argparse

def merge_object_gripper_hand_pc(object_pc, gripper_pc, hand_pc):
    object_shape = object_pc.shape
    gripper_shape = gripper_pc.shape
    hand_shape = hand_pc.shape
    batch_size = object_shape[0]

    merged_xyz = torch.cat((object_pc, gripper_pc, hand_pc), 1)
    labels = [torch.zeros((object_shape[1],1), dtype=torch.float), torch.ones((gripper_shape[1],1), dtype=torch.float), -torch.ones((hand_shape[1],1), dtype=torch.float)]
    labels = torch.cat(labels, 0)
    labels = labels.expand(1, labels.shape[0], labels.shape[1])
    labels = torch.tile(labels, [batch_size, 1, 1])

    merged_points = torch.concat([merged_xyz, labels], -1)
    merged_points = torch.permute(merged_points, (0, 2, 1))

    return merged_xyz.contiguous(), merged_points.contiguous()

def save(model, epoch, val_acc, val_loss, optimizer):
    PATH = f'/home/kaykay/isaacgym/python/contact_graspnet/PruningNetwork/checkpoints/checkpoint_modified/epoch_{epoch}.pt'
    torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': val_loss,
    'accuracy': val_acc
    }, PATH)
    print(f'Model saved for epoch: {epoch}')

def load_model(epoch, model, optimizer):
    PATH = f'/home/kaykay/isaacgym/python/contact_graspnet/PruningNetwork/checkpoints/checkpoint_modified/epoch_{epoch}.pt'
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    accuracy = checkpoint['accuracy']
    return model, optimizer, loss, accuracy

def train(train_dataloader, test_dataloader, cur_epoch):
    lr = 1e-4
    max_epoch = 200
    n_batches = len(train_dataloader)

    model = PruningNetwork()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    get_loss = nn.BCELoss()
    best_acc = 0
    best_loss = sys.maxsize

    if cur_epoch >= 0:
        model, optimizer, best_loss, best_acc = load_model(cur_epoch, model, optimizer)
    else:
        cur_epoch = -1

    for epoch in range(cur_epoch+1, max_epoch):
        model.train()
        for idx, (object_pc, hand_pc, gripper_pc, label) in enumerate(train_dataloader):
            batch_start_time = time.time()
            optimizer.zero_grad()
            merged_xyz, merged_points = merge_object_gripper_hand_pc(object_pc, gripper_pc, hand_pc)
            merged_xyz = merged_xyz.cuda()
            merged_points = merged_points.cuda()
            label = label.cuda()
            res = model(merged_xyz, merged_points)
            loss = get_loss(res, label)
            loss_value = loss.item()
            loss.backward()
            optimizer.step()

            if idx % 100 == 0:
                print(f'[TRAINING] [Epoch {epoch}/{max_epoch}][Batch {idx+1}/{n_batches}] Loss: {loss_value} || Batch Time: {time.time()-batch_start_time}')

        if epoch%5 == 0:
            val_loss, val_acc = validate(model, optimizer, epoch, test_dataloader)
            if val_acc > best_acc:
                best_acc = val_acc
                save(model, epoch, val_acc, val_loss, optimizer)

def validate(model, optimizer, epoch, test_dataloader):
    validation_start_time = time.time()
    print(f'[VALIDATION] Start validating epoch {epoch}')
    model.eval()
    get_loss = nn.BCELoss()
    n_samples = len(test_dataloader)
    threshold = 0.95
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for idx, (object_pc, hand_pc, gripper_pc, label) in enumerate(test_dataloader):
            merged_xyz, merged_points = merge_object_gripper_hand_pc(object_pc, gripper_pc, hand_pc)
            merged_xyz = merged_xyz.cuda()
            merged_points = merged_points.cuda()
            label = label.cuda()
            res = model(merged_xyz, merged_points)
            loss = get_loss(res, label)
            total_loss += loss.item()
            accuracy = (torch.abs(label-res) < 0.05).long()
            correct += torch.count_nonzero(accuracy)

    validation_loss = total_loss/n_samples
    validation_accuracy = correct*100/n_samples
    print(f'[VALIDATION] Loss: {validation_loss} Accuracy: {validation_accuracy}% || Validation Time: {time.time() - validation_start_time}')

    return validation_loss, validation_accuracy

def main():
    gc.collect()
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=0)
    FLAGS = parser.parse_args()
    train_dataloader = dataset_builder('train')
    test_dataloader = dataset_builder('test')
    train(train_dataloader, test_dataloader, int(FLAGS.epoch))

device = 'cuda'

if __name__ == '__main__':
    main()
