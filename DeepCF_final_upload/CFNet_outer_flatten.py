import argparse
import os
import time

import numpy as np

import torch
import torch.nn as nn
from Dataset import Dataset
from DMF_conv import DMF
from evaluate import evaluate_model
from MLP_conv import MLP
from utils import (AverageMeter, BatchDataset, get_optimizer,
                   get_train_instances, get_train_matrix)


def parse_args():
    parser = argparse.ArgumentParser(description="Run DeepCF.")
    parser.add_argument("--path", nargs="?", default="/content/drive/MyDrive/DeepCF/Data/",
                        help="Input data path.")
    parser.add_argument("--dataset", nargs="?", default="ml-1m",
                        help="Choose a dataset.")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of epochs.")
    parser.add_argument("--bsz", type=int, default=256,
                        help="Batch size.")
    parser.add_argument("--userLayers", nargs="?", default="[512, 32]",
                        help="Size of each user layer")
    parser.add_argument("--itemLayers", nargs="?", default="[1024, 32]",
                        help="Size of each item layer")
    parser.add_argument("--fcLayers", nargs="?", default="[512, 256, 128, 32]",
                        help="Size of each layer. Note that the first layer is the "
                             "concatenation of user and item embeddings. So fcLayers[0]/2 is the embedding size.")
    parser.add_argument("--nNeg", type=int, default=4,
                        help="Number of negative instances to pair with a positive instance.")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="Learning rate.")
    parser.add_argument("--optim", nargs="?", default="adam",
                        help="Specify an optimizer: adagrad, adam, rmsprop, sgd")
    parser.add_argument("--dmf", nargs="?", default="",
                        help="Specify the pretrain model file for DMF part. If empty, no pretrain will be used")
    parser.add_argument("--mlp", nargs="?", default="",
                        help="Specify the pretrain model file for MLP part. If empty, no pretrain will be used")
    return parser.parse_args()


class CFNet(nn.Module):
    def __init__(self, userLayers, itemLayers, fcLayers, userMatrix, itemMatrix, dmf, mlp):
        super(CFNet, self).__init__()
        assert userLayers[-1] == itemLayers[-1], "The last layer size of 'userLayers' and 'itemLayers' must be the same!"
        self.register_buffer("userMatrix", userMatrix)
        self.register_buffer("itemMatrix", itemMatrix)
        nUsers = self.userMatrix.size(0)
        nItems = self.itemMatrix.size(0)

        # In the official implementation, 
        # the first dense layer has no activation
        layers = []
        layers.append(nn.Linear(nItems, userLayers[0]))
        for l1, l2 in zip(userLayers[:-1], userLayers[1:]):
            layers.append(nn.Linear(l1, l2))
            layers.append(nn.ReLU(inplace=True))
        self.userModel = nn.Sequential(*layers)
        
        layers = []
        layers.append(nn.Linear(nUsers, itemLayers[0]))
        for l1, l2 in zip(itemLayers[:-1], itemLayers[1:]):
            layers.append(nn.Linear(l1, l2))
            layers.append(nn.ReLU(inplace=True))
        self.itemModel = nn.Sequential(*layers)

        # In the official implementation, 
        # the first dense layer has no activation
        self.userFC = nn.Linear(nItems, 32)
        self.itemFC = nn.Linear(nUsers, 32)
        
        # cnn setting
        self.channel_size = 1
        self.kernel_size = 1
        self.strides = 1
        self.cnn = nn.Sequential(
            # batch_size * 1 * 32 * 32
            nn.Conv2d(in_channels = 1, out_channels = self.channel_size, kernel_size = self.kernel_size, stride=self.strides),
            nn.ReLU()
            # batch_size * 1 * 32 * 32
#             nn.Conv2d(in_channels = self.channel_size, out_channels = self.channel_size, kernel_size = self.kernel_size, stride=self.strides),
#             nn.ReLU(),
#             # batch_size * 1 * 32 * 32
#             nn.Conv2d(in_channels = self.channel_size, out_channels = self.channel_size, kernel_size =  self.kernel_size, stride=self.strides),
#             nn.ReLU(),
#             # batch_size * 1 * 32 * 32
#             nn.Conv2d(in_channels = self.channel_size, out_channels =self.channel_size,  kernel_size = self.kernel_size, stride=self.strides),
#             nn.ReLU()
         )
        
        self.flatten = nn.Flatten()
 
        
        # Define the final part and copy weights from DMF and MLP
        self.final = nn.Sequential(
            nn.Linear(2048,  1),
            nn.Sigmoid(),
        )

        
 
        if dmf and mlp:
            dmfModel = DMF(userLayers, itemLayers, self.userMatrix, self.itemMatrix)
            mlpModel = MLP(fcLayers, self.userMatrix, self.itemMatrix)
            dmfModel.load_state_dict(torch.load(dmf))
            print(f"Load pretrained DMF from {dmf}")
            mlpModel.load_state_dict(torch.load(mlp))
            print(f"Load pretrained MLP from {mlp}")

            self.copy_weights(dmfModel, mlpModel)

    def copy_weights(self, dmfModel, mlpModel):
        # For DMF part
        for idx, layer in enumerate(self.userModel.children()):
            if isinstance(layer, nn.Linear):
                self.userModel[idx].weight.data.copy_(dmfModel.userModel[idx].weight.data)
                self.userModel[idx].bias.data.copy_(dmfModel.userModel[idx].bias.data)
       
        for idx, layer in enumerate(self.itemModel.children()):
            if isinstance(layer, nn.Linear):
                self.itemModel[idx].weight.data.copy_(dmfModel.itemModel[idx].weight.data)
                self.itemModel[idx].bias.data.copy_(dmfModel.itemModel[idx].bias.data)

                
        # For MLP part
        self.userFC.weight.data.copy_(mlpModel.userFC.weight.data)
        self.userFC.bias.data.copy_(mlpModel.userFC.bias.data)
        self.itemFC.weight.data.copy_(mlpModel.itemFC.weight.data)
        self.itemFC.bias.data.copy_(mlpModel.itemFC.bias.data)
        for idx, layer in enumerate(self.fcs.children()):
            if isinstance(layer, nn.Linear):
                self.fcs[idx].weight.data.copy_(mlpModel.fcs[idx].weight.data)
                self.fcs[idx].bias.data.copy_(mlpModel.fcs[idx].bias.data)

        # For final part
        self.final[0].weight.data.copy_(0.5*torch.cat((dmfModel.final[0].weight.data, mlpModel.final[0].weight.data), -1))
        self.final[0].bias.data.copy_(0.5*(dmfModel.final[0].bias.data + mlpModel.final[0].bias.data))

    def forward(self, user, item):
        userInput = self.userMatrix[user, :]                 # (B, 3706)
        itemInput = self.itemMatrix[item, :]                 # (B, 6040)

        # DMF part
        userVector = self.userModel(userInput)               # (B, userLayers[-1])
        itemVector = self.itemModel(itemInput)               # (B, itemLayers[-1])
        interaction_map_dmf = torch.bmm(userVector.unsqueeze(2), itemVector.unsqueeze(1))
        interaction_map_dmf = interaction_map_dmf.view((-1, 1, 32, 32))
        flatten_dmf = self.flatten(interaction_map_dmf)
        
        # MLP part
        userVector = self.userFC(userInput)                  # (B, fcLayers[0]//2)
        itemVector = self.itemFC(itemInput)                  # (B, fcLayers[0]//2)
        embedding_size = 32
        interaction_map_mlp = torch.bmm(userVector.unsqueeze(2), itemVector.unsqueeze(1))
        interaction_map_mlp = interaction_map_mlp.view((-1, 1, embedding_size, embedding_size))
        interaction_map_mlp = self.cnn(interaction_map_mlp)  # output: batch_size * 1 * 32 * 32  ## 해당 cnn에 1x1 cnn층 쌓기
        flatten_mlp = self.flatten(interaction_map_mlp)

        
        y = torch.cat((flatten_dmf, flatten_mlp), -1)           # (B, userLayers[-1]+fcLayers[-1])
        y = self.final(y)                                    # (B, 1)
        return y

if __name__ == "__main__":
    args = parse_args()
    userLayers = eval(args.userLayers)
    itemLayers = eval(args.itemLayers)
    fcLayers = eval(args.fcLayers)
    topK = 10

    print("DeepCF arguments: %s " %(args))
    os.makedirs("pretrained", exist_ok=True)
    modelPath = f"pretrained/{args.dataset}_CFNet_{time.time()}.pth"

    isCuda = torch.cuda.is_available()
    print(f"Use CUDA? {isCuda}")

    # Loading data
    t1 = time.time()
    dataset = Dataset(args.path + args.dataset)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives

    nUsers, nItems = train.shape
    print(f"Load data: #user={nUsers}, #item={nItems}, #train={train.nnz}, #test={len(testRatings)} [{time.time()-t1:.1f}s]")
    
    # Build model
    userMatrix = torch.Tensor(get_train_matrix(train))
    itemMatrix = torch.transpose(torch.Tensor(get_train_matrix(train)), 0, 1)
    if isCuda:
        userMatrix, itemMatrix = userMatrix.cuda(), itemMatrix.cuda()
    
    model = CFNet(userLayers, itemLayers, fcLayers, userMatrix, itemMatrix, args.dmf, args.mlp)
    if isCuda:
        model.cuda()
    torch.save(model.state_dict(), modelPath)
    
    optimizer = get_optimizer(args.optim, args.lr, model.parameters())
    criterion = torch.nn.BCELoss()

    # Check Init performance
    t1 = time.time()
    hits, ndcgs = evaluate_model(model, testRatings, testNegatives, topK, num_thread=1)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print(f"Init: HR={hr:.4f}, NDCG={ndcg:.4f} [{time.time()-t1:.1f}s]")
    bestHr, bestNdcg, bestEpoch = hr, ndcg, -1
    
    # Train model
    model.train()
    for epoch in range(args.epochs):
        t1 = time.time()
        # Generate training instances
        userInput, itemInput, labels = get_train_instances(train, args.nNeg)
        dst = BatchDataset(userInput, itemInput, labels)
        ldr = torch.utils.data.DataLoader(dst, batch_size=args.bsz, shuffle=True)
        losses = AverageMeter("Loss")
        for ui, ii, lbl in ldr:
            if isCuda:
                ui, ii, lbl = ui.cuda(), ii.cuda(), lbl.cuda()
            ri = model(ui, ii).squeeze()
            loss = criterion(ri, lbl)

            # Update model and loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.update(loss.item(), lbl.size(0))

        print(f"Epoch {epoch+1}: Loss={losses.avg:.4f} [{time.time()-t1:.1f}s]")

        # Evaluation
        t1 = time.time()
        hits, ndcgs = evaluate_model(model, testRatings, testNegatives, topK, num_thread=1)
        hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
        print(f"Epoch {epoch+1}: HR={hr:.4f}, NDCG={ndcg:.4f} [{time.time()-t1:.1f}s]")
        if hr > bestHr:
            bestHr, bestNdcg, bestEpoch = hr, ndcg, epoch
            torch.save(model.state_dict(), modelPath)

    print(f"Best epoch {bestEpoch+1}:  HR={bestHr:.4f}, NDCG={bestNdcg:.4f}")
    print(f"The best CFNet model is saved to {modelPath}")
