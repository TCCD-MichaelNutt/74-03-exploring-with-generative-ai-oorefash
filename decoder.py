'''
Author: fuchy@stu.pku.edu.cn
Date: 2021-09-17 23:30:48
LastEditTime: 2021-12-02 22:18:56
LastEditors: FCY
Description: decoder
FilePath: /compression/decoder.py
All rights reserved.
'''
#%%
import numpy as np
import torch
from tqdm import tqdm
from Octree import DeOctree, dec2bin
import pt
from dataset import default_loader as matloader
from collections import deque
import os
import time
from networkTool import *
from encoderTool import generate_square_subsequent_mask
from encoder import model, list_orifile
import numpyAc

batch_size = 1
bpttRepeatTime = 1

#%%
'''
description: decode bin file to occupancy code
param {str;input bin file name} binfile
param {N*1 array; occupancy code, only used for check} oct_data_seq
param {model} model
param {int; Context window length} bptt
return {N*1,float}occupancy code,time
'''
def decodeOct(binfile, oct_data_seq, model, bptt):
    model.eval()  # set model to evaluation mode
    with torch.no_grad():  # no gradient calculation during inference
        elapsed = time.time()

        KfatherNode = [[255,0,0]]*levelNumK  # initialize parent node list
        nodeQ = deque()  # queue for decoding tree nodes
        oct_seq = []  # stores decoded octree sequence
        src_mask = generate_square_subsequent_mask(bptt).to(device)  # create attention mask

        # initialize input tensor for transformer (bptt window)
        input = torch.zeros((bptt, batch_size, levelNumK, 3)).long().to(device)
        padinginbptt = torch.zeros((bptt, batch_size, levelNumK, 3)).long().to(device)
        bpttMovSize = bptt // bpttRepeatTime  # how much we move the bptt window each step

        # run model once with all zeros to get initial prediction
        output = model(input, src_mask, [])

        # get softmax probability of last output
        freqsinit = torch.softmax(output[-1], 1).squeeze().cpu().detach().numpy()

        oct_len = len(oct_data_seq)  # number of nodes to decode

        # decode the bin file using arithmetic decoder
        dec = numpyAc.arithmeticDeCoding(None, oct_len, 255, binfile)

        # decode the first root node
        root = decodeNode(freqsinit, dec)
        nodeId = 0

        # pad initial parent node information
        KfatherNode = KfatherNode[3:] + [[root,1,1]] + [[root,1,1]]

        nodeQ.append(KfatherNode)
        oct_seq.append(root)  # append root to result

        with tqdm(total=oct_len+10) as pbar:
            while True:
                father = nodeQ.popleft()
                childOcu = dec2bin(father[-1][0])  # decode occupancy bits
                childOcu.reverse()
                faterLevel = father[-1][1]

                for i in range(8):  # loop through 8 possible child nodes
                    if(childOcu[i]):
                        # prepare input for transformer
                        faterFeat = [[father + [[root, faterLevel+1, i+1]]]]
                        faterFeatTensor = torch.Tensor(faterFeat).long().to(device)
                        faterFeatTensor[:,:,:,0] -= 1  # shift octree indices

                        offsetInbpttt = (nodeId) % (bpttMovSize)  # current position in the window
                        if offsetInbpttt == 0:
                            # roll the window forward
                            input = torch.vstack((input[bpttMovSize:], faterFeatTensor, padinginbptt[0:bpttMovSize-1]))
                        else:
                            input[bptt-bpttMovSize+offsetInbpttt] = faterFeatTensor

                        # run model and get prediction
                        output = model(input, src_mask, [])
                        Pro = torch.softmax(output[offsetInbpttt + bptt - bpttMovSize], 1).squeeze().cpu().detach().numpy()

                        root = decodeNode(Pro, dec)
                        nodeId += 1
                        pbar.update(1)

                        # store new node and continue
                        KfatherNode = father[1:] + [[root, faterLevel+1, i+1]]
                        nodeQ.append(KfatherNode)

                        if(root == 256 or nodeId == oct_len):
                            assert len(oct_data_seq) == nodeId  # sanity check
                            Code = oct_seq
                            return Code, time.time() - elapsed

                        oct_seq.append(root)

                    assert oct_data_seq[nodeId] == root  # validate decoding

# helper to decode a node using the arithmetic decoder
def decodeNode(pro, dec):
    root = dec.decode(np.expand_dims(pro, 0))
    return root + 1


if __name__ == "__main__":

    for oriFile in list_orifile:  # list of original files (from encoder.py)
        ptName = os.path.basename(oriFile)[:-4]  # get file base name
        matName = 'Data/testPly/' + ptName + '.mat'  # .mat file path
        binfile = expName + '/data/' + ptName + '.bin'  # encoded binary file path
        cell, mat = matloader(matName)  # load .mat file

        # load side information for decoding
        oct_data_seq = np.transpose(mat[cell[0,0]]).astype(int)[:,-1:,0]  # true occupancy sequence for validation
        p = np.transpose(mat[cell[1,0]]['Location'])  # original point cloud
        offset = np.transpose(mat[cell[2,0]]['offset'])  # offset vector
        qs = mat[cell[2,0]]['qs'][0]  # quantization step size

        Code, elapsed = decodeOct(binfile, oct_data_seq, model, bptt)  # perform decoding
        print('decode success, time:', elapsed)
        print('oct len:', len(Code))

        # reconstruct point cloud from decoded occupancy code
        ptrec = DeOctree(Code)
        DQpt = (ptrec * qs + offset)  # apply dequantization
        pt.write_ply_data(expName + "/temp/test/rec.ply", DQpt)  # save to .ply
        pt.pcerror(p, DQpt, None, '-r 1', None).wait()  # evaluate reconstruction accuracy
