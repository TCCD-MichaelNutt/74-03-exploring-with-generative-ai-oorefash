import os
import os.path
import numpy as np
import glob
import torch.utils.data as data
from PIL import Image
import h5py
from networkTool import trainDataRoot, levelNumK

# Supported dataset identifiers
IMG_EXTENSIONS = [
    'MPEG',
    'MVUB'
]

# Check if filename contains a valid dataset identifier
def is_image_file(filename):
    return any(extension in filename for extension in IMG_EXTENSIONS)

# Default function to load .mat-style data using h5py
def default_loader(path):
    mat = h5py.File(path)
    cell = mat['patchFile']
    return cell, mat

# Custom PyTorch Dataset class to load octree data
class DataFolder(data.Dataset):
    """ DataFolder can be used to load voxel/octree data from .mat files (converted from .ply). """

    def __init__(self, root, TreePoint, dataLenPerFile, transform=None, loader=default_loader):
        # TreePoint is the number of octree codes per data segment
        # dataLenPerFile is the average number of octnodes per .mat file

        dataNames = []
        for filename in sorted(glob.glob(root)):
            if is_image_file(filename):
                dataNames.append('{}'.format(filename))

        self.root = root
        self.dataNames = sorted(dataNames)
        self.transform = transform
        self.loader = loader
        self.index = 0               # Current index into loaded buffer
        self.datalen = 0             # Number of octree entries in buffer
        self.dataBuffer = []        # Stores loaded octree data temporarily
        self.fileIndx = 0           # Index for the current file being processed
        self.TreePoint = TreePoint
        self.fileLen = len(self.dataNames)
        assert self.fileLen > 0, 'no file found!'
        self.dataLenPerFile = dataLenPerFile

    # Optional method to calculate average number of octree nodes per file (used once, can be disabled)
    def calcdataLenPerFile(self):
        dataLenPerFile = 0
        for filename in self.dataNames:
            cell, mat = self.loader(filename)
            for i in range(cell.shape[1]):
                dataLenPerFile += mat[cell[0, i]].shape[2]
        dataLenPerFile = dataLenPerFile / self.fileLen
        print('dataLenPerFile:', dataLenPerFile, 'you just use this function for the first time')
        return dataLenPerFile

    # Retrieve a batch of TreePoint consecutive octree codes
    def __getitem__(self, index):
        while (self.index + self.TreePoint > self.datalen):
            filename = self.dataNames[self.fileIndx]

            # If buffer exists, retain the tail portion
            if self.dataBuffer:
                a = [self.dataBuffer[0][self.index:].copy()]
            else:
                a = []

            cell, mat = self.loader(filename)
            for i in range(cell.shape[1]):
                data = np.transpose(mat[cell[0, i]])  # Reorder dimensions
                data[:, :, 0] = data[:, :, 0] - 1      # Convert 1-based indexing to 0-based
                a.append(data[:, -levelNumK:, :])      # Retain only last K level features

            self.dataBuffer = []
            self.dataBuffer.append(np.vstack(tuple(a)))  # Stack all patches vertically

            self.datalen = self.dataBuffer[0].shape[0]    # Update buffer length
            self.fileIndx += 200                          # Jump to next set of files (stride = 200)
            self.index = 0
            if self.fileIndx >= self.fileLen:
                self.fileIndx = index % self.fileLen

        # Extract the required batch from buffer
        img = []
        img.append(self.dataBuffer[0][self.index:self.index + self.TreePoint])
        self.index += self.TreePoint

        if self.transform is not None:
            img = self.transform(img)
        return img

    # Returns total number of batches possible
    def __len__(self):
        return int(self.dataLenPerFile * self.fileLen / self.TreePoint)

# For testing the dataset and loader
if __name__ == "__main__":

    TreePoint = 4096 * 16  # Number of consecutive octree codes per sample (TreePoint * batchSize should be divisible by batchSize)
    batchSize = 32

    # Instantiate dataset and DataLoader
    train_set = DataFolder(
        root=trainDataRoot,
        TreePoint=TreePoint,
        transform=None,
        dataLenPerFile=356484.1
    )

    train_loader = data.DataLoader(
        dataset=train_set,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )

    print('total octrees(TreePoint*7): {}; total batches: {}'.format(len(train_set), len(train_loader)))

    # Iterate through batches and display shape
    for batch, d in enumerate(train_loader):
        data_source = d[0].reshape((batchSize, -1, 4, 6)).permute(1, 0, 2, 3)
        print(batch, data_source.shape)
