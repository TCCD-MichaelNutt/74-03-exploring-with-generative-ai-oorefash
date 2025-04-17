'''
Author: fuchy@stu.pku.edu.cn
Description: Octree
FilePath: /compression/Octree.py
All rights reserved.
'''
import numpy as np
from OctreeCPP.Octreewarpper import GenOctree

# Class to represent a node in the octree structure
class CNode():
    def __init__(self, nodeid=0, childPoint=[[]]*8, parent=0, oct=0, pos = np.array([0,0,0]), octant = 0) -> None:
        self.nodeid = nodeid  # Unique identifier for the node
        self.childPoint = childPoint.copy()  # Child points of the node (list of 8 children)
        self.parent = parent  # Parent node identifier
        self.oct = oct  # Occupancy code (1~255)
        self.pos = pos  # Position of the node (coordinates in space)
        self.octant = octant  # Octant (1~8) of the node

# Class to represent an octree
class COctree():
    def __init__(self, node=[], level=0) -> None:
        self.node = node.copy()  # List of nodes at a specific level in the octree
        self.level = level  # The level of the octree

# Function to convert a decimal integer to binary with a fixed number of bits
def dec2bin(n, count=8):
    """Returns the binary of integer n, using count number of digits"""
    return [int((n >> y) & 1) for y in range(count-1, -1, -1)]

# Function to convert an array of decimals to binary arrays with specified bit-width
def dec2binAry(x, bits):
    mask = np.expand_dims(2**np.arange(bits-1,-1,-1), 1).T  # Create a bitmask
    return (np.bitwise_and(np.expand_dims(x, 1), mask) != 0).astype(int)  # Apply mask to generate binary representation

# Function to convert binary arrays back to decimals
def bin2decAry(x):
    if(x.ndim == 1):
        x = np.expand_dims(x, 0)
    bits = x.shape[1]
    mask = np.expand_dims(2**np.arange(bits-1, -1, -1), 1)
    return x.dot(mask).astype(int)  # Dot product with the bitmask to convert back to decimal

# Function to compute Morton code (also known as Z-order) for 3D points
def Morton(A):
    A = A.astype(int)
    n = np.ceil(np.log2(np.max(A) + 1)).astype(int)  # Compute the bit-width based on the largest value
    x = dec2binAry(A[:, 0], n)  # Convert X, Y, Z coordinates to binary
    y = dec2binAry(A[:, 1], n)
    z = dec2binAry(A[:, 2], n)
    m = np.stack((x, y, z), 2)  # Stack binary arrays along the 3rd axis (3 channels for X, Y, Z)
    m = np.transpose(m, (0, 2, 1))  # Transpose to reorder the bits for Morton encoding
    mcode = np.reshape(m, (A.shape[0], 3 * n), order='F')  # Flatten the array into Morton codes
    return mcode

# Function to decode an octree from occupancy codes
def DeOctree(Codes):
    Codes = np.squeeze(Codes)  # Remove single-dimensional entries
    occupancyCode = np.flip(dec2binAry(Codes, 8), axis=1)  # Convert codes to binary and flip order
    codeL = occupancyCode.shape[0]  # Number of occupancy codes
    N = np.ones((30), int)  # Preallocate for level counts
    codcal = 0
    L = 0
    # Loop to determine levels and occupancy
    while codcal + N[L] <= codeL:
        L += 1
        try:
            N[L + 1] = np.sum(occupancyCode[codcal:codcal + N[L], :])
        except:
            assert 0
        codcal = codcal + N[L]
    Lmax = L  # Maximum level
    Octree = [COctree() for _ in range(Lmax + 1)]  # Create a list of octrees at different levels
    proot = [np.array([0, 0, 0])]  # Root node at level 0
    Octree[0].node = proot  # Assign the root node to the first octree
    codei = 0
    # Loop over levels and construct the octree by creating child nodes
    for L in range(1, Lmax + 1):
        childNode = []  # List for child nodes at the current level
        for currentNode in Octree[L - 1].node:  # Process nodes from the previous level
            code = occupancyCode[codei, :]
            for bit in np.where(code == 1)[0].tolist():  # Create children based on occupancy code
                newnode = currentNode + (np.array(dec2bin(bit, count=3)) << (Lmax - L))  # Child node position
                childNode.append(newnode)
            codei += 1
        Octree[L].node = childNode.copy()  # Assign child nodes to the current level's octree
    points = np.array(Octree[Lmax].node)  # Return points from the final octree level
    return points

# Function to generate the K-parent sequence of an octree
def GenKparentSeq(Octree, K):
    LevelNum = len(Octree)  # Number of levels in the octree
    nodeNum = Octree[-1].node[-1].nodeid  # Number of nodes at the last level
    Seq = np.ones((nodeNum, K), 'int') * 255  # Sequence to store occupancy codes for each node's K ancestors
    LevelOctant = np.zeros((nodeNum, K, 2), 'int')  # Store Level and Octant for each node's ancestors
    Pos = np.zeros((nodeNum, K, 3), 'int')  # Position of ancestors (initialized to zero)
    ChildID = [[] for _ in range(nodeNum)]  # Child IDs for each node
    Seq[0, K - 1] = Octree[0].node[0].oct  # Initialize root node's oct code
    LevelOctant[0, K - 1, 0] = 1  # Root node's level
    LevelOctant[0, K - 1, 1] = 1  # Root node's octant
    Pos[0, K - 1, :] = Octree[0].node[0].pos  # Root node's position
    Octree[0].node[0].parent = 1  # Set root's parent to itself (1)
    n = 0
    # Loop over levels to fill the K-parent sequence
    for L in range(0, LevelNum):
        for node in Octree[L].node:
            Seq[n, K - 1] = node.oct  # Occupancy code of the current node
            Seq[n, 0:K - 1] = Seq[node.parent - 1, 1:K]  # Propagate parent sequence
            LevelOctant[n, K - 1, :] = [L + 1, node.octant]  # Level and Octant of current node
            LevelOctant[n, 0:K - 1] = LevelOctant[node.parent - 1, 1:K, :]  # Propagate parent's level and octant
            Pos[n, K - 1] = node.pos  # Position of the current node
            Pos[n, 0:K - 1, :] = Pos[node.parent - 1, 1:K, :]  # Propagate parent's position
            if (L == LevelNum - 1):
                pass
            n += 1
    assert n == nodeNum  # Ensure the number of processed nodes matches the expected count
    DataStruct = {'Seq': Seq, 'Level': LevelOctant, 'ChildID': ChildID, 'Pos': Pos}  # Return the data structure
    return DataStruct
