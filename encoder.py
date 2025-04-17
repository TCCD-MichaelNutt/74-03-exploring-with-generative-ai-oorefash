'''
Author: fuchy@stu.pku.edu.cn
Description: this file encodes point cloud
FilePath: /compression/encoder.py
All rights reserved.
'''
from numpy import mod
from Preparedata.data import dataPrepare  # Function to prepare point cloud data
from encoderTool import main             # Main function for encoding
from networkTool import reload, CPrintl, expName, device  # Utilities for loading model, printing, device setup
from octAttention import model           # OctAttention model definition
import glob, datetime, os
import pt as pointCloud                  # Point cloud utilities (e.g., for computing errors)

############## warning ###############
## decoder.py relies on this model here
## do not move these lines elsewhere

# Move model to the appropriate device (CPU/GPU)
model = model.to(device)

# Load model weights from checkpoint
saveDic = reload(None, 'modelsave/obj/encoder_epoch_00800093.pth')
model.load_state_dict(saveDic['encoder'])

########### Object ##############
# List of original .ply files to process
list_orifile = ['file/Ply/2851.ply']

if __name__ == "__main__":
    # Initialize custom print logger
    printl = CPrintl(expName + '/encoderPLY.txt')

    # Print headers and timestamp
    printl('_' * 50, 'OctAttention V0.4', '_' * 50)
    printl(datetime.datetime.now().strftime('%Y-%m-%d:%H:%M:%S'))
    printl('load checkpoint', saveDic['path'])

    for oriFile in list_orifile:
        printl(oriFile)

        # Skip files larger than 300MB
        if (os.path.getsize(oriFile) > 300 * (1024 ** 2)):  # 300MB
            printl('too large!')
            continue

        # Extract file name without extension
        ptName = os.path.splitext(os.path.basename(oriFile))[0]

        for qs in [1]:  # Quality settings loop (could expand this list)
            ptNamePrefix = ptName

            # Prepare the data (returns matFile, decoded point cloud, and reference point cloud)
            matFile, DQpt, refPt = dataPrepare(
                oriFile,
                saveMatDir='./Data/testPly',
                qs=qs,
                ptNamePrefix='',
                rotation=False
            )
            # NOTE: Set `rotation=True` in `dataPrepare` when working with MVUB dataset

            # Call main encoding function (actualcode=False disables binary file generation)
            main(matFile, model, actualcode=True, printl=printl)

            # Print header for point cloud error evaluation
            print('_' * 50, 'pc_error', '_' * 50)

            # Run error evaluation between reference and decoded point clouds
            pointCloud.pcerror(refPt, DQpt, None, '-r 1023', None).wait()
