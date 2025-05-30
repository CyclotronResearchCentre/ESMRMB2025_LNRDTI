import nibabel as nib
import numpy as np
import sys,os

bval_init=sys.argv[1]
saveDir=os.path.dirname(bval_init)
name=os.path.basename(bval_init)
nameF = name.replace(".","_new.")
print(nameF)
bval_final = os.path.join(saveDir,nameF)
with open(bval_final,'w') as f2:
 with open(bval_init) as f:
    lines = f.readlines()
    for ll in lines:
        lll = ll.replace("100 ","101 ")
        f2.writelines(lll)
