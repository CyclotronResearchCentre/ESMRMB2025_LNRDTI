#!/home/fuji/anaconda3/bin/python

import sys, os, getopt, datetime
import numpy as np
import nibabel as nib
import shutil
from dipy.io import read_bvals_bvecs

subjDir = sys.argv[1]
n_vols = int(sys.argv[2])
dwi = sys.argv[3]
bsmallest = sys.argv[4]
dirNII = os.path.join(subjDir,"dwi")
if os.path.exists(dirNII):
         ffiles = os.listdir(dirNII)
         toKeep = -1
         # Here we will check which bsmall volume of the reversedDWI we should keep, based on the smallest given by the previous step as input here. 
         for ff,ffile in enumerate(ffiles):
            if "%s" % dwi in ffile and ".bval" in ffile and "S0" not in ffile:
             name = ffile.split(".")[0]
             fbval = os.path.join(dirNII,ffile)
             fbvec = os.path.join(dirNII,"%s.bvec" % name)
             bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
             for bv,bvu in enumerate(bvals):
              if int(bvu) == int(bsmallest):
                  toKeep = bv
         if toKeep != -1:
          for ff,ffile in enumerate(ffiles):
            if "%s" % dwi in ffile and ".bval" in ffile and "S0" not in ffile:
               name = ffile.split(".")[0]
               name_new = "%s_S0.bval" % name
               dti = os.path.join(dirNII,ffile)
               dti_new = os.path.join(dirNII,name_new)
               dti_new2 = dti
               with open(dti_new,'w') as bval_new:
                with open(dti,'r') as bval:
                   entry = bval.read()
                   entries = entry.split(" ")
                   num = len(entries)
                   for ee,eee in enumerate(entries):
                       bval_new.write("%s " % eee)
               with open(dti_new2,'w') as bval_new2:
                with open(dti_new,'r') as bval3:
                   entry = bval3.read()
                   entries = entry.split(" ")
                   num = len(entries)
                   for ee,eee in enumerate(entries):
                    if ee == toKeep:
                       bval_new2.write("%s " % eee) 
               os.remove(dti_new)
            elif "%s" % dwi in ffile and ".bvec" in ffile and "S0" not in ffile:
               name = ffile.split(".")[0]
               name_new = "%s_S0.bvec" % name
               dti = os.path.join(dirNII,ffile)
               dti_new = os.path.join(dirNII,name_new)
               dti_new2 = dti
               with open(dti_new,'w') as bval_new:
                with open(dti,'r') as bval:
                 entry = bval.read()
                 entries1 = entry.split("\n")
                 for ee1,eee1 in enumerate(entries1):
                   entries = eee1.split(" ")
                   num = len(entries)
                   for ee,eee in enumerate(entries):
                       bval_new.write("%s " % eee)
                   bval_new.write("\n")
               with open(dti_new2,'w') as bval_new2:
                with open(dti_new,'r') as bval3:
                 entry = bval3.read()
                 entries1 = entry.split("\n")
                 for ee1,eee1 in enumerate(entries1):
                   entries = eee1.split(" ")
                   num = len(entries)
                   for ee,eee in enumerate(entries):
                    if ee == toKeep:
                       bval_new2.write("%s " % eee) 
                   bval_new2.write("\n")
               os.remove(dti_new)
            elif "%s" % dwi in ffile and ".nii" in ffile and "S0" not in ffile:
               name = ffile.split(".")[0]
               name_new = "%s_S0.nii.gz" % name
               dti = os.path.join(dirNII,ffile)
               dti_new = os.path.join(dirNII,name_new)
               try:
                im = nib.load(dti)
                im_data = im.get_fdata()
                foundGood = False
                if len(im_data.shape) > 3:
                 im_data_new = im_data[:,:,:,toKeep]  
                 foundGood = True
                else:
                 if len(im_data.shape) > 2:
                  im_data_new = im_data[:,:,:] 
                  foundGood = True
                if foundGood:
                 toSave = nib.Nifti1Image(im_data_new, affine=im.affine, header=im.header) 
                 nib.save(toSave, "%s" % dti)
               except nib.filebasedimages.ImageFileError:
                      print("Don't work on %s" % dti) 
