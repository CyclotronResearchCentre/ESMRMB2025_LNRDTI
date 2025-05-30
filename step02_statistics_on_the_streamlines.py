import nibabel as nib
import numpy as np
import re,os
import pandas as pd  # <-- Added pandas
from openpyxl import Workbook
from nibabel.affines import apply_affine

def compute_length_in_voxels(streamline):
    # Calculate the Euclidean distance between consecutive points and sum them
    return np.sum(np.sqrt(np.sum(np.diff(streamline, axis=0)**2, axis=1)))

def compute_length_in_mm(streamline, affine):
    """
    Compute the total length of a streamline in physical space (mm).
    
    Parameters:
        streamline (np.ndarray): An array of shape (N, 3) containing voxel coordinates.
        affine (np.ndarray): The affine transformation matrix of the image.
    
    Returns:
        float: Total length of the streamline in mm.
    """
    # Convert streamline coordinates from voxel space to mm (physical space)
    streamline_mm = apply_affine(affine, streamline)
    # Calculate Euclidean distances between consecutive points and sum them
    return np.sum(np.sqrt(np.sum(np.diff(streamline_mm, axis=0)**2, axis=1)))

wb = Workbook()
ws = wb.active
# The following data-directory is organised as follows: first later -> diffusion protocols, second layer -> subjects, third layer -> postprocessing pipelines
dataDir = "/path/to/BIDS/dataset/with/tractography/results"
res1, res2, res3, res4, res5 = [],[],[],[],[]
dwis = os.listdir(dataDir)
for dd,dwi in enumerate(dwis):
 subsDir = os.path.join(dataDir,dwi)
 if os.path.isdir(subsDir): 
   subs = os.listdir(subsDir)
   for ss,sub in enumerate(subs):
    strategiesDir = os.path.join(subsDir,sub)
    strategies = os.listdir(strategiesDir)
    for st,strategy in enumerate(strategies):
     tractographyDir = os.path.join(strategiesDir,strategy)
     tractographies = os.listdir(tractographyDir)
     for tracto,tractstrat in enumerate(tractographies):
      if tractstrat.startswith("Tract"):  
       seedcutoff = re.search(r'_seedcutoff_([0-9.]+)', tractstrat)
       cutoff = re.search(r'_cutoff_([0-9.]+)', tractstrat)
       minlength = re.search(r'minlength_([0-9.]+)', tractstrat)
       maxlength = re.search(r'maxlength_([0-9.]+)', tractstrat)
       select = re.search(r'select_([0-9.]+)', tractstrat)
       step = re.search(r'step_([0-9.]+)', tractstrat)
       seeds = re.search(r'seeds_([0-9]+)', tractstrat)
       seedcutoff_value = float(seedcutoff.group(1)) if seedcutoff else None
       cutoff_value = float(cutoff.group(1)) if cutoff else None
       minlength_value = float(minlength.group(1)) if minlength else None
       maxlength_value = float(maxlength.group(1)) if maxlength else None
       select_value = float(select.group(1)) if select else None
       step_value = float(step.group(1)) if step else None
       seeds_value = float(seeds.group(1)) if seeds else None
       tractfilesDir = os.path.join(tractographyDir,tractstrat)
       for tractfile in os.listdir(tractfilesDir):
        dwi_path = os.path.join(tractfilesDir, tractfile,"DATASINK","DWISpace")
        if os.path.exists(dwi_path) and os.path.isdir(dwi_path):
                        the_affine = None
                        for ffile_name in os.listdir(dwi_path):
                            ffile_path = os.path.join(dwi_path, ffile_name)
                            if ".nii.gz" in ffile_name:
                                im_nib = nib.load(ffile_path)
                                the_affine = im_nib.affine
                        for ffile_name in os.listdir(dwi_path):
                            ffile_path = os.path.join(dwi_path, ffile_name)
                            if ".tck" in ffile_name and "include" in ffile_name:
                               # Load the .tck file (make sure nibabel supports .tck format; you might need to update or install nibabel's streamlines module)
                                tck_file = nib.streamlines.load(ffile_path)
                                streamlines = tck_file.streamlines
                               # Compute the lengths for all streamlines
                                lengths = [compute_length_in_mm(s,the_affine) for s in streamlines]
                                if len(lengths) > 0:
                                    match = re.search(r"ROI_([^_]+_[^_]+)", ffile_name)
                                    if match:
                                        nerve = match.group(1)
                                        row = (seedcutoff_value,cutoff_value,minlength_value,maxlength_value,step_value,seeds_value,select_value,dwi,sub,strategy,nerve,len(lengths),np.mean(lengths),np.median(lengths),np.std(lengths),np.min(lengths),np.max(lengths),)
                                        ws.append(row)
wb.save("%s/stats_results.xlsx"%dataDir)
