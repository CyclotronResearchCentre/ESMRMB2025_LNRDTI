#!/usr/bin/env python3

import pandas as pd  # <-- Added pandas
import os,re
import nibabel as nib
import numpy as np

def compute_fa_symetry_score(fa_path, tract_path, output_path, formula = "symmetry"):
    """
    Loads fa_path and tract_path as 3D NIfTI images, multiplies them voxel-by-voxel,
    and saves the result as output_path.
    """
    # Load FA volume
    fa_img = nib.load(fa_path)
    fa_data = fa_img.get_fdata()
    fa_data = np.nan_to_num(fa_data, nan=0.0)  # Replace NaN with 0

    # Load tract volume
    tract_img = nib.load(tract_path)
    tract_data = tract_img.get_fdata()

    # Check that both volumes have the same shape
    if fa_data.shape != tract_data.shape:
        raise ValueError("FA and tract volumes have different shapes!")

    # Multiply the volumes voxel-by-voxel
    weighted_data = fa_data * tract_data

    # Create a new NIfTI image using the FA image's affine and header
    weighted_img = nib.Nifti1Image(weighted_data, fa_img.affine, fa_img.header)

    # Save the output
    nib.save(weighted_img, output_path)
    print(f"Saved multiplied volume to {output_path}")

    xmid = tract_data.shape[0] // 2

    left_tracks = tract_data[:xmid, :, :]
    right_tracks = tract_data[xmid:, :, :]

    left_fa = fa_data[:xmid, :, :]
    right_fa = fa_data[xmid:, :, :]

    # Weighted sum approach
    left_sum = np.sum(left_tracks * left_fa)
    right_sum = np.sum(right_tracks * right_fa)

    if formula == "assymetry":
        score = (right_sum - left_sum) / (right_sum + left_sum)
    else:
        score = 2.0 * min(left_sum, right_sum) / (left_sum + right_sum)
        
    return score
    
def compute_tdi_symmetry_score(tdi_path,formula = "relative"):
    """
    Load a track density image (TDI) and compute a simple left-right symmetry score.
    Returns a value in [0, 1], where 1 = perfectly symmetric, 0 = no overlap.
    """
    img = nib.load(tdi_path)
    data = img.get_fdata()

    x_size = data.shape[0]
    x_mid = x_size // 2

    left_sum = data[:x_mid, :, :].sum()
    right_sum = data[x_mid:, :, :].sum()

    if (left_sum + right_sum) == 0:
        return 0.0

    if formula == "assymetry":
        score = (right_sum - left_sum) / (right_sum + left_sum)
    else:
        score = 2.0 * min(left_sum, right_sum) / (left_sum + right_sum)
    return score

# The following data-directory is organised as follows: first later -> diffusion protocols, second layer -> subjects, third layer -> postprocessing pipelines
dataDir = "/path/to/BIDS/dataset/with/tractography/results"
results1, results2,results3, results4,results5, results6,results7, results8,results9, results10 = [],[],[],[],[],[],[],[],[],[]
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
                        tractALL = None
                        tractL3 = None
                        tractL4 = None
                        tractL5 = None
                        tractS1 = None
                        FA = None
                        # 4) Look for relevant files
                        for ffile_name in os.listdir(dwi_path):
                            ffile_path = os.path.join(dwi_path, ffile_name)
                            if "ALL_ROIs_MERGED_tracto.nii.gz" in ffile_name:
                                tractALL = ffile_path    
                            elif "L3_ROIs_MERGED_tracto.nii.gz" in ffile_name:
                                tractL3 = ffile_path    
                            elif "L4_ROIs_MERGED_tracto.nii.gz" in ffile_name:
                                tractL4 = ffile_path
                            elif "L5_ROIs_MERGED_tracto.nii.gz" in ffile_name:
                                tractL5 = ffile_path    
                            elif "S1_ROIs_MERGED_tracto.nii.gz" in ffile_name:
                                tractS1 = ffile_path
                        FAthere = os.path.join(dataDir,dwi,sub,strategy,"pipeline_output", "DATASINK", "DWISpace","selectedDWI_tensor_FA.nii.gz")
                        if os.path.exists(FAthere):
                                FA = FAthere
                        # 5) If both are found, multiply them via fslmaths
                        if tractALL and FA:
                            print("Now at: %s - %s"%(dwi,sub))
                            filenameFA = os.path.join(dwi_path, "weighted_FA.nii.gz")
                            score1 = compute_tdi_symmetry_score(tractALL)
                            results1.append((dwi,sub,strategy,seedcutoff_value,cutoff_value,minlength_value,maxlength_value,step_value,seeds_value,select_value,score1,))
                            score2 = compute_fa_symetry_score(FA,tractALL,filenameFA)
                            results2.append((dwi,sub,strategy,seedcutoff_value,cutoff_value,minlength_value,maxlength_value,step_value,seeds_value,select_value,score2,))
                        if tractL3 and FA:
                            score3 = compute_tdi_symmetry_score(tractL3)
                            results3.append((dwi,sub,strategy,seedcutoff_value,cutoff_value,minlength_value,maxlength_value,step_value,seeds_value,select_value,score3,))
                            score4 = compute_fa_symetry_score(FA,tractL3,filenameFA)
                            results4.append((dwi,sub,strategy,seedcutoff_value,cutoff_value,minlength_value,maxlength_value,step_value,seeds_value,select_value,score4,))
                        if tractL4 and FA:
                            score5 = compute_tdi_symmetry_score(tractL4)
                            results5.append((dwi,sub,strategy,seedcutoff_value,cutoff_value,minlength_value,maxlength_value,step_value,seeds_value,select_value,score5,))
                            score6 = compute_fa_symetry_score(FA,tractL4,filenameFA)
                            results6.append((dwi,sub,strategy,seedcutoff_value,cutoff_value,minlength_value,maxlength_value,step_value,seeds_value,select_value,score6,))
                        if tractL5 and FA:
                            score7 = compute_tdi_symmetry_score(tractL5)
                            results7.append((dwi,sub,strategy,seedcutoff_value,cutoff_value,minlength_value,maxlength_value,step_value,seeds_value,select_value,score7,))
                            score8 = compute_fa_symetry_score(FA,tractL5,filenameFA)
                            results8.append((dwi,sub,strategy,seedcutoff_value,cutoff_value,minlength_value,maxlength_value,step_value,seeds_value,select_value,score8,))
                        if tractS1 and FA:
                            score9 = compute_tdi_symmetry_score(tractS1)
                            results9.append((dwi,sub,strategy,seedcutoff_value,cutoff_value,minlength_value,maxlength_value,step_value,seeds_value,select_value,score9,))
                            score10 = compute_fa_symetry_score(FA,tractS1,filenameFA)
                            results10.append((dwi,sub,strategy,seedcutoff_value,cutoff_value,minlength_value,maxlength_value,step_value,seeds_value,select_value,score10,))

with pd.ExcelWriter(f"{dataDir}/symmetry_results.xlsx", engine='openpyxl') as writer:
    pd.DataFrame(results1, columns=["DWI","Subject","Strategy","min_FOD_to_start","min_FOD_to_continue","min_length_of_streamline","max_length_of_streamline","advancing_distance","maximum_seeds","select_among_max_seeds","Symmetry"]).to_excel(writer, index=False, sheet_name='ALL_TDI')
    pd.DataFrame(results2, columns=["DWI","Subject","Strategy","min_FOD_to_start","min_FOD_to_continue","min_length_of_streamline","max_length_of_streamline","advancing_distance","maximum_seeds","select_among_max_seeds","Symmetry"]).to_excel(writer, index=False, sheet_name='ALL_FA')
    pd.DataFrame(results3, columns=["DWI","Subject","Strategy","min_FOD_to_start","min_FOD_to_continue","min_length_of_streamline","max_length_of_streamline","advancing_distance","maximum_seeds","select_among_max_seeds","Symmetry"]).to_excel(writer, index=False, sheet_name='L3_TDI')
    pd.DataFrame(results4, columns=["DWI","Subject","Strategy","min_FOD_to_start","min_FOD_to_continue","min_length_of_streamline","max_length_of_streamline","advancing_distance","maximum_seeds","select_among_max_seeds","Symmetry"]).to_excel(writer, index=False, sheet_name='L3_FA')
    pd.DataFrame(results5, columns=["DWI","Subject","Strategy","min_FOD_to_start","min_FOD_to_continue","min_length_of_streamline","max_length_of_streamline","advancing_distance","maximum_seeds","select_among_max_seeds","Symmetry"]).to_excel(writer, index=False, sheet_name='L4_TDI')
    pd.DataFrame(results6, columns=["DWI","Subject","Strategy","min_FOD_to_start","min_FOD_to_continue","min_length_of_streamline","max_length_of_streamline","advancing_distance","maximum_seeds","select_among_max_seeds","Symmetry"]).to_excel(writer, index=False, sheet_name='L4_FA')
    pd.DataFrame(results7, columns=["DWI","Subject","Strategy","min_FOD_to_start","min_FOD_to_continue","min_length_of_streamline","max_length_of_streamline","advancing_distance","maximum_seeds","select_among_max_seeds","Symmetry"]).to_excel(writer, index=False, sheet_name='L5_TDI')
    pd.DataFrame(results8, columns=["DWI","Subject","Strategy","min_FOD_to_start","min_FOD_to_continue","min_length_of_streamline","max_length_of_streamline","advancing_distance","maximum_seeds","select_among_max_seeds","Symmetry"]).to_excel(writer, index=False, sheet_name='L5_FA')
    pd.DataFrame(results9, columns=["DWI","Subject","Strategy","min_FOD_to_start","min_FOD_to_continue","min_length_of_streamline","max_length_of_streamline","advancing_distance","maximum_seeds","select_among_max_seeds","Symmetry"]).to_excel(writer, index=False, sheet_name='S1_TDI')
    pd.DataFrame(results10, columns=["DWI","Subject","Strategy","min_FOD_to_start","min_FOD_to_continue","min_length_of_streamline","max_length_of_streamline","advancing_distance","maximum_seeds","select_among_max_seeds","Symmetry"]).to_excel(writer, index=False, sheet_name='S1_FA')
