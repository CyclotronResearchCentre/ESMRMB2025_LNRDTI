import sys,os,glob,subprocess,nibabel, shutil
from datetime import datetime
import numpy as np

address_for_annotations_init = "/path/to/annotated/data"
pipeline_address = "./"
pipeline1 = os.path.join(pipeline_address, "code", "pipeline_proc.py")
tractography_script= os.path.join(pipeline_address, "code", "pipeline_tractography.py")
data_address = "/path/to/BIDS/dataset"
dataset = data_address.split("/")[-1]
output_address = "/path/to/results"
results_address1 = os.path.join(output_address,f"EvgLNRPipe_on_{dataset}")
if not os.path.exists(results_address1):
    os.makedirs(results_address1)

strategies = ["none","mppca","gibbs","moco","eddy","eddy_topup","mppca_gibbs","mppca_gibbs_moco","mppca_gibbs_moco_eddy","mppca_gibbs_moco_eddy_topup","mppca_moco","mppca_moco_eddy","mppca_moco_eddy_topup","mppca_eddy","mppca_eddy_topup","gibbs_moco","gibbs_moco_eddy","gibbs_moco_eddy_topup","gibbs_eddy","gibbs_eddy_topup","moco_eddy","moco_eddy_topup"]
dwis = ["DTI-AP_2p5mm_ZOOMit","DTI-AP-0-300","DTI-AP-0-600","DTI-AP-0-800_no_body_array","DTI-AP-0-800_with_body_array"]
other_seqs = "none"  # could be: T1w, MT, ihMT
txt = "none"
dwi_mask = "none"  # at the moment, this one is not used
nerve_roots = ["L3","L4","L5","S1"]
sides = ["L","R"]
points = ["S","F"]
tr_seed_cutoff_vals = [0.08]# The minimum amplitude of the FOD to allow seeding, applied before tractography starts. It basically controls where streamlines can originate. If you're missing expected streamlines, try reducing to 0.06; if you're getting too many noisy ones, increase toward 0.1
tr_cutoff_vals = [0.1] # The minimum FOD amplitude required to continue propagating a streamline as it grows. It allows tracking in low FOD regions and it is applied during streamline propagation. Its purpose is to prevent streamlines from continuing into regions with low FOD amplitude (like CSF or noise)
tr_minlength_vals = [25]# Sets the minimum physical length (in mm) for accepted streamline - Lumbosacral roots often span ~30–60 mm depending on segment and patient anatomy, values <20 mm may capture artifacts or CSF noise, you might slightly increase this if you're getting lots of short, looping tracts
tr_maxlength_vals = [100]# Sets the maximum physical length (in mm) for accepted streamline - Lumbosacral roots often span ~30–60 mm depending on segment and patient anatomy, values <20 mm may capture artifacts or CSF noise, you might slightly increase this if you're getting lots of short, looping tracts
tr_select_vals = [10000]# Tells tckgen to stop when exactly N streamlines have been successfully generated (i.e., pass all constraints like length, angle, mask inclusion, etc.). I used the value of 10k for my initial (March 2025) results
tr_seeds_vals = [10]# Sets the total number of seed points to try. Not all seed points will necessarily result in successful streamlines (some will be rejected due to constraints). I used the value of 50k for my initial (March 2025) results
tr_step_vals = [0.5]# Matches ~0.5 × voxel size (for 2 mm voxels). It controls the distance (in mm) that the streamline advances at each tracking iteration during propagation (e.g. the streamline takes 1.0 mm steps as it follows the FOD peaks). Smaller -step → longer computation time and potentially more streamlines failing if the FOD amplitude fluctuates a lot between small steps.
bash = "/bin/bash"
for dd, dwi in enumerate(dwis):
 # global values for the tractography parameters
 for tr1,tr_seed_cutoff_val in enumerate(tr_seed_cutoff_vals):
  for tr2,tr_cutoff_val in enumerate(tr_cutoff_vals):
   for tr3,tr_minlength_val in enumerate(tr_minlength_vals):
    for tr4,tr_maxlength_val in enumerate(tr_maxlength_vals):
     for tr5,tr_select_val in enumerate(tr_select_vals):
      for tr6,tr_seeds_val_init in enumerate(tr_seeds_vals):
       for tr7,tr_step_val in enumerate(tr_step_vals):
        tr_seeds_val = int(tr_seeds_val_init)*int(tr_select_val)
        for strr,strategy in enumerate(strategies):
         if "AP" in dwi:
            dwi_reversed = dwi.replace("AP","PA")
         else:
            dwi_reversed = "none"
         results_txt = f"{results_address1}/pipeline_summary_using_{dwi}.txt"
        
         try:
            results_address2 = os.path.join(results_address1, dwi)
            if not os.path.exists(results_address2):
                os.makedirs(results_address2)

            sequences = other_seqs.split()
            sessions = []
            T2ws = []

            print(f"Current date and time: {datetime.now()}")

            if "none" not in txt.lower():
                with open(txt, 'r') as file:
                    for line in file:
                        session = line.strip()
                        T2w_files = glob.glob(os.path.join(session, '**', f'*T2w_2D.nii*'), recursive=True) + glob.glob(os.path.join(session, '**', f'*T2w_3D.nii*'), recursive=True)
                        sessions.append(session)
                        T2ws.extend(T2w_files)
            else:
                for T2w in glob.glob(os.path.join(data_address, '**', f'*T2w_2D.nii*'), recursive=True) + glob.glob(os.path.join(data_address, '**', f'*T2w_3D.nii*'), recursive=True):
                    T2ws.append(T2w)
                    split_path = T2w.split(os.sep)
                    session = os.path.join('/',*split_path[:-2])  # Getting parent session directory
                    sessions.append(session)

            # Sorting sessions
            sorted_sessions = sorted(set(sessions))
            for session in sorted_sessions:
                T2 = glob.glob(os.path.join(session, '**', f'*T2w_2D.nii*'), recursive=True) + glob.glob(os.path.join(session, '**', f'*T2w_3D.nii*'), recursive=True)
                DWI_NII = glob.glob(os.path.join(session, '**', f'*{dwi}.nii*'), recursive=True)
                DWI_BVAL = glob.glob(os.path.join(session, '**', f'*{dwi}.bval'), recursive=True)
                DWI_BVEC = glob.glob(os.path.join(session, '**', f'*{dwi}.bvec'), recursive=True)
                DWI_JSON = glob.glob(os.path.join(session, '**', f'*{dwi}.json'), recursive=True)
                DWI_REV_NII = glob.glob(os.path.join(session, '**', f'*{dwi_reversed}.nii*'), recursive=True)
                DWI_REV_BVAL = glob.glob(os.path.join(session, '**', f'*{dwi_reversed}.bval'), recursive=True)
                DWI_REV_BVEC = glob.glob(os.path.join(session, '**', f'*{dwi_reversed}.bvec'), recursive=True)
                if len(T2) > 1:
                   T2 = [sorted(T2)[-1]]
                if len(T2) == 1 and len(DWI_NII) == 1 and len(DWI_BVAL) == 1 and len(DWI_BVEC) == 1: 
                    sub_id = os.path.basename(session)
                    results_address3 = os.path.join(results_address2, sub_id, f"{strategy}")
                    if not os.path.exists(results_address3):
                           os.makedirs(results_address3)
                    if dwi_reversed != "none" and len(DWI_REV_NII) == 1 and len(DWI_REV_BVAL) == 1 and len(DWI_REV_BVEC) == 1:
                        script_to_run1 = f"python {pipeline1} -a {results_address3} -b {T2[0]} -c {DWI_NII[0]} -d {DWI_BVAL[0]} -e {DWI_BVEC[0]} -g {DWI_REV_NII[0]} -i {DWI_REV_BVAL[0]} -k {DWI_REV_BVEC[0]} -l {strategy} "
                    else:
                        if "topup" in strategy:
                            continue
                        script_to_run1 = f"python {pipeline1} -a {results_address3} -b {T2[0]} -c {DWI_NII[0]} -d {DWI_BVAL[0]} -e {DWI_BVEC[0]} -l {strategy} " 
                    if len(DWI_JSON) == 1: 
                        script_to_run1 += f"-f {DWI_JSON[0]} "
                    command1 = (script_to_run1)
                    #os.system(command1)
                    to_delete = os.path.join(results_address3,"pipeline_output","all_outputs")
                    #if os.path.exists(to_delete): 
                    #   shutil.rmtree(to_delete)
                    # TRACTOGRAPHY now
                    results_address_final = os.path.join(results_address3, f"Tractography_seedcutoff_{tr_seed_cutoff_val}_cutoff_{tr_cutoff_val}_minlength_{tr_minlength_val}_maxlength_{tr_maxlength_val}_select_{tr_select_val}_seeds_{tr_seeds_val}_step_{tr_step_val}_basedon_EvgLNRPipe_on_{dataset}")
                    if not os.path.exists(results_address_final):
                           os.makedirs(results_address_final)
                    script_to_run = f"python {tractography_script} -a {results_address_final} -b {DWI_NII[0]} -c {DWI_BVAL[0]} -d {DWI_BVEC[0]} -e {tr_seed_cutoff_val} -f {tr_cutoff_val} -g {tr_minlength_val} -i {tr_maxlength_val} -j {tr_select_val} -k {tr_seeds_val} -l {tr_step_val} "
                    DWI_NII = glob.glob(os.path.join(results_address3, "pipeline_output","DATASINK","DWISpace", "selectedDWI.nii.gz"), recursive=True) + glob.glob(os.path.join(results_address3, "pipeline_output","DATASINK","DWISpace", "*_eddy.nii.gz"), recursive=True)
                    if len(DWI_NII) > 1:
                       DWI_NII = [sorted(DWI_NII)[-1]]
                    if "eddy" in strategy:
                       DWI_BVEC = glob.glob(os.path.join(results_address3, "pipeline_output","DATASINK","DWISpace", "*.bvec"), recursive=True)
                       DWI_BVAL = glob.glob(os.path.join(results_address3, "pipeline_output","DATASINK","DWISpace", "*.bval"), recursive=True)
                    if len(DWI_NII) == 1 and len(DWI_BVAL) == 1 and len(DWI_BVEC) == 1:
                     address_for_annotations = os.path.join(address_for_annotations_init,f"EvgLNRPipe_on_{dataset}",dwi,sub_id)
                     script_to_run = f"python {tractography_script} -a {results_address_final} -b {DWI_NII[0]} -c {DWI_BVAL[0]} -d {DWI_BVEC[0]} -e {tr_seed_cutoff_val} -f {tr_cutoff_val} -g {tr_minlength_val} -i {tr_maxlength_val} -j {tr_select_val} -k {tr_seeds_val} -l {tr_step_val} "
                     SEEDS_init,SEEDS = [],[]
                     for nerve_root in nerve_roots:
                      for side in sides:   
                       foundS = False
                       foundF = False
                       for point in points:
                        nerve_root_to_search = "%s_%s_%s"%(nerve_root,side,point)
                        seed = glob.glob(os.path.join(address_for_annotations, '**', f'*{nerve_root_to_search}.nii*'), recursive=True)
                        if len(seed) == 1:
                            seed_nib = nibabel.load(seed[0])
                            seed_data = seed_nib.get_fdata()
                            has_one = np.any(seed_data == 1)
                            if has_one: 
                                SEEDS_init.append(seed[0])
                                if point == "S":
                                    foundS = True
                                else:
                                    foundF = True
                       if foundS and foundF:
                          SEEDS.append(SEEDS_init[-2])  
                          SEEDS.append(SEEDS_init[-1])  
                     if len(SEEDS) > 0:
                        script_to_run += f"-m \"{' '.join(SEEDS)} \""
                        command = (script_to_run)
                        os.system(command)

                     #to_delete = os.path.join(results_address_final,"pipeline_output","all_outputs")
                     #if os.path.exists(to_delete): 
                     #   shutil.rmtree(to_delete)
            with open(results_txt, 'w') as file:
                file.write("Pipeline execution completed successfully.")

         except subprocess.CalledProcessError as e:
            print("An error occurred while running the pipeline:")
            print(getattr(e, 'stderr', 'No stderr results_dir1'))
