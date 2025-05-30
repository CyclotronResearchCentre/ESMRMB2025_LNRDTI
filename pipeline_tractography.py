# Written by
# Evgenios Kornaropoulos- ekornaropoulos@uliege.be

# IMPORT BASIC PYTHON INTERFACES
import sys, os, getopt, getpass, datetime, nibabel
import numpy as np
## IMPORT SPECIFIC NIPYPE INTERFACES
import nipype.pipeline.engine as pe
from nipype.interfaces import io, fsl, ants, utility
from nipype import IdentityInterface
from nipype.algorithms import metrics, misc
from nipype.interfaces import io, fsl, dipy, ants, mrtrix3
from nipype.interfaces.utility import IdentityInterface, Merge, Rename

# HELPER INTERFACES
# appends path where "pipeline.py" is stored, keep helper interfaces in there
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
dirAtlases = os.path.dirname(os.path.realpath(__file__))

import helperPipeline as HelperInterfaces
import platform
from PrintColours import colours


from nipype import config
#config.enable_debug_mode()


def BuildPipeLine(arguments):
    """ This will build and run the pipeline for anatomical scans
    """
    # parse arguments
    try:
        opts, args = getopt.getopt(sys.argv[1:],'-ha:b:c:d:e:f:g:i:j:k:l:m:', ['help', 'outDir=','DWI_NII=', 'DWI_BVAL=', 'DWI_BVEC=','tr_seed_cutoff_val=','tr_cutoff_val=','tr_minlength_val=','tr_maxlength_val=','tr_select_val=','tr_seeds_val=','tr_step_val=','seeds='])
    except getopt.GetoptError as err:
        print(err)
        print('usage: pipeline.py',
            '\n -a </full/path/to/output/directory>', # all pipeline output will be stored in that folder in the 'nipype' subfolder
            '\n -b </full/path/to/DWI.nii.gz>',
            '\n -c </full/path/to/DWI.bval>',
            '\n -d </full/path/to/DWI.bvec>',
            '\n -e <seeds_cutoff_value>',
            '\n -f <cutoff_value>',
            '\n -g <track_min_length_value>',
            '\n -i <track_max_length_value>',
            '\n -j <seeds_select_value>',
            '\n -k <seeds_number_value>',
            '\n -l <step_value>',
            '\n -m "list(</full/path/to/nerve/root/seeds.nii.gz>"') # if pipeline runs locally you need to use "" to make the argument a list
        sys.exit(2)

    if len(opts)==0:
        sys.exit(colours.red + 'No arguments given. Use -h for more information' + colours.ENDC)

    # pre-assign some variables as 'None' in case particular files are not available
    L3RSfile, L3LSfile, L4RSfile, L4LSfile, L5RSfile, L5LSfile, S1RSfile, S1LSfile = None,None,None,None,None,None,None,None
    L3RFfile, L3LFfile, L4RFfile, L4LFfile, L5RFfile, L5LFfile, S1RFfile, S1LFfile = None,None,None,None,None,None,None,None

    print(len(opts))
    for opt, arg in opts:
        if opt in ('-h','--help') or len(opts)<12:
            print('usage: pipeline.py',
                '\n -a </full/path/to/output/directory>', # all pipeline output will be stored in that folder in the 'nipype' subfolder
                '\n -b </full/path/to/DWI.nii.gz>',
                '\n -c </full/path/to/DWI.bval>',
                '\n -d </full/path/to/DWI.bvec>',
                '\n -e <seeds_cutoff_value>',
                '\n -f <cutoff_value>',
                '\n -g <track_min_length_value>',
                '\n -i <track_max_length_value>',
                '\n -j <seeds_select_value>',
                '\n -k <seeds_number_value>',
                '\n -l <step_value>',
                '\n -m "list(</full/path/to/nerve/root/seeds.nii.gz>"') # if pipeline runs locally you need to use "" to make the argument a list
            sys.exit(2)

        elif opt in ('-a','--outDir'):
            outDir = arg
        elif opt in ('-b','--DWI_NII'):
            DWINIIfile = arg    
        elif opt in ('-c','--DWI_BVAL'):
            DWIBVALfile = arg    
        elif opt in ('-d','--DWI_BVEC'):
            DWIBVECfile = arg  
        elif opt in ('-e','--tr_seed_cutoff_val'):
             tr_seed_cutoff_val = arg   
        elif opt in ('-f','--tr_cutoff_val'):
             tr_cutoff_val = arg   
        elif opt in ('-g','--tr_minlength_val'):
             tr_minlength_val = arg   
        elif opt in ('-i','--tr_maxlength_val'):
             tr_maxlength_val = arg   
        elif opt in ('-j','--tr_select_val'):
             tr_select_val = arg   
        elif opt in ('-k','--tr_seeds_val'):
             tr_seeds_val = arg  
        elif opt in ('-l','--tr_step_val'):
             tr_step_val = arg   
        elif opt in ('-m','--seeds'):
            Seedfiles = arg       
        else:
            assert False, "unhandled option"

    # mullti shread processing for HPHI (make sure it corresponds to slurm file)
    nbThreads = 4

    #-----------------------------------------------------------------------------------------------------#
    # INPUT SOURCE NODES
    #-----------------------------------------------------------------------------------------------------#
    print(colours.green + "Build Pipeline." + colours.ENDC)
    print(colours.green + "Create Source Node." + colours.ENDC)

    seedList = Seedfiles.split()
    seedListNew = []
    for ss,sss in enumerate(seedList):
            if sss != "None":
               seedListNew.append(sss)
#        seqListlower = Seqfiles.lower().split()
#        seqListlowerNew = []
#        for ss,sss in enumerate(seqListlower):
#            if sss != "none":
#               seqListlowerNew.append(sss)
    seed_match = [0]*len(seedListNew)

    # go through the file names and find channels as defined below
    seedNames, seedFiles = [], []
    for seed_name in ['L3_R_S', 'L3_L_S', 'L4_R_S', 'L4_L_S','L5_R_S', 'L5_L_S', 'S1_R_S', 'S1_L_S','L3_R_F', 'L3_L_F', 'L4_R_F', 'L4_L_F','L5_R_F', 'L5_L_F', 'S1_R_F', 'S1_L_F']:
            index = [i for i,f in enumerate(seedListNew) if seed_name in f.split("/")[-1]]
            # if only one match was found assign to corresponding variable

            if len(index)==1:
                index = index[0]
                if 'L3_R_S' in seed_name:
                    L3RSfile = seedListNew[index] 
                    L3RSIndex = index
                elif 'L3_L_S' in seed_name:
                    L3LSfile = seedListNew[index]
                    L3LSIndex = index
                elif 'L4_R_S' in seed_name:
                    L4RSfile = seedListNew[index]
                    L4RSIndex = index
                elif 'L4_L_S' in seed_name:
                    L4LSfile = seedListNew[index]
                    L4LSIndex = index
                elif 'L5_R_S' in seed_name:
                    L5RSfile = seedListNew[index]
                    L5RSIndex = index
                elif 'L5_L_S' in seed_name:
                    L5LSfile = seedListNew[index]
                    L5LSIndex = index
                elif 'S1_R_S' in seed_name:
                    S1RSfile = seedListNew[index]
                    S1RSIndex = index
                elif 'S1_L_S' in seed_name:
                    S1LSfile = seedListNew[index]
                    S1LSIndex = index
                elif 'L3_R_F' in seed_name:
                    L3RFfile = seedListNew[index] 
                    L3RFIndex = index
                elif 'L3_L_F' in seed_name:
                    L3LFfile = seedListNew[index]
                    L3LFIndex = index
                elif 'L4_R_F' in seed_name:
                    L4RFfile = seedListNew[index]
                    L4RFIndex = index
                elif 'L4_L_F' in seed_name:
                    L4LFfile = seedListNew[index]
                    L4LFIndex = index
                elif 'L5_R_F' in seed_name:
                    L5RFfile = seedListNew[index]
                    L5RFIndex = index
                elif 'L5_L_F' in seed_name:
                    L5LFfile = seedListNew[index]
                    L5LFIndex = index
                elif 'S1_R_F' in seed_name:
                    S1RFfile = seedListNew[index]
                    S1RFIndex = index
                elif 'S1_L_F' in seed_name:
                    S1LFfile = seedListNew[index]
                    S1LFIndex = index

                # add found channel to lists
                seedNames.append(seed_name)
                seedFiles.append(seedListNew[index])
                seed_match[index] = 1

            # if several matches were found, through error message and end programme
            elif len(index)>1:
                print('Ambigous file names for', seed_name, ':')
                for i, ind in enumerate(index):
                    print(str(i+1)+')', seedListNew[ind])
                sys.exit()


    # feed found files to inforsoure
    infoSource = pe.Node(IdentityInterface(fields=['DWI_NII','DWI_BVAL','DWI_BVEC','L3RS','L3LS','L4RS','L4LS','L5RS','L5LS','S1RS','S1LS','L3RF','L3LF','L4RF','L4LF','L5RF','L5LF','S1RF','S1LF']), name='infosource')
    infoSource.inputs.DWI_NII = DWINIIfile
    infoSource.inputs.DWI_BVAL = DWIBVALfile
    infoSource.inputs.DWI_BVEC = DWIBVECfile
    if L3RSfile:
            infoSource.inputs.L3RS = L3RSfile
    if L3LSfile:
            infoSource.inputs.L3LS = L3LSfile
    if L4RSfile:
            infoSource.inputs.L4RS = L4RSfile
    if L4LSfile:
            infoSource.inputs.L4LS = L4LSfile
    if L5RSfile:
            infoSource.inputs.L5RS = L5RSfile
    if L5LSfile:
            infoSource.inputs.L5LS = L5LSfile    
    if S1RSfile:
            infoSource.inputs.S1RS = S1RSfile
    if S1LSfile:
           infoSource.inputs.S1LS = S1LSfile    
    if L3RFfile:
            infoSource.inputs.L3RF = L3RFfile
    if L3LFfile:
            infoSource.inputs.L3LF = L3LFfile
    if L4RFfile:
            infoSource.inputs.L4RF = L4RFfile
    if L4LFfile:
            infoSource.inputs.L4LF = L4LFfile
    if L5RFfile:
            infoSource.inputs.L5RF = L5RFfile
    if L5LFfile:
            infoSource.inputs.L5LF = L5LFfile    
    if S1RFfile:
            infoSource.inputs.S1RF = S1RFfile
    if S1LFfile:
           infoSource.inputs.S1LF = S1LFfile    

    # creates a text find that lists the input files for later reference
    # if file already exists, file is deleted and recreated
    txtFile = os.path.join(outDir,'inputFiles_for_pipeline.txt')
    if os.path.isfile(txtFile):
        print('Old <inputFiles_for_pipeline.txt> removed.')
        os.remove(txtFile)

    print('-------------------')
    print(colours.green + 'Input Files' + colours.ENDC)
    for tag, image in zip(['L3RS seed image', 'L3LS seed image','L4RS seed image', 'L4LS seed image','L5RS seed image', 'L5LS seed image','S1RS seed image', 'S1LS seed image','L3RF seed image', 'L3LF seed image','L4RF seed image', 'L4LF seed image','L5RF seed image', 'L5LF seed image','S1RF seed image', 'S1LF seed image','DWI image','DWI bvals','DWI bvecs'], [L3RSfile, L3LSfile,L4RSfile, L4LSfile, L5RSfile, L5LSfile, S1RSfile, S1LSfile, L3RFfile, L3LFfile,L4RFfile, L4LFfile, L5RFfile, L5LFfile, S1RFfile, S1LFfile, DWINIIfile, DWIBVALfile,DWIBVECfile]):
        #output= ' | '.join([tag, str(image)]) + '\n'
        if image==None or image=="None":
            print('{0:20} | '.format(tag), colours.red + str(image) + colours.ENDC)
            output = '{0:20} | '.format(tag) + colours.red + str(image) + colours.ENDC  + '\n'

        else:
            print('{0:20} | '.format(tag), image)
            output = '{0:20} | '.format(tag) + str(image) + '\n'


    output = '{0:8} | '.format('outDir') + outDir + '\n-------------------\n'
    print('{0:20} | '.format('outDir'), outDir)
    print('-------------------')

    #-----------------------------------------------------------------------------------------------------#
    # DATA SINK & WORKFLOW
    #-----------------------------------------------------------------------------------------------------#
    # initialise workflow
    pipeline_name = 'pipeline_output'
    print(colours.green + "Create Workflow." + colours.ENDC)
    preproc = pe.Workflow(name='all_outputs')
    preproc.base_dir = os.path.join(outDir, pipeline_name)
    preproc.config['execution'] = {'remove_unnecessary_outputs' : 'true', 'keep_inputs' : 'true', 'stop_on_first_crash' : 'true'}

    # create sink node where all output is stored in specifc order
    print(colours.green + "Create Sink Node." + colours.ENDC)
    dataSink = pe.Node(io.DataSink(parameterization=False), name='datasink')
    dataSink.inputs.base_directory = os.path.join(outDir, pipeline_name)
    dataSink.inputs.container = 'DATASINK'

    #-----------------------------------------------------------------------------------------------------#
    # PROCESSING & HELPER NODES
    #-----------------------------------------------------------------------------------------------------#
    print(colours.green + "Create Processing Nodes." + colours.ENDC)

    # FOD calculation
    method_now = "FOD"
    fod = pe.Node(HelperInterfaces.FOD(), name = '%s'%method_now)
    fod.inputs.response_method = 'tournier'
    fod.inputs.fod_method = 'csd'
    fod.inputs.out_dir = "%s/all_outputs/%s" % (dataSink.inputs.base_directory,method_now) 

    # tractography - step 1
    method_now = 'tractography_at_L3R'
    tractoL3R = pe.Node(HelperInterfaces.Tractography(), name='%s'%method_now)
    tractoL3R.inputs.seeds = int(tr_seeds_val)
    tractoL3R.inputs.algorithm = 'iFOD2'
    tractoL3R.inputs.select = int(tr_select_val)
    tractoL3R.inputs.seed_cutoff = float(tr_seed_cutoff_val)
    tractoL3R.inputs.cutoff = float(tr_cutoff_val)
    tractoL3R.inputs.minlength = float(tr_minlength_val)
    tractoL3R.inputs.maxlength = float(tr_maxlength_val)
    tractoL3R.inputs.step = float(tr_step_val)
    tractoL3R.inputs.out_dir = "%s/all_outputs/%s" % (dataSink.inputs.base_directory,method_now) 
    method_now = 'tractography_at_L3L'
    tractoL3L = pe.Node(HelperInterfaces.Tractography(), name='%s'%method_now)
    tractoL3L.inputs.seeds = int(tr_seeds_val)
    tractoL3L.inputs.algorithm = 'iFOD2'
    tractoL3L.inputs.select = int(tr_select_val)
    tractoL3L.inputs.seed_cutoff = float(tr_seed_cutoff_val)
    tractoL3L.inputs.cutoff = float(tr_cutoff_val)
    tractoL3L.inputs.minlength = float(tr_minlength_val)
    tractoL3L.inputs.maxlength = float(tr_maxlength_val)
    tractoL3L.inputs.step = float(tr_step_val)
    tractoL3L.inputs.out_dir = "%s/all_outputs/%s" % (dataSink.inputs.base_directory,method_now) 
    method_now = 'tractography_at_L4R'
    tractoL4R = pe.Node(HelperInterfaces.Tractography(), name='%s'%method_now)
    tractoL4R.inputs.seeds = int(tr_seeds_val)
    tractoL4R.inputs.algorithm = 'iFOD2'
    tractoL4R.inputs.select = int(tr_select_val)
    tractoL4R.inputs.seed_cutoff = float(tr_seed_cutoff_val)
    tractoL4R.inputs.cutoff = float(tr_cutoff_val)
    tractoL4R.inputs.step = float(tr_step_val)
    tractoL4R.inputs.minlength = float(tr_minlength_val)
    tractoL4R.inputs.maxlength = float(tr_maxlength_val)
    tractoL4R.inputs.out_dir = "%s/all_outputs/%s" % (dataSink.inputs.base_directory,method_now) 
    method_now = 'tractography_at_L4L'
    tractoL4L = pe.Node(HelperInterfaces.Tractography(), name='%s'%method_now)
    tractoL4L.inputs.seeds = int(tr_seeds_val)
    tractoL4L.inputs.algorithm = 'iFOD2'
    tractoL4L.inputs.select = int(tr_select_val)
    tractoL4L.inputs.seed_cutoff = float(tr_seed_cutoff_val)
    tractoL4L.inputs.cutoff = float(tr_cutoff_val)
    tractoL4L.inputs.step = float(tr_step_val)
    tractoL4L.inputs.minlength = float(tr_minlength_val)
    tractoL4L.inputs.maxlength = float(tr_maxlength_val)
    tractoL4L.inputs.out_dir = "%s/all_outputs/%s" % (dataSink.inputs.base_directory,method_now) 
    method_now = 'tractography_at_L5R'
    tractoL5R = pe.Node(HelperInterfaces.Tractography(), name='%s'%method_now)
    tractoL5R.inputs.seeds = int(tr_seeds_val)
    tractoL5R.inputs.algorithm = 'iFOD2'
    tractoL5R.inputs.select = int(tr_select_val)
    tractoL5R.inputs.seed_cutoff = float(tr_seed_cutoff_val)
    tractoL5R.inputs.cutoff = float(tr_cutoff_val)
    tractoL5R.inputs.step = float(tr_step_val)
    tractoL5R.inputs.minlength = float(tr_minlength_val)
    tractoL5R.inputs.maxlength = float(tr_maxlength_val)
    tractoL5R.inputs.out_dir = "%s/all_outputs/%s" % (dataSink.inputs.base_directory,method_now) 
    method_now = 'tractography_at_L5L'
    tractoL5L = pe.Node(HelperInterfaces.Tractography(), name='%s'%method_now)
    tractoL5L.inputs.seeds = int(tr_seeds_val)
    tractoL5L.inputs.algorithm = 'iFOD2'
    tractoL5L.inputs.select = int(tr_select_val)
    tractoL5L.inputs.seed_cutoff = float(tr_seed_cutoff_val)
    tractoL5L.inputs.cutoff = float(tr_cutoff_val)
    tractoL5L.inputs.step = float(tr_step_val)
    tractoL5L.inputs.minlength = float(tr_minlength_val)
    tractoL5L.inputs.maxlength = float(tr_maxlength_val)
    tractoL5L.inputs.out_dir = "%s/all_outputs/%s" % (dataSink.inputs.base_directory,method_now) 
    method_now = 'tractography_at_S1R'
    tractoS1R = pe.Node(HelperInterfaces.Tractography(), name='%s'%method_now)
    tractoS1R.inputs.seeds = int(tr_seeds_val)
    tractoS1R.inputs.algorithm = 'iFOD2'
    tractoS1R.inputs.select = int(tr_select_val)
    tractoS1R.inputs.seed_cutoff = float(tr_seed_cutoff_val)
    tractoS1R.inputs.cutoff = float(tr_cutoff_val)
    tractoS1R.inputs.step = float(tr_step_val)
    tractoS1R.inputs.minlength = float(tr_minlength_val)
    tractoS1R.inputs.maxlength = float(tr_maxlength_val)
    tractoS1R.inputs.out_dir = "%s/all_outputs/%s" % (dataSink.inputs.base_directory,method_now) 
    method_now = 'tractography_at_S1L'
    tractoS1L = pe.Node(HelperInterfaces.Tractography(), name='%s'%method_now)
    tractoS1L.inputs.seeds = int(tr_seeds_val)
    tractoS1L.inputs.algorithm = 'iFOD2'
    tractoS1L.inputs.select = int(tr_select_val)
    tractoS1L.inputs.seed_cutoff = float(tr_seed_cutoff_val)
    tractoS1L.inputs.cutoff = float(tr_cutoff_val)
    tractoS1L.inputs.step = float(tr_step_val)
    tractoS1L.inputs.minlength = float(tr_minlength_val)
    tractoS1L.inputs.maxlength = float(tr_maxlength_val)
    tractoS1L.inputs.out_dir = "%s/all_outputs/%s" % (dataSink.inputs.base_directory,method_now) 

    # tractography - step 2
    method_now = 'tractedit_at_L3R'
    tractoL3Rf = pe.Node(HelperInterfaces.Tractedit(), name='%s'%method_now)
    tractoL3Rf.inputs.out_dir = "%s/all_outputs/%s" % (dataSink.inputs.base_directory,method_now) 
    method_now = 'tractedit_at_L3L'
    tractoL3Lf = pe.Node(HelperInterfaces.Tractedit(), name='%s'%method_now)
    tractoL3Lf.inputs.out_dir = "%s/all_outputs/%s" % (dataSink.inputs.base_directory,method_now) 
    method_now = 'tractedit_at_L4R'
    tractoL4Rf = pe.Node(HelperInterfaces.Tractedit(), name='%s'%method_now)
    tractoL4Rf.inputs.out_dir = "%s/all_outputs/%s" % (dataSink.inputs.base_directory,method_now) 
    method_now = 'tractedit_at_L4L'
    tractoL4Lf = pe.Node(HelperInterfaces.Tractedit(), name='%s'%method_now)
    tractoL4Lf.inputs.out_dir = "%s/all_outputs/%s" % (dataSink.inputs.base_directory,method_now) 
    method_now = 'tractedit_at_L5R'
    tractoL5Rf = pe.Node(HelperInterfaces.Tractedit(), name='%s'%method_now)
    tractoL5Rf.inputs.out_dir = "%s/all_outputs/%s" % (dataSink.inputs.base_directory,method_now) 
    method_now = 'tractedit_at_L5L'
    tractoL5Lf = pe.Node(HelperInterfaces.Tractedit(), name='%s'%method_now)
    tractoL5Lf.inputs.out_dir = "%s/all_outputs/%s" % (dataSink.inputs.base_directory,method_now) 
    method_now = 'tractedit_at_S1R'
    tractoS1Rf = pe.Node(HelperInterfaces.Tractedit(), name='%s'%method_now)
    tractoS1Rf.inputs.out_dir = "%s/all_outputs/%s" % (dataSink.inputs.base_directory,method_now) 
    method_now = 'tractedit_at_S1L'
    tractoS1Lf = pe.Node(HelperInterfaces.Tractedit(), name='%s'%method_now)
    tractoS1Lf.inputs.out_dir = "%s/all_outputs/%s" % (dataSink.inputs.base_directory,method_now) 
     
    #-----------------------------------------------------------------------------------------------------#
    # CONNECT INPUT AND OUTPUT NODES
    #-----------------------------------------------------------------------------------------------------#
    print(colours.green + "Connect Nodes." + colours.ENDC)
    preproc.connect([
    # compute FOD
       (infoSource, fod, [('DWI_NII', 'dwi_image')]),
       (infoSource, fod, [('DWI_BVEC', 'bvec')]),
       (infoSource, fod, [('DWI_BVAL', 'bval')]),
       (fod, dataSink, [('out_fod', 'DWISpace.@FOD')]),
       (fod, dataSink, [('out_response', 'DWISpace.@response')]),
     ])
    if L3RSfile:
     preproc.connect([
        # compute tractography  
        (fod, tractoL3R, [('out_fod', 'dwi_image')]),
        (infoSource, tractoL3R, [('DWI_BVEC', 'bvec')]),
        (infoSource, tractoL3R, [('DWI_BVAL', 'bval')]),
        (infoSource, tractoL3R, [('L3RS', 'seed_image')]),
        #(tractoL3R, tractoL3Rf, [('out_tracto', 'init_out_image')]),
        #(infoSource, tractoL3Rf, [('L3RS', 'seed_image')]),
        (tractoL3R, dataSink, [('out_tracto', 'DWISpace.@tractography_at_L3R')]),
        #(tractoL3Rf, dataSink, [('out_tracto', 'DWISpace.@tractography_at_L3R_final')]),
      ]) 
     if L3RFfile:
      preproc.connect([ 
        (infoSource, tractoL3R, [('L3RF', 'include_image')]),
        #(infoSource, tractoL3Rf, [('L3RF', 'end_image')]),
      ])

    if L3LSfile:
     preproc.connect([
        # compute tractography  
        (fod, tractoL3L, [('out_fod', 'dwi_image')]),
        (infoSource, tractoL3L, [('DWI_BVEC', 'bvec')]),
        (infoSource, tractoL3L, [('DWI_BVAL', 'bval')]),
        (infoSource, tractoL3L, [('L3LS', 'seed_image')]),
        #(tractoL3L, tractoL3Lf, [('out_tracto', 'init_out_image')]),
        #(infoSource, tractoL3Lf, [('L3LS', 'seed_image')]),
        (tractoL3L, dataSink, [('out_tracto', 'DWISpace.@tractography_at_L3L')]),
        #(tractoL3Lf, dataSink, [('out_tracto', 'DWISpace.@tractography_at_L3L_final')]),
      ]) 
     if L3LFfile:
      preproc.connect([ 
        (infoSource, tractoL3L, [('L3LF', 'include_image')]),
        #(infoSource, tractoL3Lf, [('L3LF', 'end_image')]),
      ])

    if L4RSfile:
     preproc.connect([
        # compute tractography  
        (fod, tractoL4R, [('out_fod', 'dwi_image')]),
        (infoSource, tractoL4R, [('DWI_BVEC', 'bvec')]),
        (infoSource, tractoL4R, [('DWI_BVAL', 'bval')]),
        (infoSource, tractoL4R, [('L4RS', 'seed_image')]),
        #(tractoL4R, tractoL4Rf, [('out_tracto', 'init_out_image')]),
        #(infoSource, tractoL4Rf, [('L4RS', 'seed_image')]),
        (tractoL4R, dataSink, [('out_tracto', 'DWISpace.@tractography_at_L4R')]),
        #(tractoL4Rf, dataSink, [('out_tracto', 'DWISpace.@tractography_at_L4R_final')]),

      ])

     if L4RFfile:
      preproc.connect([ 
        (infoSource, tractoL4R, [('L4RF', 'include_image')]),
        #(infoSource, tractoL4Rf, [('L4RF', 'end_image')]),
      ])

    if L4LSfile:
     preproc.connect([
        # compute tractography  
        (fod, tractoL4L, [('out_fod', 'dwi_image')]),
        (infoSource, tractoL4L, [('DWI_BVEC', 'bvec')]),
        (infoSource, tractoL4L, [('DWI_BVAL', 'bval')]),
        (infoSource, tractoL4L, [('L4LS', 'seed_image')]),
        #(tractoL4L, tractoL4Lf, [('out_tracto', 'init_out_image')]),
        #(infoSource, tractoL4Lf, [('L4LS', 'seed_image')]),
        (tractoL4L, dataSink, [('out_tracto', 'DWISpace.@tractography_at_L4L')]),
        #(tractoL4Lf, dataSink, [('out_tracto', 'DWISpace.@tractography_at_L4L_final')]),

      ])
     if L4LFfile:
      preproc.connect([ 
        (infoSource, tractoL4L, [('L4LF', 'include_image')]),
        #(infoSource, tractoL4Lf, [('L4LF', 'end_image')]),
      ])

    if L5RSfile:
     preproc.connect([
        # compute tractography  
        (fod, tractoL5R, [('out_fod', 'dwi_image')]),
        (infoSource, tractoL5R, [('DWI_BVEC', 'bvec')]),
        (infoSource, tractoL5R, [('DWI_BVAL', 'bval')]),
        (infoSource, tractoL5R, [('L5RS', 'seed_image')]),
        #(tractoL5R, tractoL5Rf, [('out_tracto', 'init_out_image')]),
        #(infoSource, tractoL5Rf, [('L5RS', 'seed_image')]),
        (tractoL5R, dataSink, [('out_tracto', 'DWISpace.@tractography_at_L5R')]),
        #(tractoL5Rf, dataSink, [('out_tracto', 'DWISpace.@tractography_at_L5R_final')]),

      ])

     if L5RFfile:
      preproc.connect([ 
        (infoSource, tractoL5R, [('L5RF', 'include_image')]),
        #(infoSource, tractoL5Rf, [('L5RF', 'end_image')]),
      ])

    if L5LSfile:
     preproc.connect([
        # compute tractography  
        (fod, tractoL5L, [('out_fod', 'dwi_image')]),
        (infoSource, tractoL5L, [('DWI_BVEC', 'bvec')]),
        (infoSource, tractoL5L, [('DWI_BVAL', 'bval')]),
        (infoSource, tractoL5L, [('L5LS', 'seed_image')]),
        #(tractoL5L, tractoL5Lf, [('out_tracto', 'init_out_image')]),
        #(infoSource, tractoL5Lf, [('L5LS', 'seed_image')]),
        (tractoL5L, dataSink, [('out_tracto', 'DWISpace.@tractography_at_L5L')]),
        #(tractoL5Lf, dataSink, [('out_tracto', 'DWISpace.@tractography_at_L5L_final')]),

      ])
     if L5LFfile:
      preproc.connect([ 
        (infoSource, tractoL5L, [('L5LF', 'include_image')]),
        #(infoSource, tractoL5Lf, [('L5LF', 'end_image')]),
      ])

    if S1RSfile:
     preproc.connect([
        # compute tractography  
        (fod, tractoS1R, [('out_fod', 'dwi_image')]),
        (infoSource, tractoS1R, [('DWI_BVEC', 'bvec')]),
        (infoSource, tractoS1R, [('DWI_BVAL', 'bval')]),
        (infoSource, tractoS1R, [('S1RS', 'seed_image')]),
        #(tractoS1R, tractoS1Rf, [('out_tracto', 'init_out_image')]),
        #(infoSource, tractoS1Rf, [('S1RS', 'seed_image')]),
        (tractoS1R, dataSink, [('out_tracto', 'DWISpace.@tractography_at_S1R')]),
        #(tractoS1Rf, dataSink, [('out_tracto', 'DWISpace.@tractography_at_S1R_final')]),

      ])

     if S1RFfile:
      preproc.connect([ 
        (infoSource, tractoS1R, [('S1RF', 'include_image')]),
        #(infoSource, tractoS1Rf, [('S1RF', 'end_image')]),
      ])
     
    if S1LSfile:
     preproc.connect([
        # compute tractography  
        (fod, tractoS1L, [('out_fod', 'dwi_image')]),
        (infoSource, tractoS1L, [('DWI_BVEC', 'bvec')]),
        (infoSource, tractoS1L, [('DWI_BVAL', 'bval')]), 
        (infoSource, tractoS1L, [('S1LS', 'seed_image')]),
        #(tractoS1L, tractoS1Lf, [('out_tracto', 'init_out_image')]),
        #(infoSource, tractoS1Lf, [('S1LS', 'seed_image')]),
        (tractoS1L, dataSink, [('out_tracto', 'DWISpace.@tractography_at_S1L')]),
        #(tractoS1Lf, dataSink, [('out_tracto', 'DWISpace.@tractography_at_S1L_final')]),

      ])

     if S1LFfile:
      preproc.connect([ 
        (infoSource, tractoS1L, [('S1LF', 'include_image')]),
        #(infoSource, tractoS1Lf, [('S1LF', 'end_image')]),
      ])
    return preproc


if __name__=='__main__':
    pipeLine = BuildPipeLine(sys.argv[1:])
    pipeLine.write_graph()
    print(colours.green + "Run Pipeline..." + colours.ENDC)
    pipeLine.run(plugin='MultiProc')
    print(colours.green + "Pipeline completed." + colours.ENDC)

