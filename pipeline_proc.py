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
from dipy.io import read_bvals_bvecs

from nipype import config
#config.enable_debug_mode()

def _check_nb_bvals(fbvals, fbvecs):
        """ Takes bval and bvec file and computes the number of bvals,
                unique bvals and the number of unique bvals.
                fbvals: string to bval file
                fbvecs: string to bvec file
        """
        bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)
        nbBvals = len(bvals)
        uniqueBvals = np.unique(np.round(bvals/100)*100)
        if uniqueBvals[0]==0: uniqueBvals = uniqueBvals[1:]
        nbUniqueBvals = len(uniqueBvals)

        return uniqueBvals, nbUniqueBvals, nbBvals

def BuildPipeLine(arguments):
    """ This will build and run the pipeline for anatomical scans
    """
    # parse arguments
    try:
        opts, args = getopt.getopt(sys.argv[1:],'-ha:b:c:d:e:f:g:i:k:l:m:', ['help', 'outDir=','T2W=','DWI_NII=', 'DWI_BVAL=', 'DWI_BVEC=', 'DWI_JSON=', 'DWI_REV_NII=','DWI_REV_BVAL=', 'DWI_REV_BVEC=','strategy=','seqs='])
    except getopt.GetoptError as err:
        print(err)
        print('usage: pipeline.py',
            '\n -a </full/path/to/output/directory>', # all pipeline output will be stored in that folder in the 'nipype' subfolder
            '\n -b </full/path/to/T2w.nii.gz>',
            '\n -c </full/path/to/DWI.nii.gz>',
            '\n -d </full/path/to/DWI.bval>',
            '\n -e </full/path/to/DWI.bvec>',
            '\n -f </full/path/to/DWI.json>',
            '\n -g </full/path/to/DWI_REV.nii.gz>',
            '\n -i </full/path/to/DWI_REV.bval>',
            '\n -k </full/path/to/DWI_REV.bvec>',
            '\n -l <strategy>',
            '\n -m "list(</full/path/to/additional/anatomical/scans.nii.gz>"') # if pipeline runs locally you need to use "" to make the argument a list
        sys.exit(2)

    if len(opts)==0:
        sys.exit(colours.red + 'No arguments given. Use -h for more information' + colours.ENDC)

    # pre-assign some variables as 'None' in case particular files are not available
    Seqfiles = None
    T1file = None
    MTfile = None
    ihMTfile = None
    DWIJSONfile = None
    DWIREVNIIfile = None
    DWIREVBVALfile = None
    DWIREVBVECfile = None

    print(len(opts))
    for opt, arg in opts:
        if opt in ('-h','--help') or len(opts)<6:
            print('usage: pipeline.py',
                '\n -a </full/path/to/output/directory>', # all pipeline output will be stored in that folder in the 'nipype' subfolder
                '\n -b </full/path/to/T2w.nii.gz>',
                '\n -c </full/path/to/DWI.nii.gz>',
                '\n -d </full/path/to/DWI.bval>',
                '\n -e </full/path/to/DWI.bvec>',
                '\n -f </full/path/to/DWI.json>',
                '\n -g </full/path/to/DWI_REV.nii.gz>',
                '\n -i </full/path/to/DWI_REV.bval>',
                '\n -k </full/path/to/DWI_REV.bvec>',
                '\n -l <strategy>',
                '\n -m "list(</full/path/to/additional/anatomical/scans.nii.gz>"') # if pipeline runs locally you need to use "" to make the argument a list
            sys.exit(2)

        elif opt in ('-a','--outDir'):
            outDir = arg
        elif opt in ('-b','--T2w'):
            T2file = arg        
        elif opt in ('-c','--DWI_NII'):
            DWINIIfile = arg    
        elif opt in ('-d','--DWI_BVAL'):
            DWIBVALfile = arg    
        elif opt in ('-e','--DWI_BVEC'):
            DWIBVECfile = arg   
        elif opt in ('-f','--DWI_JSON'):
            DWIJSONfile = arg    
        elif opt in ('-g','--DWI_REV_NII'):
            DWIREVNIIfile = arg
        elif opt in ('-i','--DWI_REV_BVAL'):
            DWIREVBVALfile = arg    
        elif opt in ('-k','--DWI_REV_BVEC'):
            DWIREVBVECfile = arg       
        elif opt in ('-l','--strategy'):
            strategy = arg   
        elif opt in ('-m','--seqs'):
            Seqfiles = arg    
        else:
            assert False, "unhandled option"

    # mullti shread processing for HPHI (make sure it corresponds to slurm file)
    nbThreads = 4

    #-----------------------------------------------------------------------------------------------------#
    # INPUT SOURCE NODES
    #-----------------------------------------------------------------------------------------------------#
    print(colours.green + "Build Pipeline." + colours.ENDC)
    print(colours.green + "Create Source Node." + colours.ENDC)
    if not Seqfiles:
        infoSource = pe.Node(IdentityInterface(fields=['T1']), name='infosource')
        infoSource.inputs.T1 = T1file
        match = [0]
        channelNames = ['', '']
    else:
        seqList = Seqfiles.split()
        seqListNew = []
        for ss,sss in enumerate(seqList):
            if sss != "None":
               seqListNew.append(sss)
#        seqListlower = Seqfiles.lower().split()
#        seqListlowerNew = []
#        for ss,sss in enumerate(seqListlower):
#            if sss != "none":
#               seqListlowerNew.append(sss)
        match = [0]*len(seqListNew)

        # go through the file names and find channels as defined below
        channelNames, channelFiles = [], []
        for channel in ['T1w', 'MT', 'ihMT']:
            index = [i for i,f in enumerate(seqListNew) if channel in f.split("/")[-1]]
            #index = [i for i,f in enumerate(seqListlower) if channel.lower() in f]
            # if only one match was found assign to corresponding variable
            if len(index)==1:
                index = index[0]
                if channel=='T1w':
                    T1file = seqListNew[index]
                elif channel=='MT':
                    MTfile = seqListNew[index]
                elif channel=='ihMT':
                    ihMTfile = seqListNew[index]

                # add found channel to lists
                channelNames.append(channel)
                channelFiles.append(seqListNew[index])
                match[index] = 1

                # remember the index of specific sequences if available
                # necessary for later processing nodes
                if channel=='T1w':
                    T1Index = index

                if channel=='MT':
                    MTIndex = index

                if channel=='ihMT':
                    ihMTIndex = index

            # if several matches were found, through error message and end programme
            elif len(index)>1:
                print('Ambigous file names for', channel, ':')
                for i, ind in enumerate(index):
                    print(str(i+1)+')', seqListNew[ind])
                sys.exit()

        # if files were found that could not me assigned, through error message and end programme
        if len(seqListNew)!=sum(match):
                   for i, m in enumerate(match):
                     if m==0:
                      if seqListNew[i] != "None":
                        print("ccc %s" % seqListNew[i])
                        print('Input string includes unspecific file(s):')
                        for i, m in enumerate(match):
                            if m==0:
                               print(seqListNew[i])
                        sys.exit()


    # feed found files to inforsoure
    infoSource = pe.Node(IdentityInterface(fields=['T2', 'DWI_NII','DWI_BVAL','DWI_BVEC','DWI_JSON','DWI_REV_NII','DWI_REV_BVAL','DWI_REV_BVEC','strategy','T1', 'MT','ihMT']), name='infosource')
    infoSource.inputs.T2 = T2file
    infoSource.inputs.DWI_NII = DWINIIfile
    infoSource.inputs.DWI_BVAL = DWIBVALfile
    infoSource.inputs.DWI_BVEC = DWIBVECfile
    if DWIREVNIIfile:
       infoSource.inputs.DWI_REV_NII = DWIREVNIIfile
    if DWIREVBVALfile:
       infoSource.inputs.DWI_REV_BVAL = DWIREVBVALfile
    if DWIREVBVECfile:
       infoSource.inputs.DWI_REV_BVEC = DWIREVBVECfile
    infoSource.inputs.strategy = strategy
    if DWIJSONfile:
            infoSource.inputs.DWI_JSON = DWIJSONfile   
    if T1file:
            infoSource.inputs.T1 = T1file            
    if MTfile:
            infoSource.inputs.MT = MTfile    
    if ihMTfile:
            infoSource.inputs.ihMT = ihMTfile

    #-----------------------------------------------------------------------------------------------------#
    # TERMINAL OUTPUT
    #-----------------------------------------------------------------------------------------------------#
    # ASSESS BVAL/BVEC CHARACTERISATION
    uniqueBvals, nbUniqueBvals, nbBvals = _check_nb_bvals(DWIBVALfile, DWIBVECfile)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('In the case of main DWI acquisition:')
    print('Number of Bvals: ', nbBvals)
    print('Number of Unique Bvals: ', nbUniqueBvals)
    print('Unique Bvals: ', uniqueBvals)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~')
#    uniqueBvals_rev, nbUniqueBvals_rev, nbBvals_rev = _check_nb_bvals(DWIREVBVALfile, DWIREVBVECfile)
#    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~')
#    print('In the case of reverse-phase-enconding DWI acquisition:')
#    print('Number of Bvals: ', nbBvals_rev)
#    print('Number of Unique Bvals: ', nbUniqueBvals_rev)
#    print('Unique Bvals: ', uniqueBvals_rev)
#    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    # creates a text find that lists the input files for later reference
    # if file already exists, file is deleted and recreated
    txtFile = os.path.join(outDir,'inputFiles_for_pipeline.txt')
    if os.path.isfile(txtFile):
        print('Old <inputFiles_for_pipeline.txt> removed.')
        os.remove(txtFile)

    print('-------------------')
    print(colours.green + 'Input Files' + colours.ENDC)
    for tag, image in zip(['T2w image','DWI image','DWI bvals','DWI bvecs','DWI reversed image','DWI reversed bvals','DWI reversed bvecs','strategy','T1w image', 'MT image', 'ihMT image'], [T2file, DWINIIfile, DWIBVALfile,DWIBVECfile, DWIREVNIIfile, DWIREVBVALfile,DWIREVBVECfile,strategy,T1file, MTfile, ihMTfile]):
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

    # ASSESS 3rd DIMENSION OF DWI IMAGE
    # topup runs with given paprameters only when dimensions are even
    # if dimesnions are odd, cut off first slice, only do so when extra b0 is provided
    # otherwise topup is not applied anyways, so no cropping will take place
    valueForCutOfAxialSlice_X = 0
    valueForCutOfAxialSlice_Y = 0
    valueForCutOfAxialSlice_Z = 0

    if DWIREVNIIfile!=None:
                print(DWIREVNIIfile)
                print(nibabel.load(DWIREVNIIfile).get_fdata().shape)
                extraB0ImageDimX, extraB0ImageDimY, extraB0ImageDimZ = nibabel.load(DWIREVNIIfile).get_fdata().shape
                if extraB0ImageDimX % 2!=0:
                        valueForCutOfAxialSlice_X = 1
                        print('DWI volume has odd x-dimension: Cut of lowest slice')
                if extraB0ImageDimY % 2!=0:
                        valueForCutOfAxialSlice_Y = 1
                        print('DWI volume has odd y-dimension: Cut of lowest slice')
                if extraB0ImageDimZ % 2!=0: #
                        valueForCutOfAxialSlice_Z = 1
                        print('DWI volume has odd z-dimension: Cut of lowest slice')
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

    # segment the spinal cord
    method_now = 'spinal_cord_segmentation_deepseg'
    spinal_cord_segmentation_deepseg = pe.Node(HelperInterfaces.SpinalCordSegmentationDeepSeg(), name='%s'%method_now)
    spinal_cord_segmentation_deepseg.inputs.seq_type = 't2'
    spinal_cord_segmentation_deepseg.inputs.out_dir = "%s/all_outputs/%s" % (dataSink.inputs.base_directory,method_now)
    # alternative method to segment the spinal cord
    method_now = 'spinal_cord_segmentation_propseg'
    spinal_cord_segmentation_propseg = pe.Node(HelperInterfaces.SpinalCordSegmentationPropSeg(), name='%s'%method_now)
    spinal_cord_segmentation_propseg.inputs.seq_type = 't2'
    spinal_cord_segmentation_propseg.inputs.out_dir = "%s/all_outputs/%s" % (dataSink.inputs.base_directory,method_now)

    # segment the spinal cord
    method_now = 'spinal_cord_segmentation_deepseg_dwi'
    spinal_cord_segmentation_deepseg_dwi = pe.Node(HelperInterfaces.SpinalCordSegmentationDeepSeg(), name='%s'%method_now)
    spinal_cord_segmentation_deepseg_dwi.inputs.seq_type = 't2'
    spinal_cord_segmentation_deepseg_dwi.inputs.out_dir = "%s/all_outputs/%s" % (dataSink.inputs.base_directory,method_now)
    # alternative method to segment the spinal cord
    method_now = 'spinal_cord_segmentation_propseg_dwi'
    spinal_cord_segmentation_propseg_dwi = pe.Node(HelperInterfaces.SpinalCordSegmentationPropSeg(), name='%s'%method_now)
    spinal_cord_segmentation_propseg_dwi.inputs.seq_type = 't2'
    spinal_cord_segmentation_propseg_dwi.inputs.out_dir = "%s/all_outputs/%s" % (dataSink.inputs.base_directory,method_now)

    # fit binarized centerline from SC seg (default settings)
    method_now = 'centerline_binarized_after_deepseg'
    centerline1_bin = pe.Node(HelperInterfaces.Centerline(), name="%s"%method_now)
    centerline1_bin.inputs.method = 'fitseg'
    centerline1_bin.inputs.soft = 0
    centerline1_bin.inputs.out_dir = "%s/all_outputs/%s" % (dataSink.inputs.base_directory,method_now)
    # fit soft centerline from SC seg
    method_now = 'centerline_soft_after_deepseg'
    centerline1_soft = pe.Node(HelperInterfaces.Centerline(), name="%s"%method_now)
    centerline1_soft.inputs.method = 'fitseg'
    centerline1_soft.inputs.soft = 1
    centerline1_soft.inputs.out_dir = "%s/all_outputs/%s" % (dataSink.inputs.base_directory,method_now)
    # fit binarized centerline from SC seg (default settings)
    method_now = 'centerline_binarized_after_propseg'
    centerline2_bin = pe.Node(HelperInterfaces.Centerline(), name="%s"%method_now)
    centerline2_bin.inputs.method = 'fitseg'
    centerline2_bin.inputs.soft = 0
    centerline2_bin.inputs.out_dir = "%s/all_outputs/%s" % (dataSink.inputs.base_directory,method_now)
    # fit soft centerline from SC seg
    method_now = 'centerline_soft_after_propseg'
    centerline2_soft = pe.Node(HelperInterfaces.Centerline(), name="%s"%method_now)
    centerline2_soft.inputs.method = 'fitseg'
    centerline2_soft.inputs.soft = 1
    centerline2_soft.inputs.out_dir = "%s/all_outputs/%s" % (dataSink.inputs.base_directory,method_now)

    # vertebral labeling
    method_now = 'vertebral_labeling'
    vertrebal_labeling = pe.Node(HelperInterfaces.VertrebalLabeling(), name="%s"%method_now)
    vertrebal_labeling.inputs.seq_type = 't2'
    vertrebal_labeling.inputs.out_dir = "%s/all_outputs/%s" % (dataSink.inputs.base_directory,method_now)    
    # create labels at in the cord at C2 and C5 mid-vertebral levels
    method_now = 'mid-vertebral-levels'
    mid_vertrebal_labeling = pe.Node(HelperInterfaces.MidVertrebalLabeling(), name="%s"%method_now)
    mid_vertrebal_labeling.inputs.vert_body = 2.5
    mid_vertrebal_labeling.inputs.out_dir = "%s/all_outputs/%s" % (dataSink.inputs.base_directory,method_now)
   
    # register to template
    method_now = 'register_to_template'
    register_to_template = pe.Node(HelperInterfaces.RegisterToTemplate(), name="%s"%method_now)
    register_to_template.inputs.seq_type = 't2'
    register_to_template.inputs.out_dir = "%s/all_outputs/%s" % (dataSink.inputs.base_directory,method_now)

    # compute cross-sectional area (and other morphometry measures) for each slice 
    method_now = 'cross_sectional_area'
    cross_area = pe.Node(HelperInterfaces.CrossSectionArea(), name="%s"%method_now)
    cross_area.inputs.out_dir = "%s/all_outputs/%s" % (dataSink.inputs.base_directory,method_now)
    
    # compute cross-sectionnal area based on distance from pontomedullary junction (PMJ)
    # detect PMJ
    method_now = 'detect_pmj'
    pmj = pe.Node(HelperInterfaces.DetectPMJ(), name="%s"%method_now)
    pmj.inputs.seq_type = 't2'
    pmj.inputs.out_dir = "%s/all_outputs/%s" % (dataSink.inputs.base_directory,method_now)

    # separate b0 and dwi
    method_now = 'separate_b0_and_dwi'
    b0 = pe.Node(HelperInterfaces.SeparateB0(), name="%s"%method_now)
    b0.inputs.out_dir = "%s/all_outputs/%s" % (dataSink.inputs.base_directory,method_now)

    # bring t2 segmentation in dmri space to create mask (no optimization)
    method_now = 'register_t2w_dwi'
    reg_t2w_dwi = pe.Node(HelperInterfaces.T2wDWIReg(), name="%s"%method_now)
    reg_t2w_dwi.inputs.identity = 1
    reg_t2w_dwi.inputs.interpolation = 'linear'
    reg_t2w_dwi.inputs.out_dir = "%s/all_outputs/%s" % (dataSink.inputs.base_directory,method_now)

    # bring t2 segmentation in dmri space via mtrix3
    method_now = 'register_t2w_dwi_via_mrtrix3'
    reg_t2w_dwi_mrtrix3 = pe.Node(HelperInterfaces.T2wDWIRegMRTRIX3(), name="%s"%method_now)
    reg_t2w_dwi_mrtrix3.inputs.type_chosen = 'rigid'
    reg_t2w_dwi_mrtrix3.inputs.out_dir = "%s/all_outputs/%s" % (dataSink.inputs.base_directory,method_now)

    # bring t2 segmentation in dmri space via ANTs (default)
    method_now = 'register_t2w_dwi_via_ants'
    reg_t2w_dwi_ants = pe.Node(HelperInterfaces.T2wDWIRegANTs(), name="%s"%method_now)
    reg_t2w_dwi_ants.inputs.gradient_step = 0.1
    reg_t2w_dwi_ants.inputs.dimensionality = 3
    reg_t2w_dwi_ants.inputs.transformation = 'b'
    reg_t2w_dwi_ants.inputs.out_dir = "%s/all_outputs/%s" % (dataSink.inputs.base_directory,method_now)

    # create mask to help moco and for faster processing
    method_now = 'mask_dwi'
    mask_dwi = pe.Node(HelperInterfaces.MaskDWI(), name="%s"%method_now)
    mask_dwi.inputs.size = '35mm'
    mask_dwi.inputs.out_dir = "%s/all_outputs/%s" % (dataSink.inputs.base_directory,method_now)

    # crop DWI
    method_now = 'crop_dwi'
    crop_dwi = pe.Node(HelperInterfaces.cropDWI(), name="%s"%method_now)
    crop_dwi.inputs.out_dir = "%s/all_outputs/%s" % (dataSink.inputs.base_directory,method_now)

    # denoise DWI
    method_now = 'mppca'
    mppca = pe.Node(mrtrix3.DWIDenoise(), name="%s"%method_now)

    # gibbs removal in DWI
    method_now = 'gibbs_removal'
    gibbs = pe.Node(HelperInterfaces.MRTRIX3GibbsRemoval(), name="%s"%method_now)

    # head motion & eddy current correction (version 1)
    method_now = 'eddy'
    eddy = pe.Node(HelperInterfaces.DWIFSLPREPROC(), name="%s"%method_now)
    if "topup" in strategy:
        eddy.inputs.rpe = '-rpe_pair'
    else:
        eddy.inputs.rpe = '-rpe_none'
    eddy.inputs.pe_dir = 'AP'
    #eddy.inputs.extra_options = "--slm=linear"
    eddy.inputs.out_dir = "%s/all_outputs/%s" % (dataSink.inputs.base_directory,method_now) 

    # head motion & eddy current correction (version 2)
    #method_now = 'eddy'
    #eddy = pe.Node(fsl.Eddy(), name="%s"%method_now)
    #eddy.inputs.interp = 'spline'
    #eddy.inputs.num_threads = 4
    #eddy.inputs.use_cuda = True
    #eddy.inputs.is_shelled = True
    #eddy.inputs.args = '--ol_nstd=5'
    #eddy.inputs.output_type = 'NIFTI_GZ'

    if DWIREVNIIfile!=None:
                # HELPER NODE TO SELECT DWI or DWI_B0 file
                renameDWI =  pe.Node(Rename(), name='select_dwi')
                renameDWI.inputs.format_string = 'selectedDWI.nii.gz'

                # SELECT 1st VOLUME
                selectLowB = pe.Node(fsl.ExtractROI(), name='select_b0')
                selectLowB.inputs.t_min = nbBvals-1
                selectLowB.inputs.t_size = 1
                selectLowB.inputs.roi_file = 'b0.nii.gz'
                selectLowB.inputs.output_type = 'NIFTI_GZ'

                # STRIP AWAY THE B0 VOLUME AFTER ARTIFACT CORRECTION
                selectCorrectedDWI = pe.Node(fsl.ExtractROI(), name='strip_b0_after_artefact_correction')
                selectCorrectedDWI.inputs.t_min = 0
                selectCorrectedDWI.inputs.t_size = nbBvals
                selectCorrectedDWI.inputs.roi_file = 'corrected_DWI.nii.gz'
                selectCorrectedDWI.inputs.output_type = 'NIFTI_GZ'

                # INTERFACES TO PREPARE FOR AND USE TOPUP
                # if odd dimesions cut of slice=0 from all three planes of DWI image
                cutDWI = pe.Node(fsl.ExtractROI(), name='cut_DWI')
                cutDWI.inputs.x_min = valueForCutOfAxialSlice_X
                cutDWI.inputs.x_size = -1
                cutDWI.inputs.y_min = valueForCutOfAxialSlice_Y
                cutDWI.inputs.y_size = -1
                cutDWI.inputs.z_min = valueForCutOfAxialSlice_Z
                cutDWI.inputs.z_size = -1
                cutDWI.inputs.roi_file = 'cutDWI.nii.gz'
                cutDWI.inputs.output_type = 'NIFTI_GZ'

                # if odd dimesions cut of slice=0 from all three planes of extra b0 image
                cutB0 = pe.Node(fsl.ExtractROI(), name='cut_extraB0')
                cutB0.inputs.x_min = valueForCutOfAxialSlice_X
                cutB0.inputs.x_size = -1
                cutB0.inputs.y_min = valueForCutOfAxialSlice_Y
                cutB0.inputs.y_size = -1
                cutB0.inputs.z_min = valueForCutOfAxialSlice_Z
                cutB0.inputs.z_size = -1
                cutB0.inputs.roi_file = 'cutExtraB0.nii.gz'
                cutB0.inputs.output_type = 'NIFTI_GZ'

                #make list of DWI and extra b0
                listDWI = pe.Node(Merge(2), name='list_DWI')

                #make list of b0 and rebb0
                listB0 = pe.Node(Merge(2), name='list_b0')

                # merge DWI and extra b0 volume for artifact correction (i.e. LPCA & Gibbs)
                mergeOppostitePE = pe.Node(fsl.Merge(), name='merge_opposite_PE')
                mergeOppostitePE.inputs.dimension = 't'
                mergeOppostitePE.inputs.merged_file = 'merged_DWI_extrab0.nii.gz'
                mergeOppostitePE.inputs.output_type = 'NIFTI_GZ'

                # select extra b0 volume after artifact correction
                selectExtraB = pe.Node(fsl.ExtractROI(), name='select_revB0')
                selectExtraB.inputs.t_min = 1
                selectExtraB.inputs.t_size = 1
                selectExtraB.inputs.roi_file = 'revB0.nii.gz'
                selectExtraB.inputs.output_type = 'NIFTI_GZ'

                # merge both corrected b0s with oposite phase encoding directions
                mergeOppostiteB0 = pe.Node(fsl.Merge(), name='merge_opposite_b0')
                mergeOppostiteB0.inputs.dimension = 't'
                mergeOppostiteB0.inputs.merged_file = 'merged_b0_revB0.nii.gz'
                mergeOppostiteB0.inputs.output_type = 'NIFTI_GZ'

                # apply topup
                topup = pe.Node(fsl.TOPUP(), name='topup')
                topup.inputs.output_type = "NIFTI_GZ"

    # apply moco for further motion correction
    method_now = 'moco_on_dwi'
    moco = pe.Node(HelperInterfaces.mocoDWI(), name="%s"%method_now)
    moco.inputs.out_dir = "%s/all_outputs/%s" % (dataSink.inputs.base_directory,method_now) 

    method_now = 'select_corrected_dwi'
    renameDWIagain =  pe.Node(Rename(), name="%s"%method_now)
    renameDWIagain.inputs.format_string = 'selectedDWI.nii.gz'

    # Generate QC for sct_dmri_moco ('dmri_crop_moco_dwi_mean_seg.nii.gz' is needed to align each slice in the QC mosaic)
    method_now = 'qc_on_dwi'
    qc_dwi = pe.Node(HelperInterfaces.qcDWI(), name="%s"%method_now)
    qc_dwi.inputs.step = 'sct_dmri_moco'
    qc_dwi.inputs.out_dir = "%s/all_outputs/%s" % (dataSink.inputs.base_directory,method_now) 
   
    # compute dwi metrics
    # tips: The flag -method "restore" allows you to estimate the tensor with robust fit (see: sct_dmri_compute_dti -h)
    method_now = 'dmri_standard_compute_dti'
    dmri_compute_dwi_st = pe.Node(HelperInterfaces.DMRIcomputeDTI(), name="%s"%method_now)
    dmri_compute_dwi_st.inputs.method = "standard" # options: standard and restore 
    dmri_compute_dwi_st.inputs.out_dir = "%s/all_outputs/%s" % (dataSink.inputs.base_directory,method_now) 
    method_now = 'dmri_restore_compute_dti'
    dmri_compute_dwi_re = pe.Node(HelperInterfaces.DMRIcomputeDTI(), name="%s"%method_now)
    dmri_compute_dwi_re.inputs.method = "restore" # options: standard and restore 
    dmri_compute_dwi_re.inputs.out_dir = "%s/all_outputs/%s" % (dataSink.inputs.base_directory,method_now) 
   
    # compute FA within right and left lateral corticospinal tracts from slices 2 to 14 using weighted average method
    method_now = 'right_left_lateral_corticospinal_tracts'
    rllct = pe.Node(HelperInterfaces.RightLeftFA(), name="%s"%method_now)
    rllct.inputs.slice_range = "2:14" 
    rllct.inputs.method = "wa" 
    rllct.inputs.out_dir = "%s/all_outputs/%s" % (dataSink.inputs.base_directory,method_now) 
    
    # diffusion tensor reconstruction using FSL
    method_now = 'dtifit_by_fsl'
    dtifit = pe.Node(fsl.DTIFit(), name="%s"%method_now)
    dtifit.inputs.args = '-w'
    dtifit.inputs.base_name = 'dtifitWLS'
    dtifit.inputs.output_type = 'NIFTI_GZ'

    # diffusion tensor reconstruction using MRTRIX3
    method_now = 'dtifit_by_MRTRIX3'
    dwifit_mrtrix = pe.Node(HelperInterfaces.DWI2Tensor(), name="%s"%method_now)
    dwifit_mrtrix.inputs.out_dir = "%s/all_outputs/%s" % (dataSink.inputs.base_directory,method_now) 

    # diffusion parameters extraction after tensor reconstruction using MRTRIX3
    method_now = 'dti_params_by_MRTRIX3'
    tensor2metric_mrtrix = pe.Node(HelperInterfaces.Tensor2Metric(), name="%s"%method_now)
    tensor2metric_mrtrix.inputs.out_dir = "%s/all_outputs/%s" % (dataSink.inputs.base_directory,method_now) 
        
    # dti-rd 
    method_now = 'dti-rd'
    dtird = pe.Node(HelperInterfaces.RDCompute(), name="%s"%method_now)

    # diffusion kurtosis reconstruction 
    method_now = 'dkifit'
    dkifit = pe.Node(HelperInterfaces.DKIfit(), name="%s"%method_now)
    dkifit.inputs.in_smooth = 0.0 # could be an extra input argument , like float(infoSource.inputs.smooth)
 
    # median filter on MK (remove black voxels) 
    method_now = 'mk-median-filter'
    mkmedfilt = pe.Node(HelperInterfaces.ApplyMedianOnMK(), name='%s'%method_now)
    method_now = 'ak-median-filter'
    akmedfilt = pe.Node(HelperInterfaces.ApplyMedianOnAK(), name='%s'%method_now)
    method_now = 'rk-median-filter'
    rkmedfilt = pe.Node(HelperInterfaces.ApplyMedianOnAK(), name='%s'%method_now)
 
    #-----------------------------------------------------------------------------------------------------#
    # CONNECT INPUT AND OUTPUT NODES
    #-----------------------------------------------------------------------------------------------------#
    print(colours.green + "Connect Nodes." + colours.ENDC)
    
    if strategy == "none":           
     preproc.connect([ 
        (infoSource, renameDWIagain, [('DWI_NII', 'in_file')]),
        (infoSource, b0, [('DWI_BVEC', 'bvec')]),
        (infoSource, dwifit_mrtrix, [('DWI_BVEC', 'bvec')]),
        (infoSource, dwifit_mrtrix, [('DWI_BVAL', 'bval')]),
        (infoSource, dataSink, [('DWI_BVEC', 'DWISpace.@bvec')]),
        (infoSource, dataSink, [('DWI_BVAL', 'DWISpace.@bval')]),
     ])
    elif strategy == "mppca":           
     preproc.connect([ 
        (infoSource, mppca, [('DWI_NII', 'in_file')]),
        # select corrected dwi
        (mppca, renameDWIagain, [('out_file', 'in_file')]),
        (infoSource, dwifit_mrtrix, [('DWI_BVEC', 'bvec')]),
        (infoSource, dwifit_mrtrix, [('DWI_BVAL', 'bval')]),
        (infoSource, b0, [('DWI_BVEC', 'bvec')]),
        (infoSource, dataSink, [('DWI_BVEC', 'DWISpace.@bvec')]),
        (infoSource, dataSink, [('DWI_BVAL', 'DWISpace.@bval')]),
     ])
    elif strategy == "gibbs":   
     preproc.connect([ 
        (infoSource, gibbs, [('DWI_NII', 'in_file')]),
        # select corrected dwi
        (gibbs, renameDWIagain, [('out_file', 'in_file')]),
        (infoSource, dwifit_mrtrix, [('DWI_BVEC', 'bvec')]),
        (infoSource, dwifit_mrtrix, [('DWI_BVAL', 'bval')]),
        (infoSource, b0, [('DWI_BVEC', 'bvec')]),
        (infoSource, dataSink, [('DWI_BVEC', 'DWISpace.@bvec')]),
        (infoSource, dataSink, [('DWI_BVAL', 'DWISpace.@bval')]),
     ])
    elif strategy == "moco":   
     preproc.connect([ 
        (infoSource, moco, [('DWI_NII', 'dwi_image')]),
        (infoSource, moco, [('DWI_BVEC', 'bvec')]),
        # select corrected dwi
        (moco, renameDWIagain, [('out_corrected', 'in_file')]), 
        (infoSource, dwifit_mrtrix, [('DWI_BVEC', 'bvec')]),
        (infoSource, dwifit_mrtrix, [('DWI_BVAL', 'bval')]),
        (infoSource, b0, [('DWI_BVEC', 'bvec')]),
        (moco, dataSink, [('out_corrected_mean', 'DWISpace.@dwi_moco_corrected_mean')]),
        (infoSource, dataSink, [('DWI_BVEC', 'DWISpace.@bvec')]),
        (infoSource, dataSink, [('DWI_BVAL', 'DWISpace.@bval')]),
     ])
    elif strategy == "eddy":   
     preproc.connect([ 
        (infoSource, eddy, [('DWI_NII', 'in_file')]),
        (infoSource, eddy, [('DWI_BVAL', 'in_bval')]),
        (infoSource, eddy, [('DWI_BVEC', 'in_bvec')]),
        (infoSource, eddy, [('DWI_JSON', 'in_json')]),
        # select corrected dwi
        (eddy, renameDWIagain, [('out_file', 'in_file')]), 
        (eddy, b0, [('new_bvec', 'bvec')]), 
        (eddy, dwifit_mrtrix, [('new_bvec', 'bvec')]),
        (eddy, dwifit_mrtrix, [('new_bval', 'bval')]),
        (eddy, dataSink, [('new_bvec', 'DWISpace.@bvec')]),
        (eddy, dataSink, [('new_bval', 'DWISpace.@bval')]),
     ])
    elif strategy == "eddy_topup":   
     preproc.connect([ 
        # possibly cut of slices in case it dimensions are odd to use topup default parameters
        (infoSource, cutDWI, [('DWI_NII', 'in_file')]),
        (infoSource, cutB0, [('DWI_REV_NII', 'in_file')]),

        # create list of both files to merge later
        (cutDWI, listDWI, [('roi_file', 'in1')]),
        (cutB0, listDWI, [('roi_file', 'in2')]),

        # merge a diffusion data and extra b0
        (listDWI, mergeOppostitePE, [('out', 'in_files')]),
        (mergeOppostitePE, renameDWI, [('merged_file', 'in_file')]),

        # after correction (noise & Gibbs) select b0 volumes only and merge to one file
        (infoSource, selectLowB, [('DWI_NII', 'in_file')]),
        (renameDWI, selectExtraB, [('out_file', 'in_file')]),
        (selectLowB, listB0, [('roi_file', 'in1')]),
        (selectExtraB, listB0, [('roi_file', 'in2')]),
        (listB0, mergeOppostiteB0, [('out', 'in_files')]),

        # apply eddy with topup information
       (infoSource, selectCorrectedDWI, [('DWI_NII', 'in_file')]),
       (selectCorrectedDWI, eddy, [('roi_file', 'in_file')]),
       (infoSource, eddy, [('DWI_BVAL', 'in_bval')]),
       (infoSource, eddy, [('DWI_BVEC', 'in_bvec')]),
       (infoSource, eddy, [('DWI_JSON', 'in_json')]),
       (mergeOppostiteB0, eddy, [('merged_file', 'se_epi')]),
       #(rename, eddy, [('out_file', 'in_mask')]),
       #(mask_dwi, eddy, [('out_mask', 'in_mask')]), 
        (eddy, b0, [('new_bvec', 'bvec')]),
        (eddy, dwifit_mrtrix, [('new_bvec', 'bvec')]),
        (eddy, dwifit_mrtrix, [('new_bval', 'bval')]),
        (eddy, dataSink, [('new_bvec', 'DWISpace.@bvec')]),
        (eddy, dataSink, [('new_bval', 'DWISpace.@bval')]),
        # select corrected dwi
        (eddy, renameDWIagain, [('out_file', 'in_file')]),
     ])
    elif strategy == "mppca_gibbs":   
     preproc.connect([ 
        (infoSource, mppca, [('DWI_NII', 'in_file')]),
        (mppca, gibbs, [('out_file', 'in_file')]),
        # select corrected dwi
        (gibbs, renameDWIagain, [('out_file', 'in_file')]), 
        (infoSource, dwifit_mrtrix, [('DWI_BVEC', 'bvec')]),
        (infoSource, dwifit_mrtrix, [('DWI_BVAL', 'bval')]),
        (infoSource, b0, [('DWI_BVEC', 'bvec')]),
        (infoSource, dataSink, [('DWI_BVEC', 'DWISpace.@bvec')]),
        (infoSource, dataSink, [('DWI_BVAL', 'DWISpace.@bval')]),
     ])
    elif strategy == "mppca_moco":   
     preproc.connect([ 
        (infoSource, mppca, [('DWI_NII', 'in_file')]),
        (mppca, moco, [('out_file', 'dwi_image')]),
        (infoSource, moco, [('DWI_BVEC', 'bvec')]),
        # select corrected dwi
        (moco, renameDWIagain, [('out_corrected', 'in_file')]), 
        (infoSource, dwifit_mrtrix, [('DWI_BVEC', 'bvec')]),
        (infoSource, dwifit_mrtrix, [('DWI_BVAL', 'bval')]),
        (infoSource, b0, [('DWI_BVEC', 'bvec')]),
        (moco, dataSink, [('out_corrected_mean', 'DWISpace.@dwi_moco_corrected_mean')]),
        (infoSource, dataSink, [('DWI_BVEC', 'DWISpace.@bvec')]),
        (infoSource, dataSink, [('DWI_BVAL', 'DWISpace.@bval')]),
     ])
    elif strategy == "mppca_gibbs_moco":   
     preproc.connect([ 
        (infoSource, mppca, [('DWI_NII', 'in_file')]),
        (mppca, gibbs, [('out_file', 'in_file')]),
        (infoSource, moco, [('DWI_BVEC', 'bvec')]),
        (gibbs, moco, [('out_file', 'dwi_image')]),
        # select corrected dwi
        (moco, renameDWIagain, [('out_corrected', 'in_file')]), 
        (infoSource, dwifit_mrtrix, [('DWI_BVEC', 'bvec')]),
        (infoSource, dwifit_mrtrix, [('DWI_BVAL', 'bval')]),
        (infoSource, b0, [('DWI_BVEC', 'bvec')]),
        (infoSource, dataSink, [('DWI_BVEC', 'DWISpace.@bvec')]),
        (infoSource, dataSink, [('DWI_BVAL', 'DWISpace.@bval')]),
     ])
    elif strategy == "mppca_eddy":   
     preproc.connect([ 
        (infoSource, mppca, [('DWI_NII', 'in_file')]),
        (mppca, eddy, [('out_file', 'in_file')]),
        (infoSource, eddy, [('DWI_BVAL', 'in_bval')]),
        (infoSource, eddy, [('DWI_BVEC', 'in_bvec')]),
        (infoSource, eddy, [('DWI_JSON', 'in_json')]),
        # select corrected dwi
        (eddy, renameDWIagain, [('out_file', 'in_file')]),
        (eddy, b0, [('new_bvec', 'bvec')]),
        (eddy, dwifit_mrtrix, [('new_bvec', 'bvec')]),
        (eddy, dwifit_mrtrix, [('new_bval', 'bval')]),
        (eddy, dataSink, [('new_bvec', 'DWISpace.@bvec')]),
        (eddy, dataSink, [('new_bval', 'DWISpace.@bval')]),
     ])
    elif strategy == "mppca_gibbs_moco_eddy":   
     preproc.connect([ 
        (infoSource, mppca, [('DWI_NII', 'in_file')]),
        (mppca, gibbs, [('out_file', 'in_file')]),
        (gibbs, moco, [('out_file', 'dwi_image')]),
        (infoSource, moco, [('DWI_BVEC', 'bvec')]),
        (moco, eddy, [('out_corrected', 'in_file')]),
        (infoSource, eddy, [('DWI_BVAL', 'in_bval')]),
        (infoSource, eddy, [('DWI_BVEC', 'in_bvec')]),
        (infoSource, eddy, [('DWI_JSON', 'in_json')]),
        # select corrected dwi
        (eddy, renameDWIagain, [('out_file', 'in_file')]),
        (eddy, b0, [('new_bvec', 'bvec')]),
        (eddy, dwifit_mrtrix, [('new_bvec', 'bvec')]),
        (eddy, dwifit_mrtrix, [('new_bval', 'bval')]),
        (eddy, dataSink, [('new_bvec', 'DWISpace.@bvec')]),
        (eddy, dataSink, [('new_bval', 'DWISpace.@bval')]),
     ])
    elif strategy == "gibbs_moco_eddy":   
     preproc.connect([ 
        (infoSource, gibbs, [('DWI_NII', 'in_file')]),
        (gibbs, moco, [('out_file', 'dwi_image')]),
        (infoSource, moco, [('DWI_BVEC', 'bvec')]),
        (moco, eddy, [('out_corrected', 'in_file')]),
        (infoSource, eddy, [('DWI_BVAL', 'in_bval')]),
        (infoSource, eddy, [('DWI_BVEC', 'in_bvec')]),
        (infoSource, eddy, [('DWI_JSON', 'in_json')]),
        # select corrected dwi
        (eddy, renameDWIagain, [('out_file', 'in_file')]),
        (eddy, b0, [('new_bvec', 'bvec')]),
        (eddy, dwifit_mrtrix, [('new_bvec', 'bvec')]),
        (eddy, dwifit_mrtrix, [('new_bval', 'bval')]),
        (eddy, dataSink, [('new_bvec', 'DWISpace.@bvec')]),
        (eddy, dataSink, [('new_bval', 'DWISpace.@bval')]),
     ])
    elif strategy == "mppca_eddy_topup":   
     preproc.connect([ 
        (infoSource, mppca, [('DWI_NII', 'in_file')]),
        # possibly cut of slices in case it dimensions are odd to use topup default parameters
        (mppca, cutDWI, [('out_file', 'in_file')]),
        (infoSource, cutB0, [('DWI_REV_NII', 'in_file')]),

        # create list of both files to merge later
        (cutDWI, listDWI, [('roi_file', 'in1')]),
        (cutB0, listDWI, [('roi_file', 'in2')]),

        # merge a diffusion data and extra b0
        (listDWI, mergeOppostitePE, [('out', 'in_files')]),
        (mergeOppostitePE, renameDWI, [('merged_file', 'in_file')]),

        # after correction (noise & Gibbs) select b0 volumes only and merge to one file
        (mppca, selectLowB, [('out_file', 'in_file')]),
        (renameDWI, selectExtraB, [('out_file', 'in_file')]),
        (selectLowB, listB0, [('roi_file', 'in1')]),
        (selectExtraB, listB0, [('roi_file', 'in2')]),
        (listB0, mergeOppostiteB0, [('out', 'in_files')]),

        # apply eddy with topup information
        (mppca, selectCorrectedDWI, [('out_file', 'in_file')]),
        (selectCorrectedDWI, eddy, [('roi_file', 'in_file')]),
        (infoSource, eddy, [('DWI_BVAL', 'in_bval')]),
        (infoSource, eddy, [('DWI_BVEC', 'in_bvec')]),
        (infoSource, eddy, [('DWI_JSON', 'in_json')]),
        (mergeOppostiteB0, eddy, [('merged_file', 'se_epi')]),
        # select corrected dwi
        (eddy, renameDWIagain, [('out_file', 'in_file')]),
        #(rename, eddy, [('out_file', 'in_mask')]),
        #(mask_dwi, eddy, [('out_mask', 'in_mask')]),
        (eddy, b0, [('new_bvec', 'bvec')]),
        (eddy, dwifit_mrtrix, [('new_bvec', 'bvec')]),
        (eddy, dwifit_mrtrix, [('new_bval', 'bval')]),
        (eddy, dataSink, [('new_bvec', 'DWISpace.@bvec')]),
        (eddy, dataSink, [('new_bval', 'DWISpace.@bval')]),
     ])
    elif strategy == "gibbs_moco":   
     preproc.connect([ 
        (infoSource, gibbs, [('DWI_NII', 'in_file')]),
        (gibbs, moco, [('out_file', 'dwi_image')]),
        (infoSource, moco, [('DWI_BVEC', 'bvec')]),
        # select corrected dwi
        (moco, renameDWIagain, [('out_corrected', 'in_file')]),
        (infoSource, dwifit_mrtrix, [('DWI_BVEC', 'bvec')]),
        (infoSource, dwifit_mrtrix, [('DWI_BVAL', 'bval')]),
        (infoSource, b0, [('DWI_BVEC', 'bvec')]),
        (moco, dataSink, [('out_corrected_mean', 'DWISpace.@dwi_moco_corrected_mean')]),
        (infoSource, dataSink, [('DWI_BVEC', 'DWISpace.@bvec')]),
        (infoSource, dataSink, [('DWI_BVAL', 'DWISpace.@bval')]),
     ])
    elif strategy == "gibbs_eddy":   
     preproc.connect([ 
        (infoSource, gibbs, [('DWI_NII', 'in_file')]),
        (gibbs, eddy, [('out_file', 'in_file')]),
        (infoSource, eddy, [('DWI_BVAL', 'in_bval')]),
        (infoSource, eddy, [('DWI_BVEC', 'in_bvec')]),
        (infoSource, eddy, [('DWI_JSON', 'in_json')]),
        # select corrected dwi
        (eddy, renameDWIagain, [('out_file', 'in_file')]),
        (eddy, b0, [('new_bvec', 'bvec')]), 
        (eddy, dwifit_mrtrix, [('new_bvec', 'bvec')]),
        (eddy, dwifit_mrtrix, [('new_bval', 'bval')]),
        (eddy, dataSink, [('new_bvec', 'DWISpace.@bvec')]),
        (eddy, dataSink, [('new_bval', 'DWISpace.@bval')]),
     ])
    elif strategy == "gibbs_eddy_topup":   
     preproc.connect([ 
        (infoSource, gibbs, [('DWI_NII', 'in_file')]),
        # possibly cut of slices in case it dimensions are odd to use topup default parameters
        (gibbs, cutDWI, [('out_file', 'in_file')]),
        (infoSource, cutB0, [('DWI_REV_NII', 'in_file')]),

        # create list of both files to merge later
        (cutDWI, listDWI, [('roi_file', 'in1')]),
        (cutB0, listDWI, [('roi_file', 'in2')]),

        # merge a diffusion data and extra b0
        (listDWI, mergeOppostitePE, [('out', 'in_files')]),
        (mergeOppostitePE, renameDWI, [('merged_file', 'in_file')]),

        # after correction (noise & Gibbs) select b0 volumes only and merge to one file
        (gibbs, selectLowB, [('out_file', 'in_file')]),
        (renameDWI, selectExtraB, [('out_file', 'in_file')]),
        (selectLowB, listB0, [('roi_file', 'in1')]),
        (selectExtraB, listB0, [('roi_file', 'in2')]),
        (listB0, mergeOppostiteB0, [('out', 'in_files')]),

        # apply eddy with topup information
        (gibbs, selectCorrectedDWI, [('out_file', 'in_file')]),
        (selectCorrectedDWI, eddy, [('roi_file', 'in_file')]),
        (infoSource, eddy, [('DWI_BVAL', 'in_bval')]),
        (infoSource, eddy, [('DWI_BVEC', 'in_bvec')]),
        (infoSource, eddy, [('DWI_JSON', 'in_json')]),
        (mergeOppostiteB0, eddy, [('merged_file', 'se_epi')]),
        #(rename, eddy, [('out_file', 'in_mask')]),
        #(mask_dwi, eddy, [('out_mask', 'in_mask')]),
        # select corrected dwi
        (eddy, renameDWIagain, [('out_file', 'in_file')]),
        (eddy, b0, [('new_bvec', 'bvec')]),
        (eddy, dwifit_mrtrix, [('new_bvec', 'bvec')]),
        (eddy, dwifit_mrtrix, [('new_bval', 'bval')]),
        (eddy, dataSink, [('new_bvec', 'DWISpace.@bvec')]),
        (eddy, dataSink, [('new_bval', 'DWISpace.@bval')]),
     ])
    elif strategy == "moco_eddy":   
     preproc.connect([ 
        (infoSource, moco, [('DWI_NII', 'dwi_image')]),
        (infoSource, moco, [('DWI_BVEC', 'bvec')]),
        (moco, eddy, [('out_corrected', 'in_file')]),
        (infoSource, eddy, [('DWI_BVAL', 'in_bval')]),
        (infoSource, eddy, [('DWI_BVEC', 'in_bvec')]),
        (infoSource, eddy, [('DWI_JSON', 'in_json')]),
        # select corrected dwi
        (eddy, renameDWIagain, [('out_file', 'in_file')]),
        (eddy, b0, [('new_bvec', 'bvec')]),
        (eddy, dwifit_mrtrix, [('new_bvec', 'bvec')]),
        (eddy, dwifit_mrtrix, [('new_bval', 'bval')]),
        (moco, dataSink, [('out_corrected_mean', 'DWISpace.@dwi_moco_corrected_mean')]),
        (eddy, dataSink, [('new_bvec', 'DWISpace.@bvec')]),
        (eddy, dataSink, [('new_bval', 'DWISpace.@bval')]),
     ])
    elif strategy == "moco_eddy_topup":   
     preproc.connect([ 
        (infoSource, moco, [('DWI_NII', 'dwi_image')]),
        (infoSource, moco, [('DWI_BVEC', 'bvec')]),
        # possibly cut of slices in case it dimensions are odd to use topup default parameters
        (moco, cutDWI, [('out_corrected', 'in_file')]),
        (infoSource, cutB0, [('DWI_REV_NII', 'in_file')]),

        # create list of both files to merge later
        (cutDWI, listDWI, [('roi_file', 'in1')]),
        (cutB0, listDWI, [('roi_file', 'in2')]),

        # merge a diffusion data and extra b0
        (listDWI, mergeOppostitePE, [('out', 'in_files')]),
        (mergeOppostitePE, renameDWI, [('merged_file', 'in_file')]),

        # after correction (noise & Gibbs) select b0 volumes only and merge to one file
        (moco, selectLowB, [('out_corrected', 'in_file')]),
        (renameDWI, selectExtraB, [('out_file', 'in_file')]),
        (selectLowB, listB0, [('roi_file', 'in1')]),
        (selectExtraB, listB0, [('roi_file', 'in2')]),
        (listB0, mergeOppostiteB0, [('out', 'in_files')]),

        # apply eddy with topup information
        (moco, selectCorrectedDWI, [('out_corrected', 'in_file')]),
        (selectCorrectedDWI, eddy, [('roi_file', 'in_file')]),
        (infoSource, eddy, [('DWI_BVAL', 'in_bval')]),
        (infoSource, eddy, [('DWI_BVEC', 'in_bvec')]),
        (infoSource, eddy, [('DWI_JSON', 'in_json')]),
        (mergeOppostiteB0, eddy, [('merged_file', 'se_epi')]),
        # select corrected dwi
        (eddy, renameDWIagain, [('out_file', 'in_file')]),
        #(rename, eddy, [('out_file', 'in_mask')]),
        #(mask_dwi, eddy, [('out_mask', 'in_mask')]),
        (eddy, b0, [('new_bvec', 'bvec')]),
        (eddy, dwifit_mrtrix, [('new_bvec', 'bvec')]),
        (eddy, dwifit_mrtrix, [('new_bval', 'bval')]),
        (moco, dataSink, [('out_corrected_mean', 'DWISpace.@dwi_moco_corrected_mean')]),
        (eddy, dataSink, [('new_bvec', 'DWISpace.@bvec')]),
        (eddy, dataSink, [('new_bval', 'DWISpace.@bval')]),
     ])
    elif strategy == "mppca_gibbs_eddy":   
     preproc.connect([ 
        (infoSource, mppca, [('DWI_NII', 'in_file')]),
        (mppca, gibbs, [('out_file', 'in_file')]),
        (gibbs, eddy, [('out_file', 'in_file')]),
        (infoSource, eddy, [('DWI_BVAL', 'in_bval')]),
        (infoSource, eddy, [('DWI_BVEC', 'in_bvec')]),
        (infoSource, eddy, [('DWI_JSON', 'in_json')]),
        # select corrected dwi
        (eddy, renameDWIagain, [('out_file', 'in_file')]),
        (eddy, b0, [('new_bvec', 'bvec')]),
        (eddy, dwifit_mrtrix, [('new_bvec', 'bvec')]),
        (eddy, dwifit_mrtrix, [('new_bval', 'bval')]),
        (eddy, dataSink, [('new_bvec', 'DWISpace.@bvec')]),
        (eddy, dataSink, [('new_bval', 'DWISpace.@bval')]),
     ])
    elif strategy == "mppca_gibbs_eddy_topup":   
     preproc.connect([ 
        (infoSource, mppca, [('DWI_NII', 'in_file')]),
        (mppca, gibbs, [('out_file', 'in_file')]),
        # possibly cut of slices in case it dimensions are odd to use topup default parameters
        (gibbs, cutDWI, [('out_file', 'in_file')]),
        (infoSource, cutB0, [('DWI_REV_NII', 'in_file')]),

        # create list of both files to merge later
        (cutDWI, listDWI, [('roi_file', 'in1')]),
        (cutB0, listDWI, [('roi_file', 'in2')]),

        # merge a diffusion data and extra b0
        (listDWI, mergeOppostitePE, [('out', 'in_files')]),
        (mergeOppostitePE, renameDWI, [('merged_file', 'in_file')]),

        # after correction (noise & Gibbs) select b0 volumes only and merge to one file
        (gibbs, selectLowB, [('out_file', 'in_file')]),
        (renameDWI, selectExtraB, [('out_file', 'in_file')]),
        (selectLowB, listB0, [('roi_file', 'in1')]),
        (selectExtraB, listB0, [('roi_file', 'in2')]),
        (listB0, mergeOppostiteB0, [('out', 'in_files')]),

        # apply eddy with topup information
        (gibbs, selectCorrectedDWI, [('out_file', 'in_file')]),
        (selectCorrectedDWI, eddy, [('roi_file', 'in_file')]),
        (infoSource, eddy, [('DWI_BVAL', 'in_bval')]),
        (infoSource, eddy, [('DWI_BVEC', 'in_bvec')]),
        (infoSource, eddy, [('DWI_JSON', 'in_json')]),
        (mergeOppostiteB0, eddy, [('merged_file', 'se_epi')]),
        # select corrected dwi
        (eddy, renameDWIagain, [('out_file', 'in_file')]),
        #(rename, eddy, [('out_file', 'in_mask')]),
        #(mask_dwi, eddy, [('out_mask', 'in_mask')]),
        (eddy, b0, [('new_bvec', 'bvec')]),
        (eddy, dwifit_mrtrix, [('new_bvec', 'bvec')]),
        (eddy, dwifit_mrtrix, [('new_bval', 'bval')]),
        (eddy, dataSink, [('new_bvec', 'DWISpace.@bvec')]),
        (eddy, dataSink, [('new_bval', 'DWISpace.@bval')]),
     ])
    elif strategy == "mppca_moco_eddy":   
     preproc.connect([ 
        (infoSource, mppca, [('DWI_NII', 'in_file')]),
        (mppca, moco, [('out_file', 'dwi_image')]),
        (infoSource, moco, [('DWI_BVEC', 'bvec')]),
        (moco, eddy, [('out_corrected', 'in_file')]),
        (infoSource, eddy, [('DWI_BVAL', 'in_bval')]),
        (infoSource, eddy, [('DWI_BVEC', 'in_bvec')]),
        (infoSource, eddy, [('DWI_JSON', 'in_json')]),
        # select corrected dwi
        (eddy, renameDWIagain, [('out_file', 'in_file')]),
        (eddy, b0, [('new_bvec', 'bvec')]),
        (eddy, dwifit_mrtrix, [('new_bvec', 'bvec')]),
        (eddy, dwifit_mrtrix, [('new_bval', 'bval')]),
        (moco, dataSink, [('out_corrected_mean', 'DWISpace.@dwi_moco_corrected_mean')]),
        (eddy, dataSink, [('new_bvec', 'DWISpace.@bvec')]),
        (eddy, dataSink, [('new_bval', 'DWISpace.@bval')]),
     ])
    elif strategy == "mppca_moco_eddy_topup":
     preproc.connect([ 
        (infoSource, mppca, [('DWI_NII', 'in_file')]),
        (mppca, moco, [('out_file', 'dwi_image')]),
        (infoSource, moco, [('DWI_BVEC', 'bvec')]),
        # possibly cut of slices in case it dimensions are odd to use topup default parameters
        (moco, cutDWI, [('out_corrected', 'in_file')]),
        (infoSource, cutB0, [('DWI_REV_NII', 'in_file')]),

        # create list of both files to merge later
        (cutDWI, listDWI, [('roi_file', 'in1')]),
        (cutB0, listDWI, [('roi_file', 'in2')]),

        # merge a diffusion data and extra b0
        (listDWI, mergeOppostitePE, [('out', 'in_files')]),
        (mergeOppostitePE, renameDWI, [('merged_file', 'in_file')]),

        # after correction (noise & Gibbs) select b0 volumes only and merge to one file
        (moco, selectLowB, [('out_corrected', 'in_file')]),
        (renameDWI, selectExtraB, [('out_file', 'in_file')]),
        (selectLowB, listB0, [('roi_file', 'in1')]),
        (selectExtraB, listB0, [('roi_file', 'in2')]),
        (listB0, mergeOppostiteB0, [('out', 'in_files')]),

        # apply eddy with topup information
        (moco, selectCorrectedDWI, [('out_corrected', 'in_file')]),
        (selectCorrectedDWI, eddy, [('roi_file', 'in_file')]),
        (infoSource, eddy, [('DWI_BVAL', 'in_bval')]),
        (infoSource, eddy, [('DWI_BVEC', 'in_bvec')]),
        (infoSource, eddy, [('DWI_JSON', 'in_json')]),
        (mergeOppostiteB0, eddy, [('merged_file', 'se_epi')]),
        # select corrected dwi
        (eddy, renameDWIagain, [('out_file', 'in_file')]),
        #(rename, eddy, [('out_file', 'in_mask')]),
        #(mask_dwi, eddy, [('out_mask', 'in_mask')]),
        (eddy, b0, [('new_bvec', 'bvec')]),
        (eddy, dwifit_mrtrix, [('new_bvec', 'bvec')]),
        (eddy, dwifit_mrtrix, [('new_bval', 'bval')]),
        (moco, dataSink, [('out_corrected_mean', 'DWISpace.@dwi_moco_corrected_mean')]),
        (eddy, dataSink, [('new_bvec', 'DWISpace.@bvec')]),
        (eddy, dataSink, [('new_bval', 'DWISpace.@bval')]),
     ])
    elif strategy == "mppca_gibbs_moco_eddy_topup":   
     preproc.connect([ 
        (infoSource, mppca, [('DWI_NII', 'in_file')]),
        (mppca, gibbs, [('out_file', 'in_file')]),
        (gibbs, moco, [('out_file', 'dwi_image')]),
        (infoSource, moco, [('DWI_BVEC', 'bvec')]),
        # possibly cut of slices in case it dimensions are odd to use topup default parameters
        (moco, cutDWI, [('out_corrected', 'in_file')]),
        (infoSource, cutB0, [('DWI_REV_NII', 'in_file')]),

        # create list of both files to merge later
        (cutDWI, listDWI, [('roi_file', 'in1')]),
        (cutB0, listDWI, [('roi_file', 'in2')]),

        # merge a diffusion data and extra b0
        (listDWI, mergeOppostitePE, [('out', 'in_files')]),
        (mergeOppostitePE, renameDWI, [('merged_file', 'in_file')]),

        # after correction (noise & Gibbs) select b0 volumes only and merge to one file
        (moco, selectLowB, [('out_corrected', 'in_file')]),
        (renameDWI, selectExtraB, [('out_file', 'in_file')]),
        (selectLowB, listB0, [('roi_file', 'in1')]),
        (selectExtraB, listB0, [('roi_file', 'in2')]),
        (listB0, mergeOppostiteB0, [('out', 'in_files')]),

        # apply eddy with topup information
        (moco, selectCorrectedDWI, [('out_corrected', 'in_file')]),
        (selectCorrectedDWI, eddy, [('roi_file', 'in_file')]),
        (infoSource, eddy, [('DWI_BVAL', 'in_bval')]),
        (infoSource, eddy, [('DWI_BVEC', 'in_bvec')]),
        (infoSource, eddy, [('DWI_JSON', 'in_json')]),
        (mergeOppostiteB0, eddy, [('merged_file', 'se_epi')]),
        # select corrected dwi
        (eddy, renameDWIagain, [('out_file', 'in_file')]),
        #(rename, eddy, [('out_file', 'in_mask')]),
        #(mask_dwi, eddy, [('out_mask', 'in_mask')]),
        (eddy, b0, [('new_bvec', 'bvec')]),
        (eddy, dwifit_mrtrix, [('new_bvec', 'bvec')]),
        (eddy, dwifit_mrtrix, [('new_bval', 'bval')]),
        (moco, dataSink, [('out_corrected_mean', 'DWISpace.@dwi_moco_corrected_mean')]),
        (eddy, dataSink, [('new_bvec', 'DWISpace.@bvec')]),
        (eddy, dataSink, [('new_bval', 'DWISpace.@bval')]),
     ])
    elif strategy == "gibbs_moco_eddy_topup":   
     preproc.connect([ 
        (infoSource, gibbs, [('DWI_NII', 'in_file')]),
        (gibbs, moco, [('out_file', 'dwi_image')]),
        (infoSource, moco, [('DWI_BVEC', 'bvec')]),
        # possibly cut of slices in case it dimensions are odd to use topup default parameters
        (moco, cutDWI, [('out_corrected', 'in_file')]),
        (infoSource, cutB0, [('DWI_REV_NII', 'in_file')]),

        # create list of both files to merge later
        (cutDWI, listDWI, [('roi_file', 'in1')]),
        (cutB0, listDWI, [('roi_file', 'in2')]),

        # merge a diffusion data and extra b0
        (listDWI, mergeOppostitePE, [('out', 'in_files')]),
        (mergeOppostitePE, renameDWI, [('merged_file', 'in_file')]),

        # after correction (noise & Gibbs) select b0 volumes only and merge to one file
        (moco, selectLowB, [('out_corrected', 'in_file')]),
        (renameDWI, selectExtraB, [('out_file', 'in_file')]),
        (selectLowB, listB0, [('roi_file', 'in1')]),
        (selectExtraB, listB0, [('roi_file', 'in2')]),
        (listB0, mergeOppostiteB0, [('out', 'in_files')]),

        # apply eddy with topup information
        (moco, selectCorrectedDWI, [('out_corrected', 'in_file')]),
        (selectCorrectedDWI, eddy, [('roi_file', 'in_file')]),
        (infoSource, eddy, [('DWI_BVAL', 'in_bval')]),
        (infoSource, eddy, [('DWI_BVEC', 'in_bvec')]),
        (infoSource, eddy, [('DWI_JSON', 'in_json')]),
        (mergeOppostiteB0, eddy, [('merged_file', 'se_epi')]),
        # select corrected dwi
        (eddy, renameDWIagain, [('out_file', 'in_file')]),
        #(rename, eddy, [('out_file', 'in_mask')]),
        #(mask_dwi, eddy, [('out_mask', 'in_mask')]),
        (eddy, b0, [('new_bvec', 'bvec')]),
        (eddy, dwifit_mrtrix, [('new_bvec', 'bvec')]),
        (eddy, dwifit_mrtrix, [('new_bval', 'bval')]),
        (moco, dataSink, [('out_corrected_mean', 'DWISpace.@dwi_moco_corrected_mean')]),
        (eddy, dataSink, [('new_bvec', 'DWISpace.@bvec')]),
        (eddy, dataSink, [('new_bval', 'DWISpace.@bval')]),
     ])
    preproc.connect([
        # dwi: separate b0 and dwi
        (renameDWIagain, b0, [('out_file', 'dwi_image')]),
        # tensor reconstruction
        (renameDWIagain, dwifit_mrtrix, [('out_file', 'dwi_image')]),
        (b0, dwifit_mrtrix, [('out_b0_mean', 'b0')]),
        # compute dwi metrics
        (dwifit_mrtrix, tensor2metric_mrtrix, [('out_tensor', 'dwi_tensor')]), 
        # DATASINK
        (infoSource, dataSink, [('T2', 'T2Space.@T2w_original')]),
        (b0, dataSink, [('out_b0_mean', 'DWISpace.@b0')]),
        (b0, dataSink, [('out_dwi_mean', 'DWISpace.@dwi')]),
        (renameDWIagain, dataSink, [('out_file', 'DWISpace.@dwi_corrected')]),
        (tensor2metric_mrtrix, dataSink, [('out_fa', 'DWISpace.@FA')]),
        (tensor2metric_mrtrix, dataSink, [('out_md', 'DWISpace.@MD')]),
        (tensor2metric_mrtrix, dataSink, [('out_ad', 'DWISpace.@AD')]),
        (tensor2metric_mrtrix, dataSink, [('out_rd', 'DWISpace.@RD')]),
    ])
    
    return preproc


if __name__=='__main__':
    pipeLine = BuildPipeLine(sys.argv[1:])
    pipeLine.write_graph()
    print(colours.green + "Run Pipeline..." + colours.ENDC)
    pipeLine.run(plugin='MultiProc')
    print(colours.green + "Pipeline completed." + colours.ENDC)

