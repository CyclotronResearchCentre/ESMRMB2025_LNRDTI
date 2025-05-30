# Evgenios Kornaropoulos - ekornaropoulos@uliege.be
# 27 November 2024

import sys, os, socket, copy, shutil, getpass, glob, csv
import nibabel as nb
import numpy as np

from nipype.interfaces.base import (
    BaseInterface,
    BaseInterfaceInputSpec,
    traits,
    File,
    TraitedSpec,
    Directory,
    isdefined,
    CommandLineInputSpec,
    CommandLine
)

from itertools import product

from nipype.utils.filemanip import split_filename

from dipy.io import read_bvals_bvecs
from dipy.data import gradient_table
from dipy.denoise.pca_noise_estimate import pca_noise_estimate
from dipy.denoise.localpca import localpca

import dipy.reconst.fwdti as fwdti
from dipy.reconst.ivim import IvimModel

dirToAdd = os.path.dirname(os.path.realpath(__file__))
sys.path.append('%s' % dirToAdd)
import subprocess

from nipype.interfaces import fsl

from computeSimilarityMetrics import computeSimilarityMetrics
pip_dir = "%s" % dirToAdd

from pandas import DataFrame as df
import dipy.reconst.fwdti as fwdti
from dipy.reconst.ivim import IvimModel
from scipy import stats
from scipy import signal

#######################################################################################################
#
# HELPERINTEFACE FOR STRUCTURAL PIPELINE
#
#######################################################################################################

#-----------------------------------------------------------------------------------------------------#
# Spinal Cord Segmentation (default)
#-----------------------------------------------------------------------------------------------------#
class SpinalCordSegmentationDeepSegInputSpec(BaseInterfaceInputSpec):
    input_image = traits.File(exists=True, desc='input_image', mandatory=True)
    seq_type = traits.Str(exists=True, desc='sequence type', mandatory=True)
    out_dir = traits.Str(exists=True, desc='output_dir', mandatory=True)

class SpinalCordSegmentationDeepSegOutputSpec(TraitedSpec):
    out_file = traits.File(exists=True, desc='output_segmentation')

class SpinalCordSegmentationDeepSeg(BaseInterface):
    input_spec = SpinalCordSegmentationDeepSegInputSpec
    output_spec = SpinalCordSegmentationDeepSegOutputSpec

    def _run_interface(self, runtime):

        im = self.inputs.input_image.split("/")[-1]
        self.name_out_file = im.split(".")[0]
        # define name for output file
        self.out_file = os.path.join(self.inputs.out_dir, "%s_spinal_cord_via_deepseg.nii.gz"%self.name_out_file)
        _segment = segment_deep_seg(a1 = self.inputs.input_image, a2 = self.inputs.seq_type, a3 = self.inputs.out_dir,a4=self.out_file)
        _segment.run() 

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = self.out_file
        return outputs

class segment_deep_segInputSpec(CommandLineInputSpec):
    a1 = traits.File(exists=True, desc='input image', mandatory=True, argstr="-i %s")
    a2 = traits.Str(exists=True, desc='sequence type', mandatory=True, argstr="-c %s")
    a3 = traits.Str(exists=True, desc='QC output', mandatory=True, argstr="-qc %s")
    a4 = traits.Str(exists=True, desc='output', mandatory=True, argstr="-o %s")

class segment_deep_seg(CommandLine):
    input_spec = segment_deep_segInputSpec
    _cmd = 'sct_deepseg_sc'

#-----------------------------------------------------------------------------------------------------#
# Spinal Cord Segmentation (alternative)
#-----------------------------------------------------------------------------------------------------#
class SpinalCordSegmentationPropSegInputSpec(BaseInterfaceInputSpec):
    input_image = traits.File(exists=True, desc='input_image', mandatory=True)
    seq_type = traits.Str(exists=True, desc='sequence type', mandatory=True)
    out_dir = traits.Str(exists=True, desc='output_dir', mandatory=True)

class SpinalCordSegmentationPropSegOutputSpec(TraitedSpec):
    out_file = traits.File(exists=True, desc='output_segmentation')

class SpinalCordSegmentationPropSeg(BaseInterface):
    input_spec = SpinalCordSegmentationPropSegInputSpec
    output_spec = SpinalCordSegmentationPropSegOutputSpec

    def _run_interface(self, runtime):

        im = self.inputs.input_image.split("/")[-1]
        self.name_out_file = im.split(".")[0]
        # define name for output file
        self.out_file = os.path.join(self.inputs.out_dir, "%s_spinal_cord_via_propseg.nii.gz"%self.name_out_file)
        _segment = segment_propseg(a1 = self.inputs.input_image, a2 = self.inputs.seq_type, a3 = self.inputs.out_dir,a4=self.out_file)
        _segment.run() 

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = self.out_file
        return outputs

class segment_propsegInputSpec(CommandLineInputSpec):
    a1 = traits.File(exists=True, desc='input image', mandatory=True, argstr="-i %s")
    a2 = traits.Str(exists=True, desc='sequence type', mandatory=True, argstr="-c %s")
    a3 = traits.Str(exists=True, desc='QC output', mandatory=True, argstr="-qc %s")
    a4 = traits.Str(exists=True, desc='output', mandatory=True, argstr="-o %s")

class segment_propseg(CommandLine):
    input_spec = segment_propsegInputSpec
    _cmd = 'sct_propseg'
    
#-----------------------------------------------------------------------------------------------------#
# Centerline computation
#-----------------------------------------------------------------------------------------------------#
class CenterlineInputSpec(BaseInterfaceInputSpec):
    input_image = traits.File(exists=True, desc='input_image', mandatory=True)
    method = traits.Str(exists=True, desc='method used', mandatory=True)
    soft = traits.Int(exists=True, desc='whether soft', mandatory=True)
    out_dir = traits.Str(exists=True, desc='output_dir', mandatory=True)

class CenterlineOutputSpec(TraitedSpec):
    out_file = traits.File(exists=True, desc='output_centerline')

class Centerline(BaseInterface):
    input_spec = CenterlineInputSpec
    output_spec = CenterlineOutputSpec

    def _run_interface(self, runtime):

        im = self.inputs.input_image.split("/")[-1]
        self.name_out_file = im.split(".")[0]
        # define name for output file
        if self.inputs.soft == 0:
            self.out_file = os.path.join(self.inputs.out_dir, "%s_centerline_binarized.nii.gz"%self.name_out_file)
        else:
            self.out_file = os.path.join(self.inputs.out_dir, "%s_centerline_soft.nii.gz"%self.name_out_file)
        _segment = segment_centerline(a1 = self.inputs.input_image, a2 = self.inputs.method, a3 = self.inputs.out_dir,a4=self.inputs.soft,a5=self.out_file)
        _segment.run() 

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = self.out_file
        return outputs

class segment_centerlineInputSpec(CommandLineInputSpec):
    a1 = traits.File(exists=True, desc='input image', mandatory=True, argstr="-i %s")
    a2 = traits.Str(exists=True, desc='method used', mandatory=True, argstr="-method %s")
    a3 = traits.Str(exists=True, desc='QC output', mandatory=True, argstr="-qc %s")
    a4 = traits.Int(exists=True, desc='type of centerline', mandatory=True, argstr="-centerline-soft %s")
    a5 = traits.Str(exists=True, desc='output', mandatory=True, argstr="-o %s")

class segment_centerline(CommandLine):
    input_spec = segment_centerlineInputSpec
    _cmd = 'sct_get_centerline'

#-----------------------------------------------------------------------------------------------------#
# Vertebral level detection
#-----------------------------------------------------------------------------------------------------#
class VertrebalLabelingInputSpec(BaseInterfaceInputSpec):
    anat_image = traits.File(exists=True, desc='image with structural information', mandatory=True)
    segmentation_image = traits.File(exists=True, desc='spinal-cord-segmentation image', mandatory=True)
    seq_type = traits.Str(exists=True, desc='sequence type', mandatory=True)
    out_dir = traits.Str(exists=True, desc='output_dir', mandatory=True)

class VertrebalLabelingOutputSpec(TraitedSpec):
    out_file = traits.File(exists=True, desc='output_segmentation')

class VertrebalLabeling(BaseInterface):
    input_spec = VertrebalLabelingInputSpec
    output_spec = VertrebalLabelingOutputSpec

    def _run_interface(self, runtime):

        im = self.inputs.anat_image.split("/")[-1]
        self.name_out_file = im.split(".")[0]
        # define name for output file
        self.out_file = os.path.join(self.inputs.out_dir, "%s_vertrebal_labeling.nii.gz"%self.name_out_file)
        _labeling = labeling(a1 = self.inputs.anat_image,a2 = self.inputs.segmentation_image, a3 = self.inputs.seq_type, a4 = self.inputs.out_dir,a5=self.out_file)
        _labeling.run() 

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = self.out_file
        return outputs

class labelingInputSpec(CommandLineInputSpec):
    a1 = traits.File(exists=True, desc='image with structural information', mandatory=True, argstr="-i %s")
    a2 = traits.File(exists=True, desc='spinal-cord-segmentation image', mandatory=True, argstr="-s %s")
    a3 = traits.Str(exists=True, desc='sequence type', mandatory=True, argstr="-c %s")
    a4 = traits.Str(exists=True, desc='QC output', mandatory=True, argstr="-qc %s")
    a5 = traits.Str(exists=True, desc='output', mandatory=True, argstr="-o %s")

class labeling(CommandLine):
    input_spec = labelingInputSpec
    _cmd = 'sct_label_vertebrae'
    
#-----------------------------------------------------------------------------------------------------#
# Mid-vertebral level detection
#-----------------------------------------------------------------------------------------------------#
class MidVertrebalLabelingInputSpec(BaseInterfaceInputSpec):
    anat_image = traits.File(exists=True, desc='image with structural information', mandatory=True)
    vert_body = traits.Float(exists=True, desc='level of mid-vertrebae', mandatory=True)
    out_dir = traits.Str(exists=True, desc='output_dir', mandatory=True)

class MidVertrebalLabelingOutputSpec(TraitedSpec):
    out_file = traits.File(exists=True, desc='output_segmentation')

class MidVertrebalLabeling(BaseInterface):
    input_spec = MidVertrebalLabelingInputSpec
    output_spec = MidVertrebalLabelingOutputSpec

    def _run_interface(self, runtime):

        im = self.inputs.anat_image.split("/")[-1]
        self.name_out_file = im.split(".")[0]
        # define name for output file
        self.out_file = os.path.join(self.inputs.out_dir, "%s_mid-vertrebal_labeling.nii.gz"%self.name_out_file)
        _labeling = labeling(a1 = self.inputs.anat_image,a2 = self.inputs.vert_body, a3=self.out_file)
        _labeling.run() 

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = self.out_file
        return outputs

class labelingInputSpec(CommandLineInputSpec):
    a1 = traits.File(exists=True, desc='image with structural information', mandatory=True, argstr="-i %s")
    a2 = traits.Float(exists=True, desc='level of mid-vertrebae', mandatory=True, argstr="-vert-body %s")
    a3 = traits.Str(exists=True, desc='output', mandatory=True, argstr="-o %s")

class labeling(CommandLine):
    input_spec = labelingInputSpec
    _cmd = 'sct_label_utils'

#-----------------------------------------------------------------------------------------------------#
# Registering the spinal cord to the template
#-----------------------------------------------------------------------------------------------------#
class RegisterToTemplateInputSpec(BaseInterfaceInputSpec):
    anat_image = traits.File(exists=True, desc='image with structural information', mandatory=True)
    segmentation_image = traits.File(exists=True, desc='segmentation image', mandatory=True)
    seq_type = traits.Str(exists=True, desc='sequence type', mandatory=True)
    out_dir = traits.Str(exists=True, desc='output_dir', mandatory=True)

class RegisterToTemplateOutputSpec(TraitedSpec):
    out_file = traits.File(exists=True, desc='output_segmentation')

class RegisterToTemplate(BaseInterface):
    input_spec = RegisterToTemplateInputSpec
    output_spec = RegisterToTemplateOutputSpec

    def _run_interface(self, runtime):

        im = self.inputs.anat_image.split("/")[-1]
        self.name_out_file = im.split(".")[0]
        # define name for output file
        self.out_file = os.path.join(self.inputs.out_dir, "%s_registered_to_template.nii.gz"%self.name_out_file)
        _register = register(a1 = self.inputs.anat_image,a2 = self.inputs.segmentation_image, a3 = self.inputs.seq_type, a4 = self.inputs.out_dir,a5=self.out_file)
        _register.run() 

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = self.out_file
        return outputs

class registerInputSpec(CommandLineInputSpec):
    a1 = traits.File(exists=True, desc='image with structural information', mandatory=True, argstr="-i %s")
    a2 = traits.File(exists=True, desc='segmentation image', mandatory=True, argstr="-s %s")
    a3 = traits.Str(exists=True, desc='sequence type', mandatory=True, argstr="-c %s")
    a4 = traits.Str(exists=True, desc='QC output', mandatory=True, argstr="-qc %s")
    a5 = traits.Str(exists=True, desc='output', mandatory=True, argstr="-o %s")

class register(CommandLine):
    input_spec = registerInputSpec
    _cmd = 'sct_register_to_template'

#-----------------------------------------------------------------------------------------------------#
# Computation of cross-sectional area
#-----------------------------------------------------------------------------------------------------#
class CrossSectionAreaInputSpec(BaseInterfaceInputSpec):
    input_image = traits.File(exists=True, desc='segmentation image', mandatory=True)
    out_dir = traits.Str(exists=True, desc='output_dir', mandatory=True)

class CrossSectionAreaOutputSpec(TraitedSpec):
    out_file = traits.File(exists=True, desc='output_segmentation')

class CrossSectionArea(BaseInterface):
    input_spec = CrossSectionAreaInputSpec
    output_spec = CrossSectionAreaOutputSpec

    def _run_interface(self, runtime):

        im = self.inputs.input_image.split("/")[-1]
        self.name_out_file = im.split(".")[0]
        # define name for output file
        self.out_file = os.path.join(self.inputs.out_dir, "%s_cross_sectional_area.csv"%self.name_out_file)
        _csa = csa(a1 = self.inputs.input_image, a2=self.out_file)
        _csa.run() 

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = self.out_file
        return outputs

class csaInputSpec(CommandLineInputSpec):
    a1 = traits.File(exists=True, desc='segmentation image', mandatory=True, argstr="-i %s")
    a2 = traits.Str(exists=True, desc='output', mandatory=True, argstr="-o %s")

class csa(CommandLine):
    input_spec = csaInputSpec
    _cmd = 'sct_process_segmentation'
    
#-----------------------------------------------------------------------------------------------------#
# Detection of PMJ
#-----------------------------------------------------------------------------------------------------#
class DetectPMJInputSpec(BaseInterfaceInputSpec):
    input_image = traits.File(exists=True, desc='segmentation image', mandatory=True)
    seq_type = traits.Str(exists=True, desc='sequence type', mandatory=True)
    out_dir = traits.Str(exists=True, desc='output_dir', mandatory=True)

class DetectPMJOutputSpec(TraitedSpec):
    out_file = traits.File(exists=True, desc='output_segmentation')

class DetectPMJ(BaseInterface):
    input_spec = DetectPMJInputSpec
    output_spec = DetectPMJOutputSpec

    def _run_interface(self, runtime):

        im = self.inputs.input_image.split("/")[-1]
        self.name_out_file = im.split(".")[0]
        # define name for output file
        self.out_file = os.path.join(self.inputs.out_dir, "%s_pmj.nii.gz"%self.name_out_file)

        _pmj = pmj(a1 = self.inputs.input_image, a2 = self.inputs.seq_type, a3 = self.inputs.out_dir,a4=self.out_file)
        _pmj.run() 

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = self.out_file
        return outputs

class pmjInputSpec(CommandLineInputSpec):
    a1 = traits.File(exists=True, desc='segmentation image', mandatory=True, argstr="-i %s")
    a2 = traits.Str(exists=True, desc='sequence type', mandatory=True, argstr="-c %s")
    a3 = traits.Str(exists=True, desc='QC output', mandatory=True, argstr="-qc %s")
    a4 = traits.Str(exists=True, desc='output', mandatory=True, argstr="-o %s")

class pmj(CommandLine):
    input_spec = pmjInputSpec
    _cmd = 'sct_detect_pmj'
    
#-----------------------------------------------------------------------------------------------------#
# Separate B0 from DWI
#-----------------------------------------------------------------------------------------------------#
class SeparateB0InputSpec(BaseInterfaceInputSpec):
    dwi_image = traits.File(exists=True, desc='DWI image', mandatory=True)
    bvec = traits.File(exists=True, desc='DWI bvecs', mandatory=True)
    out_dir = traits.Str(exists=True, desc='output_dir', mandatory=True)

class SeparateB0OutputSpec(TraitedSpec):
    out_b0 = traits.File(exists=True, desc='output_b0')
    out_b0_mean = traits.File(exists=True, desc='output_b0_mean')
    out_dwi = traits.File(exists=True, desc='output_DWI')
    out_dwi_mean = traits.File(exists=True, desc='output_DWI_mean')


class SeparateB0(BaseInterface):
    input_spec = SeparateB0InputSpec
    output_spec = SeparateB0OutputSpec

    def _run_interface(self, runtime):

        im = self.inputs.dwi_image.split("/")[-1]
        self.name_out_file = im.split(".")[0]
        # define name for output file
        self.out_b0 = os.path.join(self.inputs.out_dir, "%s_b0.nii.gz"%self.name_out_file)
        self.out_b0_mean = os.path.join(self.inputs.out_dir, "%s_b0_mean.nii.gz"%self.name_out_file)
        self.out_dwi = os.path.join(self.inputs.out_dir, "%s_dwi.nii.gz"%self.name_out_file)
        self.out_dwi_mean = os.path.join(self.inputs.out_dir, "%s_dwi_mean.nii.gz"%self.name_out_file)

        _separate = separate(a1 = self.inputs.dwi_image, a2 = self.inputs.bvec, a3 = self.inputs.out_dir)
        _separate.run() 

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_b0'] = self.out_b0
        outputs['out_b0_mean'] = self.out_b0_mean
        outputs['out_dwi'] = self.out_dwi
        outputs['out_dwi_mean'] = self.out_dwi_mean
        return outputs

class separateInputSpec(CommandLineInputSpec):
    a1 = traits.File(exists=True, desc='dwi image', mandatory=True, argstr="-i %s")
    a2 = traits.Str(exists=True, desc='dwi bvec', mandatory=True, argstr="-bvec %s")
    a3 = traits.Str(exists=True, desc='output', mandatory=True, argstr="-ofolder %s")

class separate(CommandLine):
    input_spec = separateInputSpec
    _cmd = 'sct_dmri_separate_b0_and_dwi'

#-----------------------------------------------------------------------------------------------------#
# Coregister T2w and DWI via SCT
#-----------------------------------------------------------------------------------------------------#
class T2wDWIRegInputSpec(BaseInterfaceInputSpec):
    dwi_image = traits.File(exists=True, desc='DWI image', mandatory=True)
    sc_image = traits.File(exists=True, desc='spinal cord segmented image', mandatory=True)
    identity = traits.Int(exists=True, desc='identity', mandatory=True)
    interpolation = traits.Str(exists=True, desc='interpolation', mandatory=True)
    out_dir = traits.Str(exists=True, desc='output_dir', mandatory=True)

class T2wDWIRegOutputSpec(TraitedSpec):
    out_DWI_on_T2space = traits.File(exists=True, desc='registered DWI')
    out_sc_on_DWIspace = traits.File(exists=True, desc='registered spinal cord segmentation')


class T2wDWIReg(BaseInterface):
    input_spec = T2wDWIRegInputSpec
    output_spec = T2wDWIRegOutputSpec

    def _run_interface(self, runtime):

        sc_name = self.inputs.sc_image.split("/")[-1]
        self.sc_name = sc_name.split(".")[0]
        dwi_name = self.inputs.dwi_image.split("/")[-1]
        self.dwi_name = dwi_name.split(".")[0]
        # define name for output file
        self.out_sc_on_DWIspace = os.path.join(self.inputs.out_dir, "%s_reg.nii.gz"%self.sc_name)
        self.out_DWI_on_T2space = os.path.join(self.inputs.out_dir, "%s_reg.nii.gz"%self.dwi_name)

        _register = register(a1 = self.inputs.dwi_image, a2 = self.inputs.sc_image, a3 = self.inputs.identity, a4 = self.inputs.interpolation, a5 = self.inputs.out_dir)
        _register.run() 

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_DWI_on_T2space'] = self.out_DWI_on_T2space
        outputs['out_sc_on_DWIspace'] = self.out_sc_on_DWIspace
        return outputs

class registerInputSpec(CommandLineInputSpec):
    a1 = traits.File(exists=True, desc='dwi image', mandatory=True, argstr="-i %s")
    a2 = traits.File(exists=True, desc='sc image', mandatory=True, argstr="-d %s")
    a3 = traits.Int(exists=True, desc='identity', mandatory=True, argstr="-identity %s")
    a4 = traits.Str(exists=True, desc='interpolation', mandatory=True, argstr="-x %s")
    a5 = traits.Str(exists=True, desc='output', mandatory=True, argstr="-ofolder %s")

class register(CommandLine):
    input_spec = registerInputSpec
    _cmd = 'sct_register_multimodal'
 
#-----------------------------------------------------------------------------------------------------#
# Coregister T2w and DWI via MRTRIX3
#-----------------------------------------------------------------------------------------------------#
class T2wDWIRegMRTRIX3InputSpec(BaseInterfaceInputSpec):
    dwi_image = traits.File(exists=True, desc='DWI image', mandatory=True)
    anat_image = traits.File(exists=True, desc='T2w image', mandatory=True)
    type_chosen = traits.Str(exists=True, desc='type of transformation', mandatory=True)
    out_dir = traits.Str(exists=True, desc='output_dir', mandatory=True)

class T2wDWIRegMRTRIX3OutputSpec(TraitedSpec):
    out_transform = traits.File(exists=True, desc='transformation matrix')

class T2wDWIRegMRTRIX3(BaseInterface):
    input_spec = T2wDWIRegMRTRIX3InputSpec
    output_spec = T2wDWIRegMRTRIX3OutputSpec

    def _run_interface(self, runtime):

        dwi_name = self.inputs.dwi_image.split("/")[-1]
        self.dwi_name = dwi_name.split(".")[0]
        # define name for output file
        self.out_transform = os.path.join(self.inputs.out_dir, "%s_rigid_transformation.txt"%self.dwi_name)

        _mrregister = mrregister(a1 = self.inputs.type_chosen, a2 = self.inputs.type_chosen, a3 = self.out_transform, a4 = self.inputs.dwi_image, a5 = self.inputs.anat_image)
        _mrregister.run() 

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_transform'] = self.out_transform
        return outputs

class mrregisterInputSpec(CommandLineInputSpec):
    a1 = traits.Str(exists=True, desc='type', mandatory=True, argstr="-type %s")
    a2 = traits.Str(exists=True, desc='type', mandatory=True, argstr="-%s")
    a3 = traits.Str(exists=True, desc='output', mandatory=True, argstr="%s")
    a4 = traits.File(exists=True, desc='dwi image', mandatory=True, argstr="%s")
    a5 = traits.File(exists=True, desc='anat image', mandatory=True, argstr="%s")

class mrregister(CommandLine):
    input_spec = mrregisterInputSpec
    _cmd = 'mrregister -force'

#-----------------------------------------------------------------------------------------------------#
# Make the transformation matrix ITK-compatible
#-----------------------------------------------------------------------------------------------------#
class ITKcompatibleInputSpec(BaseInterfaceInputSpec):
    reference_image = traits.File(exists=True, desc='reference image', mandatory=True)
    input_transform = traits.File(exists=True, desc='input tranform', mandatory=True)
    operation = traits.Str(exists=True, desc='convert operation', mandatory=True)
    out_dir = traits.Str(exists=True, desc='output_dir', mandatory=True)

class ITKcompatibleOutputSpec(TraitedSpec):
    out_transform = traits.File(exists=True, desc='transformation matrix')

class ITKcompatible(BaseInterface):
    input_spec = ITKcompatibleInputSpec
    output_spec = ITKcompatibleOutputSpec

    def _run_interface(self, runtime):

        reference_name = self.inputs.reference_image.split("/")[-1]
        self.reference_name = reference_name.split(".")[0]
        # define name for output file
        self.out_transform = os.path.join(self.inputs.out_dir, "%s_rigid_transformation.txt"%self.reference_name)

        _compat = compat(a1 = self.inputs.input_transform, a2 = self.inputs.operation ,a3 = self.out_transform)
        _compat.run() 

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_transform'] = self.out_transform
        return outputs

class compatInputSpec(CommandLineInputSpec):
    a1 = traits.File(exists=True, desc='input transform', mandatory=True, argstr="%s")
    a2 = traits.Str(exists=True, desc='operation', mandatory=True, argstr="%s")
    a3 = traits.Str(exists=True, desc='output transform', mandatory=True, argstr="%s")

class compat(CommandLine):
    input_spec = compatInputSpec
    _cmd = 'transformconvert'

#-----------------------------------------------------------------------------------------------------#
# Coregister images via ANTs
#-----------------------------------------------------------------------------------------------------#
class T2wDWIRegANTsInputSpec(BaseInterfaceInputSpec):
    reference_image = traits.File(exists=True, desc='reference image', mandatory=True)
    moving_image = traits.File(exists=True, desc='moving image', mandatory=True)
    dimensionality = traits.Int(exists=True, desc='dimensionality transformation', mandatory=True)
    transformation = traits.Str(exists=True, desc='transformation matrix', mandatory=True)
    gradient_step = traits.Float(exists=True, desc='gradient step', mandatory=True)
    out_dir = traits.Str(exists=True, desc='output_dir', mandatory=True)

class T2wDWIRegANTsOutputSpec(TraitedSpec):
    out_coreg_moving_im = traits.File(exists=True, desc='registered moving image')
    out_coreg_moving_def = traits.File(exists=True, desc='deformation field source-to-target')
    out_coreg_reference_def = traits.File(exists=True, desc='deformation field target-to-source')
    out_fwtransform = traits.File(exists=True, desc='forward transformation matrix')

class T2wDWIRegANTs(BaseInterface):
    input_spec = T2wDWIRegANTsInputSpec
    output_spec = T2wDWIRegANTsOutputSpec

    def _run_interface(self, runtime):

        moving_im_name = self.inputs.moving_image.split("/")[-1]
        self.moving_im_name = moving_im_name.split(".")[0]
        reference_im_name = self.inputs.reference_image.split("/")[-1]
        self.reference_im_name = reference_im_name.split(".")[0]
        # define name for output file
        self.out_ = os.path.join(self.inputs.out_dir, "%s_"%self.moving_im_name)
        self.out_coreg_moving_im = os.path.join(self.inputs.out_dir, "%s_Warped.nii.gz"%self.moving_im_name)
        self.out_coreg_moving_def = os.path.join(self.inputs.out_dir, "%s_1Warp.nii.gz"%self.moving_im_name)
        self.out_coreg_reference_def = os.path.join(self.inputs.out_dir, "%s_1InverseWarp.nii.gz"%self.moving_im_name)
        self.out_fwtransform = os.path.join(self.inputs.out_dir, "%s_0GenericAffine.mat"%self.moving_im_name)
        _mrregister2 = mrregister2(a1 = self.inputs.dimensionality, a2 = self.out_, a3 = self.inputs.moving_image, a4 = self.inputs.reference_image, a5 = self.inputs.gradient_step, a6 = self.inputs.transformation)
        _mrregister2.run() 

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_coreg_moving_im'] = self.out_coreg_moving_im
        outputs['out_coreg_moving_def'] = self.out_coreg_moving_def
        outputs['out_coreg_reference_def'] = self.out_coreg_reference_def
        outputs['out_fwtransform'] = self.out_fwtransform
        return outputs

class mrregister2InputSpec(CommandLineInputSpec):
    a1 = traits.Int(exists=True, desc='dimensionality', mandatory=True, argstr="-d %s")
    a2 = traits.Str(exists=True, desc='output', mandatory=True, argstr="-o %s")
    a3 = traits.Str(exists=True, desc='moving image', mandatory=True, argstr="-m %s")
    a4 = traits.Str(exists=True, desc='reference image', mandatory=True, argstr="-f %s")
    a5 = traits.Float(exists=True, desc='gradient step', mandatory=True, argstr="-g %s")
    a6 = traits.Str(exists=True, desc='transformation matrix', mandatory=True, argstr="-t %s")

class mrregister2(CommandLine):
    input_spec = mrregister2InputSpec
    _cmd = 'antsRegistrationSyN.sh'

#-----------------------------------------------------------------------------------------------------#
# Apply ANTs transformation
#-----------------------------------------------------------------------------------------------------#
class ApplyTransformANTsInputSpec(BaseInterfaceInputSpec):
    reference_image = traits.File(exists=True, desc='reference image', mandatory=True)
    moving_image = traits.File(exists=True, desc='moving image', mandatory=True)
    affine_transform = traits.File(exists=True, desc='affine transformation matrix', mandatory=True)
    nonlinear_transform = traits.File(exists=True, desc='non-linear transformation matrix', mandatory=True)
    dimensionality = traits.Int(exists=True, desc='dimensionality transformation', mandatory=True)
    interpolation = traits.Str(exists=True, desc='interpolation', mandatory=True)
    out_dir = traits.Str(exists=True, desc='output_dir', mandatory=True)

class ApplyTransformANTsOutputSpec(TraitedSpec):
    output_image = traits.File(exists=True, desc='registered moving image')

class ApplyTransformANTs(BaseInterface):
    input_spec = ApplyTransformANTsInputSpec
    output_spec = ApplyTransformANTsOutputSpec

    def _run_interface(self, runtime):

        im_name = self.inputs.moving_image.split("/")[-1]
        self.im_name = im_name.split(".")[0]
        # define name for output file
        self.out_coregistered = os.path.join(self.inputs.out_dir, "%s_Warped.nii.gz"%self.im_name)

        _applyTransform = applyTransform(a1 = self.inputs.dimensionality, a2 = self.inputs.reference_image, a3 = self.inputs.moving_image,a4 = self.inputs.nonlinear_transform, a5 = self.inputs.affine_transform, a6 = self.inputs.interpolation, a7 = self.out_coregistered)
        _applyTransform.run() 

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['output_image'] = self.out_coregistered
        return outputs

class applyTransformInputSpec(CommandLineInputSpec):
    a1 = traits.Int(exists=True, desc='dimensionality', mandatory=True, argstr="-d %s")
    a2 = traits.File(exists=True, desc='target image', mandatory=True, argstr="-r %s")
    a3 = traits.File(exists=True, desc='moving image', mandatory=True, argstr="-i %s")
    a4 = traits.File(exists=True, desc='affine transformation matrix', mandatory=True, argstr="-t %s")
    a5 = traits.File(exists=True, desc='non-linear transformation matrix', mandatory=True, argstr="-t %s")
    a6 = traits.Str(exists=True, desc='interpolation', mandatory=True, argstr="-n %s")
    a7 = traits.Str(exists=True, desc='output', mandatory=True, argstr="-o %s")

class applyTransform(CommandLine):
    input_spec = applyTransformInputSpec
    _cmd = 'antsApplyTransforms'

#-----------------------------------------------------------------------------------------------------#
# Mask DWI
#-----------------------------------------------------------------------------------------------------#
class MaskDWIInputSpec(BaseInterfaceInputSpec):
    dwi_image = traits.File(exists=True, desc='DWI image', mandatory=True)
    sc_image = traits.File(exists=True, desc='SC registered image', mandatory=True)
    size = traits.Str(exists=True, desc='size', mandatory=True)
    out_dir = traits.Str(exists=True, desc='output_dir', mandatory=True)

class MaskDWIOutputSpec(TraitedSpec):
    out_mask = traits.File(exists=True, desc='spinal_cord_mask')

class MaskDWI(BaseInterface):
    input_spec = MaskDWIInputSpec
    output_spec = MaskDWIOutputSpec

    def _run_interface(self, runtime):

        dwi_name = self.inputs.dwi_image.split("/")[-1]
        self.dwi_name = dwi_name.split(".")[0]
        # define name for output file
        self.out_mask = os.path.join(self.inputs.out_dir, "mask_%s.nii.gz"%self.dwi_name)

        _mask = mask(a1 = self.inputs.dwi_image, a2 = self.inputs.sc_image, a3 = self.inputs.size)
        _mask.run() 

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_mask'] = self.out_mask
        return outputs

class maskInputSpec(CommandLineInputSpec):
    a1 = traits.File(exists=True, desc='dwi image', mandatory=True, argstr="-i %s")
    a2 = traits.File(exists=True, desc='sc image', mandatory=True, argstr="-p centerline,%s")
    a3 = traits.Str(exists=True, desc='size', mandatory=True, argstr="-size %s")

class mask(CommandLine):
    input_spec = maskInputSpec
    _cmd = 'sct_create_mask'

#-----------------------------------------------------------------------------------------------------#
# Crop DWI
#-----------------------------------------------------------------------------------------------------#
class cropDWIInputSpec(BaseInterfaceInputSpec):
    dwi_image = traits.File(exists=True, desc='DWI image', mandatory=True)
    mask_image = traits.File(exists=True, desc='SC mask image', mandatory=True)
    out_dir = traits.Str(exists=True, desc='output_dir', mandatory=True)

class cropDWIOutputSpec(TraitedSpec):
    out_cropped = traits.File(exists=True, desc='spinal_cord_cropped')

class cropDWI(BaseInterface):
    input_spec = cropDWIInputSpec
    output_spec = cropDWIOutputSpec

    def _run_interface(self, runtime):

        dwi_name = self.inputs.dwi_image.split("/")[-1]
        self.dwi_name = dwi_name.split(".")[0]
        # define name for output file
        self.out_crop = os.path.join(self.inputs.out_dir, "cropped_%s.nii.gz"%self.dwi_name)

        _crop = crop(a1 = self.inputs.dwi_image, a2 = self.inputs.mask_image, a3 = self.out_crop)
        _crop.run() 

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_cropped'] = self.out_crop
        return outputs

class cropInputSpec(CommandLineInputSpec):
    a1 = traits.File(exists=True, desc='dwi image', mandatory=True, argstr="-i %s")
    a2 = traits.File(exists=True, desc='mask image', mandatory=True, argstr="-m %s")
    a3 = traits.Str(exists=True, desc='output image', mandatory=True, argstr="-o %s")

class crop(CommandLine):
    input_spec = cropInputSpec
    _cmd = 'sct_crop_image'

#-----------------------------------------------------------------------------------------------------#
# MRTRIX3 GIBBS RINGING REMOVAL
#-----------------------------------------------------------------------------------------------------#
class MRTRIX3GibbsRemovalInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, desc='input image to correct for Gibbs ringing', mandatory=True)
    #axes = List(traits.Int, desc='list select the slice axes (default: 0,1 - i.e. x-y)')
    #nshifts = traits.Int(desc='value discretization of subpixel spacing (default: 20)')
    #minW = traits.Int(desc='value left border of window used for TV computation (default: 1)')
    #maxW = traits.Int(desc='value right border of window used for TV computation (default: 3)')
    # TODO add more arguments

class MRTRIX3GibbsRemovalOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='image corrected for Gibbs ringing artefacts')

class MRTRIX3GibbsRemoval(BaseInterface):
    input_spec = MRTRIX3GibbsRemovalInputSpec
    output_spec = MRTRIX3GibbsRemovalOutputSpec

    def _run_interface(self, runtime):
        # define output folders
        _, base, _ = split_filename(self.inputs.in_file)
        output_dir = os.path.abspath('')

        # define names for output files
        self.corrected_output = os.path.join(output_dir, base + '_gibbs.nii.gz')

        # initialise and run ROBEX
        _gibbs = GIBBS(input_file=self.inputs.in_file, output_file=self.corrected_output)
        _gibbs.run()

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = self.corrected_output
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            return self._gen_outfilename()
        return None

class GIBBSInputSpec(CommandLineInputSpec):
    input_file = traits.Str(exists=True, desc='input file to be corrected for rinning artefacts',
                            mandatory=True, argstr="%s", position=0)
    output_file = traits.Str(desc='corrected file',
                            mandatory=True, argstr="%s", position=1)

class GIBBS(CommandLine):
    input_spec = GIBBSInputSpec
    _cmd = 'mrdegibbs -force'

#-----------------------------------------------------------------------------------------------------#
# FSL Eddy correction
#-----------------------------------------------------------------------------------------------------#
class DWIFSLPREPROCInputSpec(BaseInterfaceInputSpec):
    in_file = traits.File(exists=True, desc='input image to correct for Eddy currents', mandatory=True)
    in_bval = traits.File(exists=True, desc='bval', mandatory=True)
    in_bvec = traits.File(exists=True, desc='bvec', mandatory=True)
    in_json = traits.File(exists=True, desc='json', mandatory=False)
    rpe = traits.Str(exists=True, desc='rpe type', mandatory=True)
    pe_dir = traits.Str(exists=True, desc='pe dir', mandatory=True)
    se_epi = traits.File(exists=True, desc='topup', mandatory=False)
    #extra_options = traits.Str(exists=True, desc='provide extra methods options', mandatory=True)
    out_dir = traits.Str(exists=True, desc='output_dir', mandatory=True)

class DWIFSLPREPROCOutputSpec(TraitedSpec):
    out_file = traits.File(exists=True, desc='image corrected')
    new_bvec = traits.File(exists=True, desc='corrected bvec')
    new_bval = traits.File(exists=True, desc='corrected bval')

class DWIFSLPREPROC(BaseInterface):
    input_spec = DWIFSLPREPROCInputSpec
    output_spec = DWIFSLPREPROCOutputSpec

    def _run_interface(self, runtime):
        # define output folders
        _, base, _ = split_filename(self.inputs.in_file)
        output_dir = os.path.abspath('')

        # define names for output files
        self.corrected_output = os.path.join(output_dir, base + '_eddy.nii.gz')
        self.new_bvec = os.path.join(output_dir, base + '_new.bvec')
        self.new_bval = os.path.join(output_dir, base + '_new.bval')

        # initialise and run ROBEX
        _eddy = EDDY(input_file=self.inputs.in_file, output_file=self.corrected_output, bvec = self.inputs.in_bvec, bval = self.inputs.in_bval,rpe=self.inputs.rpe, pe_dir = self.inputs.pe_dir,se_epi = self.inputs.se_epi, new_bvec = self.new_bvec, new_bval = self.new_bval, json = self.inputs.in_json)
        _eddy.run()

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = self.corrected_output
        outputs['new_bvec'] = self.new_bvec
        outputs['new_bval'] = self.new_bval
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            return self._gen_outfilename()
        return None

class EDDYInputSpec(CommandLineInputSpec):
    input_file = traits.File(exists=True, desc='input file to be corrected for rinning artefacts',mandatory=True, argstr="%s", position=0)
    output_file = traits.File(desc='corrected file', mandatory=True, argstr="%s", position=1)
    bvec = traits.File(exists=True, desc='bval', mandatory=True, argstr="-fslgrad %s", position=2)
    bval = traits.File(exists=True, desc='bvec', mandatory=True, argstr="%s", position=3)
    rpe = traits.Str(exists=True, desc='rpe type', mandatory=True, argstr="%s", position=4)
    pe_dir = traits.Str(exists=True, desc='pe dir', mandatory=True, argstr="-pe_dir %s", position=5)
    se_epi = traits.Str(exists=True, desc='topup', mandatory=False, argstr="-se_epi %s", position=6)
    new_bvec = traits.Str(exists=True, desc='new bvec', mandatory=True, argstr="-export_grad_fsl %s", position=7)
    new_bval = traits.Str(exists=True, desc='new bval', mandatory=True, argstr="%s", position=8)
    json = traits.Str(exists=True, desc='json', mandatory=False, argstr="-json_import %s", position=9)

class EDDY(CommandLine):
    input_spec = EDDYInputSpec
    _cmd = 'dwifslpreproc -force'
    
#-----------------------------------------------------------------------------------------------------#
#  MOCO in DWI
#-----------------------------------------------------------------------------------------------------#
class mocoDWIInputSpec(BaseInterfaceInputSpec):
    dwi_image = traits.File(exists=True, desc='DWI image', mandatory=True)
    bvec = traits.File(exists=True, desc='bvec file', mandatory=True)
    out_dir = traits.Str(exists=True, desc='output_dir', mandatory=True)

class mocoDWIOutputSpec(TraitedSpec):
    out_corrected = traits.File(exists=True, desc='MOCO corrected DWI')
    out_corrected_mean = traits.File(exists=True, desc='mean MOCO corrected DWI')

class mocoDWI(BaseInterface):
    input_spec = mocoDWIInputSpec
    output_spec = mocoDWIOutputSpec

    def _run_interface(self, runtime):

        dwi_name = self.inputs.dwi_image.split("/")[-1]
        self.dwi_name = dwi_name.split(".")[0]
        # define name for output file
        self.out_moco = os.path.join(self.inputs.out_dir, "%s_moco.nii.gz"%self.dwi_name)
        self.out_moco_mean = os.path.join(self.inputs.out_dir, "%s_moco_dwi_mean.nii.gz"%self.dwi_name)

        _moco = moco(a1 = self.inputs.dwi_image, a2 = self.inputs.bvec, a3 = self.inputs.out_dir)
        _moco.run() 

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_corrected'] = self.out_moco
        outputs['out_corrected_mean'] = self.out_moco_mean
        return outputs

class mocoInputSpec(CommandLineInputSpec):
    a1 = traits.File(exists=True, desc='dwi image', mandatory=True, argstr="-i %s")
    a2 = traits.File(exists=True, desc='bvec', mandatory=True, argstr="-bvec %s")
    a3 = traits.Str(exists=True, desc='output image', mandatory=True, argstr="-ofolder %s")

class moco(CommandLine):
    input_spec = mocoInputSpec
    _cmd = 'sct_dmri_moco'

#-----------------------------------------------------------------------------------------------------#
#  FOD-DWI
#-----------------------------------------------------------------------------------------------------#
class FODInputSpec(BaseInterfaceInputSpec):
    dwi_image = traits.File(exists=True, desc='DWI image', mandatory=True)
    bval = traits.File(exists=True, desc='DWI bval file', mandatory=True)
    bvec = traits.File(exists=True, desc='DWI bvec file', mandatory=True)
    response_method = traits.Str(exists=True, desc='method for dwi2response', mandatory=True)
    fod_method = traits.Str(exists=True, desc='method for dwi2fod', mandatory=True)
    out_dir = traits.Str(exists=True, desc='output_dir', mandatory=True)

class FODOutputSpec(TraitedSpec):
    out_response = traits.File(exists=True, desc='response TXT')
    out_fod = traits.File(exists=True, desc='FOD-DWI image')

class FOD(BaseInterface):
    input_spec = FODInputSpec
    output_spec = FODOutputSpec

    def _run_interface(self, runtime):

        dwi_name = self.inputs.dwi_image.split("/")[-1]
        self.dwi_name = dwi_name.split(".")[0]
        # define name for output file
        self.out_response = os.path.join(self.inputs.out_dir, "%s_response.txt"%self.dwi_name)
        self.out_fod1 = os.path.join(self.inputs.out_dir, "%s_FOD.mif"%self.dwi_name)
        self.out_fod2 = os.path.join(self.inputs.out_dir, "%s_FOD.nii.gz"%self.dwi_name)

        _dwi2response = dwi2response(a1 = self.inputs.response_method, a2 = self.inputs.dwi_image,a3 = self.out_response, a4 = self.inputs.bvec, a5 = self.inputs.bval)
        _dwi2response.run() 
        _dwi2fod = dwi2fod(a1 = self.inputs.fod_method, a2 = self.inputs.dwi_image, a3 = self.out_response, a4 = self.out_fod1, a5 = self.inputs.bvec, a6 = self.inputs.bval)
        _dwi2fod.run() 
        _convert = convert(a1 = self.out_fod1, a2 = self.out_fod2)
        _convert.run() 

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_response'] = self.out_response
        outputs['out_fod'] = self.out_fod2
        return outputs

class dwi2responseInputSpec(CommandLineInputSpec):
    a1 = traits.Str(exists=True, desc='method', mandatory=True, argstr="%s")
    a2 = traits.File(exists=True, desc='DWI image', mandatory=True, argstr="%s")
    a3 = traits.Str(exists=True, desc='output', mandatory=True, argstr="%s")
    a4 = traits.File(exists=True, desc='DWI bvec', mandatory=True, argstr="-fslgrad %s")
    a5 = traits.File(exists=True, desc='DWI bval', mandatory=True, argstr="%s")

class dwi2fodInputSpec(CommandLineInputSpec):
    a1 = traits.Str(exists=True, desc='method', mandatory=True, argstr="%s")
    a2 = traits.File(exists=True, desc='DWI image', mandatory=True, argstr="%s")
    a3 = traits.File(exists=True, desc='response TXT', mandatory=True, argstr="%s")
    a4 = traits.Str(exists=True, desc='output', mandatory=True, argstr="%s")
    a5 = traits.File(exists=True, desc='DWI bvec', mandatory=True, argstr="-fslgrad %s")
    a6 = traits.File(exists=True, desc='DWI bval', mandatory=True, argstr="%s")

class convertInputSpec(CommandLineInputSpec):
    a1 = traits.File(exists=True, desc='DWI mif', mandatory=True, argstr="%s")
    a2 = traits.Str(exists=True, desc='DWI nifti', mandatory=True, argstr="%s")

class dwi2response(CommandLine):
    input_spec = dwi2responseInputSpec
    _cmd = 'dwi2response'

class dwi2fod(CommandLine):
    input_spec = dwi2fodInputSpec
    _cmd = 'dwi2fod'

class convert(CommandLine):
    input_spec = convertInputSpec
    _cmd = 'mrconvert'

#-----------------------------------------------------------------------------------------------------#
#  DWI to Tensor
#-----------------------------------------------------------------------------------------------------#
class DWI2TensorInputSpec(BaseInterfaceInputSpec):
    dwi_image = traits.File(exists=True, desc='DWI image', mandatory=True)
    b0 = traits.File(exists=True, desc='B0 file', mandatory=True)
    bvec = traits.File(exists=True, desc='bvec file', mandatory=True)
    bval = traits.File(exists=True, desc='bval file', mandatory=True)
    out_dir = traits.Str(exists=True, desc='output_dir', mandatory=True)

class DWI2TensorOutputSpec(TraitedSpec):
    out_tensor = traits.File(exists=True, desc='DWI tensor output')

class DWI2Tensor(BaseInterface):
    input_spec = DWI2TensorInputSpec
    output_spec = DWI2TensorOutputSpec

    def _run_interface(self, runtime):

        dwi_name = self.inputs.dwi_image.split("/")[-1]
        self.dwi_name = dwi_name.split(".")[0]
        # define name for output file
        self.out_tensor = os.path.join(self.inputs.out_dir, "%s_tensor.nii.gz"%self.dwi_name)

        _tensor = tensor(a1 = self.inputs.dwi_image, a2 = self.out_tensor, a3 = self.inputs.b0, a4 = self.inputs.bvec,a5 = self.inputs.bval)
        _tensor.run() 

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_tensor'] = self.out_tensor
        return outputs

class tensorInputSpec(CommandLineInputSpec):
    a1 = traits.File(exists=True, desc='dwi image', mandatory=True, argstr="%s")
    a2 = traits.Str(exists=True, desc='output image', mandatory=True, argstr="%s")
    a3 = traits.File(exists=True, desc='b0', mandatory=True, argstr="-b0 %s")
    a4 = traits.File(exists=True, desc='bvec', mandatory=True, argstr="-fslgrad %s")
    a5 = traits.File(exists=True, desc='bval', mandatory=True, argstr="%s")

class tensor(CommandLine):
    input_spec = tensorInputSpec
    _cmd = 'dwi2tensor -force'

#-----------------------------------------------------------------------------------------------------#
#  DWI Tensor to DWI metrics
#-----------------------------------------------------------------------------------------------------#
class Tensor2MetricInputSpec(BaseInterfaceInputSpec):
    dwi_tensor = traits.File(exists=True, desc='DT tensor', mandatory=True)
    out_dir = traits.Str(exists=True, desc='output_dir', mandatory=True)

class Tensor2MetricOutputSpec(TraitedSpec):
    out_fa = traits.File(exists=True, desc='extracted FA')
    out_md = traits.File(exists=True, desc='extracted MD')
    out_ad = traits.File(exists=True, desc='extracted AD')
    out_rd = traits.File(exists=True, desc='extracted RD')

class Tensor2Metric(BaseInterface):
    input_spec = Tensor2MetricInputSpec
    output_spec = Tensor2MetricOutputSpec

    def _run_interface(self, runtime):

        dwi_name = self.inputs.dwi_tensor.split("/")[-1]
        self.dwi_name = dwi_name.split(".")[0]
        # define name for output file
        self.out_fa = os.path.join(self.inputs.out_dir, "%s_FA.nii.gz"%self.dwi_name)
        self.out_md = os.path.join(self.inputs.out_dir, "%s_MD.nii.gz"%self.dwi_name)
        self.out_ad = os.path.join(self.inputs.out_dir, "%s_AD.nii.gz"%self.dwi_name)
        self.out_rd = os.path.join(self.inputs.out_dir, "%s_RD.nii.gz"%self.dwi_name)

        _extract_metric = extract_metric(a1 = self.out_md,a2 = self.out_fa,a3 = self.out_ad, a4 = self.out_rd, a5 = self.inputs.dwi_tensor)
        _extract_metric.run() 

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_md'] = self.out_md
        outputs['out_fa'] = self.out_fa
        outputs['out_ad'] = self.out_ad
        outputs['out_rd'] = self.out_rd
        return outputs

class extract_metricInputSpec(CommandLineInputSpec):
    a1 = traits.Str(exists=True, desc='MD image', mandatory=True, argstr="-adc %s")
    a2 = traits.Str(exists=True, desc='FA image', mandatory=True, argstr="-fa %s")
    a3 = traits.Str(exists=True, desc='AD image', mandatory=True, argstr="-ad %s")
    a4 = traits.Str(exists=True, desc='RD image', mandatory=True, argstr="-rd %s")
    a5 = traits.File(exists=True, desc='dwi tensor', mandatory=True, argstr="%s")
    
class extract_metric(CommandLine):
    input_spec = extract_metricInputSpec
    _cmd = 'tensor2metric -force'

#-----------------------------------------------------------------------------------------------------#
#  QC in DWI
#-----------------------------------------------------------------------------------------------------#
class qcDWIInputSpec(BaseInterfaceInputSpec):
    dwi_image_orig = traits.File(exists=True, desc='original DWI image', mandatory=True)
    dwi_image_corrected = traits.File(exists=True, desc='corrected DWI image', mandatory=True)
    dwi_image_seg = traits.File(exists=True, desc='spinal cord segmentation', mandatory=True)
    step = traits.Str(exists=True, desc='step for QC', mandatory=True)
    out_dir = traits.Str(exists=True, desc='output_dir', mandatory=True)

class qcDWIOutputSpec(TraitedSpec):
    out_folder = traits.Directory(exists=True, desc='folder with QC images')

class qcDWI(BaseInterface):
    input_spec = qcDWIInputSpec
    output_spec = qcDWIOutputSpec

    def _run_interface(self, runtime):

        self.out_folder = self.inputs.out_dir
        _qc = qc(a1 = self.inputs.dwi_image_orig, a2 = self.inputs.dwi_image_corrected, a3 = self.inputs.dwi_image_seg, a4 = self.inputs.step, a5 = self.inputs.out_dir)
        _qc.run() 

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_folder'] = self.out_folder
        return outputs

class qcInputSpec(CommandLineInputSpec):
    a1 = traits.File(exists=True, desc='original dwi image', mandatory=True, argstr="-i %s")
    a2 = traits.File(exists=True, desc='corrected dwi image', mandatory=True, argstr="-d %s")
    a3 = traits.File(exists=True, desc='spinal cord segmentation', mandatory=True, argstr="-s %s")
    a4 = traits.Str(exists=True, desc='step for QC', mandatory=True, argstr="-p %s")
    a5 = traits.Directory(exists=True, desc='output folder', mandatory=True, argstr="-qc %s")

class qc(CommandLine):
    input_spec = qcInputSpec
    _cmd = 'sct_qc'

#-----------------------------------------------------------------------------------------------------#
#  dMRI computes DTI
#-----------------------------------------------------------------------------------------------------#
class DMRIcomputeDTIInputSpec(BaseInterfaceInputSpec):
    dwi_image = traits.File(exists=True, desc='DWI image', mandatory=True)
    bval = traits.File(exists=True, desc='bval file', mandatory=True)
    bvec = traits.File(exists=True, desc='bvec file', mandatory=True)
    method = traits.Str(exists=True, desc='method used', mandatory=True)
    out_dir = traits.Str(exists=True, desc='output_dir', mandatory=True)

class DMRIcomputeDTIOutputSpec(TraitedSpec):
    out_fa = traits.File(exists=True, desc='computed FA')
    out_md = traits.File(exists=True, desc='computed MD')
    out_ad = traits.File(exists=True, desc='computed AD')
    out_rd = traits.File(exists=True, desc='computed RD')

class DMRIcomputeDTI(BaseInterface):
    input_spec = DMRIcomputeDTIInputSpec
    output_spec = DMRIcomputeDTIOutputSpec

    def _run_interface(self, runtime):

        dwi_name = self.inputs.dwi_image.split("/")[-1]
        self.dwi_name = dwi_name.split(".")[0]
        # define name for output file
        self.out_prefix = os.path.join(self.inputs.out_dir, "%s_sct_dmri_compute_"%self.dwi_name)
        self.out_fa = os.path.join(self.inputs.out_dir, "%s_sct_dmri_compute_FA.nii.gz"%self.dwi_name)
        self.out_md = os.path.join(self.inputs.out_dir, "%s_sct_dmri_compute_MD.nii.gz"%self.dwi_name)
        self.out_ad = os.path.join(self.inputs.out_dir, "%s_sct_dmri_compute_AD.nii.gz"%self.dwi_name)
        self.out_rd = os.path.join(self.inputs.out_dir, "%s_sct_dmri_compute_RD.nii.gz"%self.dwi_name)

        _compute = compute(a1 = self.inputs.dwi_image, a2 = self.inputs.bval, a3 = self.inputs.bvec, a4 = self.inputs.method, a5 = self.out_prefix)
        _compute.run() 

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_fa'] = self.out_fa
        outputs['out_md'] = self.out_md
        outputs['out_ad'] = self.out_ad
        outputs['out_rd'] = self.out_rd
        return outputs

class computeInputSpec(CommandLineInputSpec):
    a1 = traits.File(exists=True, desc='dwi image', mandatory=True, argstr="-i %s")
    a2 = traits.File(exists=True, desc='bval', mandatory=True, argstr="-bval %s")
    a3 = traits.File(exists=True, desc='bvec', mandatory=True, argstr="-bvec %s")
    a4 = traits.Str(exists=True, desc='method used', mandatory=True, argstr="-method %s")
    a5 = traits.Str(exists=True, desc='output prefix', mandatory=True, argstr="-o %s")

class compute(CommandLine):
    input_spec = computeInputSpec
    _cmd = 'sct_dmri_compute_dti'

#-----------------------------------------------------------------------------------------------------#
#  compute FA within right and left lateral corticospinal tracts
#-----------------------------------------------------------------------------------------------------#
class RightLeftFAInputSpec(BaseInterfaceInputSpec):
    fa = traits.File(exists=True, desc='DWI image', mandatory=True)
    slice_range = traits.Str(exists=True, desc='range that the slices should span', mandatory=True)
    label_IDs = traits.Str(exists=True, desc='label IDs', mandatory=True)
    method = traits.Str(exists=True, desc='method used', mandatory=False)
    out_dir = traits.Str(exists=True, desc='output_dir', mandatory=True)

class RightLeftFAOutputSpec(TraitedSpec):
    out_fa_csv = traits.File(exists=True, desc='computed FA metrics')

class RightLeftFA(BaseInterface):
    input_spec = RightLeftFAInputSpec
    output_spec = RightLeftFAOutputSpec

    def _run_interface(self, runtime):

        fa = self.inputs.fa.split("/")[-1]
        self.fa_name = fa.split(".")[0]
        # define name for output file
        self.out_fa_csv = os.path.join(self.inputs.out_dir, "%s_in_cst.csv"%self.dwi_name)

        _extract = extract(a1 = self.inputs.fa, a2 = self.inputs.slice_range, a3 = self.inputs.method,a4 = self.inputs.label_IDs, a5 = self.out_fa_csv)
        _extract.run() 

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_fa_csv'] = self.out_fa_csv
        return outputs

class extractInputSpec(CommandLineInputSpec):
    a1 = traits.File(exists=True, desc='FA', mandatory=True, argstr="-i %s")
    a2 = traits.Str(exists=True, desc='slice range', mandatory=True, argstr="-z %s")
    a3 = traits.Str(exists=True, desc='method used', mandatory=True, argstr="-method %s")
    a4 = traits.Str(exists=True, desc='label IDs', mandatory=False, argstr="-l %s")
    a5 = traits.Str(exists=True, desc='output prefix', mandatory=True, argstr="-o %s")

class extract(CommandLine):
    input_spec = extractInputSpec
    _cmd = 'sct_extract_metric'

#-----------------------------------------------------------------------------------------------------#
# COMPUTE AXIAL AND RADIAL DIFFUSIVITY MAPS
#-----------------------------------------------------------------------------------------------------#
class RDComputeInputSpec(BaseInterfaceInputSpec):
    in_l2 = traits.File(exists=True, desc='L2 volume', mandatory=True)
    in_l3 = traits.File(exists=True, desc='L3 volume', mandatory=True)
    in_mask = traits.File(exists=True, desc='brain mask', mandatory=True)

class RDComputeOutputSpec(TraitedSpec):
    out_rd = traits.File(exists=True, desc='RD map')

class RDCompute(BaseInterface):
    input_spec = RDComputeInputSpec
    output_spec = RDComputeOutputSpec

    def _run_interface(self, runtime):
        _, base, _ = split_filename(self.inputs.in_l2)
        output_dir = os.path.abspath('')
        l2 = self.inputs.in_l2
        l3 = self.inputs.in_l3
        mask = self.inputs.in_mask
        l2_load = nb.load(l2)
        l2_img = l2_load.get_fdata()
        l3_load = nb.load(l3)
        l3_img = l3_load.get_fdata()
        mask_load = nb.load(mask)
        mask_img = mask_load.get_fdata()
        rd_img = np.copy(mask_img)
        self.out_rd = os.path.join(output_dir, 'dtifitWLS_RD.nii.gz')
        for k,j,i in product(range(l2_img.shape[0]),range(l2_img.shape[1]),range(l2_img.shape[2])):
            if mask_img[k,j,i] > 0.0:
               rd_img[k,j,i] = (l2_img[k,j,i] + l3_img[k,j,i])/2

        RD_file = nb.Nifti1Image(rd_img, l2_load.affine, l2_load.header)
        nb.save(RD_file, self.out_rd)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_rd'] = self.out_rd

        return outputs
    
#-----------------------------------------------------------------------------------------------------#
# DKI FIT
#-----------------------------------------------------------------------------------------------------#
from dipy.reconst.dki import DiffusionKurtosisModel
from dipy.io import read_bvals_bvecs
from dipy.data import gradient_table
from scipy.ndimage.filters import gaussian_filter


class DKIfitInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, desc='DKI volume to be analysed', mandatory=True)
    in_bvals = File(exists=True, desc='corresponding b-values ', mandatory=True)
    in_bvecs = File(exists=True, desc='corresponding b-vectors', mandatory=True)
    in_mask = File(exists=True, desc='brain mask', mandatory=True)
    in_smooth = traits.Float(exists=True, desc='smoothing factor', mandatory=False)

class DKIfitOutputSpec(TraitedSpec):
    out_fa = File(exists=True, desc='FA Gap', mandatory=True)
    out_md = File(exists=True, desc='MD map', mandatory=True)
    out_mk = File(exists=True, desc='MK map', mandatory=True)
    out_rd = File(exists=True, desc='RD map', mandatory=True)
    out_ad = File(exists=True, desc='AD map', mandatory=True) 
    out_rk = File(exists=True, desc='RK map', mandatory=True) 
    out_ak = File(exists=True, desc='AK map', mandatory=True) 

class DKIfit(BaseInterface):
    input_spec = DKIfitInputSpec
    output_spec = DKIfitOutputSpec

    def _run_interface(self, runtime):
        # define output folder
        output_dir = os.path.abspath('')

        # define names for output files 
        self.fa = os.path.join(output_dir, 'dkifit_FA.nii.gz')
        self.md = os.path.join(output_dir, 'dkifit_MD.nii.gz')
        self.mk = os.path.join(output_dir, 'dkifit_MK.nii.gz')
        self.ad = os.path.join(output_dir, 'dkifit_AD.nii.gz')
        self.rd = os.path.join(output_dir, 'dkifit_RD.nii.gz')
        self.ak = os.path.join(output_dir, 'dkifit_AK.nii.gz')
        self.rk = os.path.join(output_dir, 'dkifit_RK.nii.gz')
    
        # read b-values and b-vectors
        bvals, bvecs = read_bvals_bvecs(self.inputs.in_bvals, self.inputs.in_bvecs)
        gtab = gradient_table(bvals, bvecs)
        
        # load file and mask
        dki_proxy = nb.load(self.inputs.in_file)
        dki_data = dki_proxy.get_fdata()
        dki_mask_proxy = nb.load(self.inputs.in_mask)
        dki_mask = dki_mask_proxy.get_fdata()
        
       
        dkimodel = DiffusionKurtosisModel(gtab, fit_method="WLS")
        dkifit = dkimodel.fit(dki_data, dki_mask)
        
        dki_fa = nb.Nifti1Image(dkifit.fa, dki_proxy.affine, dki_proxy.header)
        dki_md = nb.Nifti1Image(dkifit.md, dki_proxy.affine, dki_proxy.header)
        dki_ad = nb.Nifti1Image(dkifit.ad, dki_proxy.affine, dki_proxy.header)
        dki_rd = nb.Nifti1Image(dkifit.rd, dki_proxy.affine, dki_proxy.header)
        dki_mk = nb.Nifti1Image(dkifit.mk(0,3), dki_proxy.affine, dki_proxy.header)
        dki_ak = nb.Nifti1Image(dkifit.ak(0,3), dki_proxy.affine, dki_proxy.header)
        dki_rk = nb.Nifti1Image(dkifit.rk(0,3), dki_proxy.affine, dki_proxy.header)
        nb.save(dki_fa, self.fa)
        nb.save(dki_md, self.md)
        nb.save(dki_mk, self.mk)
        nb.save(dki_ad, self.ad)
        nb.save(dki_rd, self.rd)
        nb.save(dki_ak, self.ak)
        nb.save(dki_rk, self.rk)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_fa'] = self.fa
        outputs['out_md'] = self.md
        outputs['out_mk'] = self.mk
        outputs['out_rk'] = self.rk
        outputs['out_ak'] = self.ak
        outputs['out_ad'] = self.ad
        outputs['out_rd'] = self.rd
        return outputs

    def _gen_filename(self, name):
        if name == 'out_fa':
            return self._gen_outfilename()
        return None

#-----------------------------------------------------------------------------------------------------#
# FREE WATER ELIMINATION
#-----------------------------------------------------------------------------------------------------#
class FreeWaterEliminationInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, desc='dwi volume for dti fitting', mandatory=True)
    in_bval = traits.Str(exists=True, desc='bvals', mandatory=True)
    in_bvec = traits.Str(exists=True, desc='bvecs', mandatory=True)
    in_mask = File(exists=True, desc='brain mask', mandatory=True)
    threshold = traits.Float(desc='threshold for artifact removal')

class FreeWaterEliminationOutputSpec(TraitedSpec):
    out_FA = File(exists=True, desc='fractional anisotropy map with free water elimination')
    out_MD = File(exists=True, desc='mean diffusivity map with free water elimination')
    out_FW = File(exists=True, desc='free water contamination map')
    out_eVecs = File(exists=True, desc='DWI vectors with free water elimination')
    out_eVals = File(exists=True, desc='DWI eigen values with free water elimination')
    out_FA_thresholded = File(exists=True, desc='fractional anisotropy map with free water elimination thresholded')
    out_fwmask = File(exists=True, desc='threshold freewater map mask')
    out_mask = File(exists=True, desc='threshold freewater map mask')

class FreeWaterElimination(BaseInterface):
    input_spec = FreeWaterEliminationInputSpec
    output_spec = FreeWaterEliminationOutputSpec

    def _run_interface(self, runtime):

        # load image and get predefined slices
        imgFile = nb.load(self.inputs.in_file)
        img = imgFile.get_fdata()
        #_, base, _ = split_filename(self.inputs.in_file)

        mask = nb.load(self.inputs.in_mask).get_fdata()

        if self.inputs.in_bvec[-6:] == '.bvecs':
            in_bvec = self.inputs.in_bvec[:-1]
        elif self.inputs.in_bvec[-19:] == '.eddy_rotated_bvecs':
            dst = self.inputs.in_bvec[-19] + '.bvec'
            shutil.copy(self.inputs.in_bvec, dst)
            in_bvec = dst
        else:
            in_bvec = self.inputs.in_bvec

        if self.inputs.in_bval[-6:] == '.bvals':
            in_bval = self.inputs.in_bval[:-1]
        else:
            in_bval = self.inputs.in_bval

        if isdefined(self.inputs.threshold):
            threshold = self.inputs.threshold
        else:
            threshold = 0.7

        # load bvecs
        gtab = gradient_table(in_bval, in_bvec)

        # fit model
        fwdtimodel = fwdti.FreeWaterTensorModel(gtab)
        fwdtifit = fwdtimodel.fit(img, mask=mask)

        FA = fwdtifit.fa
        MD = fwdtifit.md
        FW = fwdtifit.f # free water contamination map
        evals = fwdtifit.evals
        evecs =  fwdtifit.evecs

        # threhsold FA map to remove artefacts of regions associated to voxels with
        # high water volume fraction (i.e. voxels containing basically CSF).
        thFA = copy.copy(FA)
        thFA[FW > threshold] = 0
        fwMask = np.where(FW>threshold,1, 0)
        nfwMask = np.abs(fwMask-1)

          # save images to folder
        faFile = nb.Nifti1Image(FA, imgFile.affine, imgFile.header)
        nb.save(faFile, 'fwFA.nii')
        self.FA = os.path.abspath('fwFA.nii')

        mdFile = nb.Nifti1Image(MD, imgFile.affine, imgFile.header)
        nb.save(mdFile, 'fwMD.nii')
        self.MD = os.path.abspath('fwMD.nii')

        fwFile = nb.Nifti1Image(FW, imgFile.affine, imgFile.header)
        nb.save(fwFile,'FW.nii')
        self.FW = os.path.abspath('FW.nii')

        evalsFile = nb.Nifti1Image(evals, imgFile.affine, imgFile.header)
        nb.save(evalsFile,'eVals.nii')
        self.eVals = os.path.abspath('eVals.nii')

        evecsFile = nb.Nifti1Image(evecs, imgFile.affine, imgFile.header)
        nb.save(evecsFile,'eVecs.nii')
        self.eVecs = os.path.abspath('eVecs.nii')

        thFaFile = nb.Nifti1Image(thFA, imgFile.affine, imgFile.header)
        nb.save(thFaFile, 'fwFA_thresholded_'+ str(threshold) +'.nii')
        self.thFA = os.path.abspath('fwFA_thresholded_'+ str(threshold) +'.nii')

        fwMaskFile = nb.Nifti1Image(fwMask, imgFile.affine, imgFile.header)
        nb.save(fwMaskFile, 'freeWater_mask_thr_'+ str(threshold) +'.nii')
        self.fwMask = os.path.abspath('freeWater_mask_thr_'+ str(threshold) +'.nii')

        nfwMaskFile = nb.Nifti1Image(nfwMask, imgFile.affine, imgFile.header)
        nb.save(nfwMaskFile, 'negativeFreeWater_mask_thr_'+ str(threshold) +'.nii')
        self.nfwMask = os.path.abspath('negativeFreeWater_mask_thr_'+ str(threshold) +'.nii')

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_FA'] = self.FA
        outputs['out_MD'] = self.MD
        outputs['out_FW'] = self.FW
        outputs['out_eVals'] = self.eVals
        outputs['out_eVecs'] = self.eVecs
        outputs['out_FA_thresholded'] = self.thFA
        outputs['out_fwmask'] = self.fwMask
        outputs['out_mask'] = self.nfwMask

        return outputs

    def _gen_filename(self, name):
        if name == 'out_FA':
            return self._gen_outfilename()

        return None


#-----------------------------------------------------------------------------------------------------#
# Intravoxel incoherent motion
#-----------------------------------------------------------------------------------------------------#
class IntraVoxelIncoherentMotionInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, desc='dwi volume for dti fitting', mandatory=True)
    in_bval = traits.Str(exists=True, desc='bvals', mandatory=True)
    in_bvec = traits.Str(exists=True, desc='bvecs', mandatory=True)

class IntraVoxelIncoherentMotionOutputSpec(TraitedSpec):
    out_predicted_S0 = File(exists=True, desc='')
    out_perfusion_fraction = File(exists=True, desc='')
    out_perfusion_coeff = File(exists=True, desc='')
    out_diffusion_coeff = File(exists=True, desc='')

class IntraVoxelIncoherentMotion(BaseInterface):
    input_spec = IntraVoxelIncoherentMotionInputSpec
    output_spec = IntraVoxelIncoherentMotionOutputSpec

    def _run_interface(self, runtime):

        # load image and get predefined slices
        imgFile = nb.load(self.inputs.in_file)
        img = imgFile.get_fdata()

##       Initially in_bvec and in_bval were created as follows:
#        if self.inputs.in_bvec[-1] == 's':
#            in_bvec = self.inputs.in_bvec[:-1]
#        else:
#            in_bvec = self.inputs.in_bvec
#
#        if self.inputs.in_bval[-1] == 's':
#            in_bval = self.inputs.in_bval[:-1]
#        else:
#            in_bval = self.inputs.in_bval

##       But then I tried to create them as follows too:
#        if self.inputs.in_bvec[-6:] == '.bvecs':
#            in_bvec = self.inputs.in_bvec[:-1]
#        elif self.inputs.in_bvec[-19:] == '.eddy_rotated_bvecs':
#            dst = self.inputs.in_bvec[-19] + '.bvec'
#            shutil.copy(self.inputs.in_bvec, dst)
#            in_bvec = dst
#        else:
#            in_bvec = self.inputs.in_bvec
#
#        if self.inputs.in_bval[-6:] == '.bvals':
#            in_bval = self.inputs.in_bval[:-1]
#        else:
#            in_bval = self.inputs.in_bval

##      Another option is to use the bval and bvec you gave as an input:
#        in_bval = self.inputs.in_bval
#        in_bvec = self.inputs.in_bvec

        # load bvecs
        gtab = gradient_table(self.inputs.in_bval, self.inputs.in_bvec)

        # fit model
        ivimmodel = IvimModel(gtab)
        ivimfit = ivimmodel.fit(img)


        S0 = ivimfit.S0_predicted
        PF = ivimfit.perfusion_fraction
        perfCoeff = ivimfit.D_star
        diffCoeff = ivimfit.D

        # save images to folder
        s0File = nb.Nifti1Image(S0, imgFile.affine, imgFile.header)
        nb.save(s0File, 'predicted_S0.nii')
        self.S0 = os.path.abspath('predicted_S0.nii')

        pfFile = nb.Nifti1Image(PF, imgFile.affine, imgFile.header)
        nb.save(pfFile, 'perfusion_fraction.nii')
        self.PF = os.path.abspath('perfusion_fraction.nii')

        perfCoeffFile = nb.Nifti1Image(perfCoeff, imgFile.affine, imgFile.header)
        nb.save(perfCoeffFile, 'perfusion_coeff.nii')
        self.perfCoeff = os.path.abspath('perfusion_coeff.nii')


        diffCoeffFile = nb.Nifti1Image(diffCoeff, imgFile.affine, imgFile.header)
        nb.save(diffCoeffFile, 'diffusion_coeff.nii')
        self.diffCoeff = os.path.abspath('diffusion_coeff.nii')


        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_predicted_S0'] = self.S0
        outputs['out_perfusion_fraction'] = self.PF
        outputs['out_perfusion_coeff'] = self.perfCoeff
        outputs['out_diffusion_coeff'] = self.diffCoeff
        return outputs

    def _gen_filename(self, name):
        if name == 'out_predicted_S0':
            return self._gen_outfilename()

        return None

#-----------------------------------------------------------------------------------------------------#
# APPLY MEDIAN FILTER ON MK
#-----------------------------------------------------------------------------------------------------#
class ApplyMedianOnMKInputSpec(BaseInterfaceInputSpec):
    in_mk = traits.File(exists=True, desc='MK volume', mandatory=True)

class ApplyMedianOnMKOutputSpec(TraitedSpec):
    medfilt_mk = traits.File(exists=True, desc='MK map')

class ApplyMedianOnMK(BaseInterface):
    input_spec = ApplyMedianOnMKInputSpec
    output_spec = ApplyMedianOnMKOutputSpec

    def _run_interface(self, runtime):
        _, base, _ = split_filename(self.inputs.in_mk)
        output_dir = os.path.abspath('')
        self.medfilt_mk = os.path.join(output_dir, 'dkifit_MK_medfilt.nii.gz')
        mk = self.inputs.in_mk
        mk_load = nb.load(mk)
        mk_img = mk_load.get_fdata()
        mk_filtered = signal.medfilt(mk_img)
        ind = mk_img < (0.8*mk_filtered)
        mk_final = np.copy(mk_img)
        mk_final[ind] = mk_filtered[ind]
        mk_file = nb.Nifti1Image(mk_final, mk_load.affine, mk_load.header)
        nb.save(mk_file, self.medfilt_mk)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['medfilt_mk'] = self.medfilt_mk

        return outputs

#-----------------------------------------------------------------------------------------------------#
# APPLY MEDIAN FILTER ON AK
#-----------------------------------------------------------------------------------------------------#
class ApplyMedianOnAKInputSpec(BaseInterfaceInputSpec):
    in_ak = traits.File(exists=True, desc='AK volume', mandatory=True)

class ApplyMedianOnAKOutputSpec(TraitedSpec):
    medfilt_ak = traits.File(exists=True, desc='AK map')

class ApplyMedianOnAK(BaseInterface):
    input_spec = ApplyMedianOnAKInputSpec
    output_spec = ApplyMedianOnAKOutputSpec

    def _run_interface(self, runtime):
        _, base, _ = split_filename(self.inputs.in_ak)
        output_dir = os.path.abspath('')
        self.medfilt_ak = os.path.join(output_dir, 'dkifit_AK_medfilt.nii.gz')
        ak = self.inputs.in_ak
        ak_load = nb.load(ak)
        ak_img = ak_load.get_fdata()
        ak_filtered = signal.medfilt(ak_img)
        ind = ak_img < (0.8*ak_filtered)
        ak_final = np.copy(ak_img)
        ak_final[ind] = ak_filtered[ind]
        ak_file = nb.Nifti1Image(ak_final, ak_load.affine, ak_load.header)
        nb.save(ak_file, self.medfilt_ak)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['medfilt_ak'] = self.medfilt_ak

        return outputs

#-----------------------------------------------------------------------------------------------------#
# APPLY MEDIAN FILTER ON RK
#-----------------------------------------------------------------------------------------------------#
class ApplyMedianOnRKInputSpec(BaseInterfaceInputSpec):
    in_rk = traits.File(exists=True, desc='RK volume', mandatory=True)

class ApplyMedianOnRKOutputSpec(TraitedSpec):
    medfilt_rk = traits.File(exists=True, desc='RK map')

class ApplyMedianOnRK(BaseInterface):
    input_spec = ApplyMedianOnRKInputSpec
    output_spec = ApplyMedianOnRKOutputSpec

    def _run_interface(self, runtime):
        _, base, _ = split_filename(self.inputs.in_rk)
        output_dir = os.path.abspath('')
        self.medfilt_rk = os.path.join(output_dir, 'dkifit_RK_medfilt.nii.gz')
        rk = self.inputs.in_rk
        rk_load = nb.load(rk)
        rk_img = rk_load.get_fdata()
        rk_filtered = signal.medfilt(rk_img)
        ind = rk_img < (0.8*rk_filtered)
        rk_final = np.copy(rk_img)
        rk_final[ind] = rk_filtered[ind]
        rk_file = nb.Nifti1Image(rk_final, rk_load.affine, rk_load.header)
        nb.save(rk_file, self.medfilt_rk)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['medfilt_rk'] = self.medfilt_rk

        return outputs

#-----------------------------------------------------------------------------------------------------#
#  Tractography via MRTRIX3 - step 1
#-----------------------------------------------------------------------------------------------------#
class TractographyInputSpec(BaseInterfaceInputSpec):
    dwi_image = traits.File(exists=True, desc='DWI processed image', mandatory=True)
    seed_image = traits.File(exists=True, desc='NIFTI image with binary nerve-root seeds', mandatory=True)
    include_image = traits.File(exists=True, desc='NIFTI image with binary nerve-root seeds', mandatory=False)
    bval = traits.File(exists=True, desc='DWI bval file', mandatory=True)
    bvec = traits.File(exists=True, desc='DWI bvec file', mandatory=True)
    seeds = traits.Int(exists=True, desc='seeds', mandatory=True)
    select = traits.Int(exists=True, desc='select', mandatory=True)
    seed_cutoff = traits.Float(exists=True, desc='seed_cutoff', mandatory=True)
    cutoff = traits.Float(exists=True, desc='cutoff', mandatory=True)
    minlength = traits.Float(exists=True, desc='minlength', mandatory=True)
    maxlength = traits.Float(exists=True, desc='maxlength', mandatory=True)
    step = traits.Float(exists=True, desc='step', mandatory=True)
    algorithm = traits.Str(exists=True, desc='the algorithm to use', mandatory=True)
    out_dir = traits.Str(exists=True, desc='output_dir', mandatory=True)

class TractographyOutputSpec(TraitedSpec):
    out_tracto = traits.File(exists=True, desc='extracted tractography map')

class Tractography(BaseInterface):
    input_spec = TractographyInputSpec
    output_spec = TractographyOutputSpec

    def _run_interface(self, runtime):

        dwi_name = self.inputs.dwi_image.split("/")[-1]
        self.dwi_name = dwi_name.split(".")[0]
        self.name = "dwi-%s"%self.dwi_name
        seed_name = self.inputs.seed_image.split("/")[-1]
        self.seed_name = seed_name.split(".")[0]
        self.name += "_nr-%s_seeds-%s_select-%s"%(self.seed_name,self.inputs.seeds,self.inputs.select)
        if self.inputs.include_image:
           include_name = self.inputs.include_image.split("/")[-1]
           self.include_name = include_name.split(".")[0]
           self.name += "_include-%s"%self.include_name
        # define name for output file
        self.out_tracto = os.path.join(self.inputs.out_dir, "%s_tracto.tck"%(self.name))

        _compute_tractograhy = compute_tractography(a1 = self.inputs.seed_image,a2 = self.inputs.include_image, a3 = self.inputs.seeds,a4 = self.inputs.algorithm, a5= self.inputs.select, a6= self.inputs.seed_cutoff,a7= self.inputs.cutoff,a8= self.inputs.minlength,a9= self.inputs.maxlength,a10= self.inputs.step,a11 = self.inputs.bvec, a12 = self.inputs.bval,a13 = self.inputs.dwi_image, a14 = self.out_tracto)
        _compute_tractograhy.run() 

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_tracto'] = self.out_tracto
        return outputs

class compute_tractographyInputSpec(CommandLineInputSpec):
    a1 = traits.File(exists=True, desc='seeds image', mandatory=True, argstr="-seed_image %s")
    a2 = traits.File(exists=True, desc='include image', mandatory=False, argstr="-include %s -stop")
    a3 = traits.Int(exists=True, desc='seeds', mandatory=True, argstr="-seeds %s")
    a4 = traits.Str(exists=True, desc='algorithm', mandatory=True, argstr="-algorithm %s")
    a5 = traits.Int(exists=True, desc='select', mandatory=True, argstr="-select %s")
    a6 = traits.Float(exists=True, desc='seed_cutoff', mandatory=True, argstr="-seed_cutoff %s")
    a7 = traits.Float(exists=True, desc='cutoff', mandatory=True, argstr="-cutoff %s")
    a8 = traits.Float(exists=True, desc='minlength', mandatory=True, argstr="-minlength %s")
    a9 = traits.Float(exists=True, desc='maxlength', mandatory=True, argstr="-maxlength %s")
    a10 = traits.Float(exists=True, desc='step', mandatory=True, argstr="-step %s")
    a11 = traits.File(exists=True, desc='DWI bvec', mandatory=True, argstr="-fslgrad %s")
    a12 = traits.File(exists=True, desc='DWI bval', mandatory=True, argstr="%s")
    a13 = traits.File(exists=True, desc='DWI image', mandatory=True, argstr="%s")
    a14 = traits.Str(exists=True, desc='output file', mandatory=True, argstr="%s")
    
class compute_tractography(CommandLine):
    input_spec = compute_tractographyInputSpec
    _cmd = 'tckgen -force'

#-----------------------------------------------------------------------------------------------------#
#  Tractography via MRTRIX3 - step 2
#-----------------------------------------------------------------------------------------------------#
class TracteditInputSpec(BaseInterfaceInputSpec):
    init_out_image = traits.File(exists=True, desc='Initial tractography', mandatory=True)
    seed_image = traits.File(exists=True, desc='NIFTI image with binary nerve-root seeds', mandatory=True)
    end_image = traits.File(exists=True, desc='NIFTI image with binary nerve-root seeds', mandatory=False)
    out_dir = traits.Str(exists=True, desc='output_dir', mandatory=True)

class TracteditOutputSpec(TraitedSpec):
    out_tracto = traits.File(exists=True, desc='extracted tractography map')

class Tractedit(BaseInterface):
    input_spec = TracteditInputSpec
    output_spec = TracteditOutputSpec

    def _run_interface(self, runtime):

        out_name = self.inputs.init_out_image.split("/")[-1]
        self.out_name = out_name.split(".")[0]
        self.out_tracto = os.path.join(self.inputs.out_dir, "%s_FINAL.tck"%(self.out_name))

        _compute_tractedit = compute_tractedit(a1 = self.inputs.init_out_image,a2 = self.out_tracto,a3 = self.inputs.seed_image,a4 = self.inputs.end_image)
        _compute_tractedit.run() 

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_tracto'] = self.out_tracto
        return outputs

class compute_tracteditInputSpec(CommandLineInputSpec):
    a1 = traits.File(exists=True, desc='initial tractography', mandatory=True, argstr="%s")
    a2 = traits.Str(exists=True, desc='output file', mandatory=True, argstr="%s")
    a3 = traits.File(exists=True, desc='seed image', mandatory=False, argstr="-include %s")
    a4 = traits.File(exists=True, desc='end image', mandatory=False, argstr="-include %s")
    
class compute_tractedit(CommandLine):
    input_spec = compute_tracteditInputSpec
    _cmd = 'tckedit -force'
