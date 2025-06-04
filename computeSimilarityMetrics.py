#!/lustre/scratch/wbic-beta/sw742/app/anaconda2/bin/python
#
# Stefan Winzeck | sw742@cam.ac.uk | 01/2019

# IMPORT PYTHON INTERFACES
import sys, os, csv
import numpy as np
import nibabel as nib

def mutual_information(hgram):
     pxy = hgram / float(np.sum(hgram))
     px = np.sum(pxy, axis=1) # marginal for x over y
     py = np.sum(pxy, axis=0) # marginal for y over x
     px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
     # Now we can do the calculation using the pxy, px_py 2D arrays
     nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
     return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

def computeSimilarityMetrics(imageStr1, imageStr2, metric, maskStr):
    """Compares two images with given metric.
        If mask is given compares only within mask
        Order of images is inrelavant as availabel metrics are symmetric.
 
        INPUT
        imageStr1: path to first image file
        imageStr2: path to second image file
        metric:       similarity metric to compute between input images     
                      MAE = Mean Absolute Error  
                      MRE = Mean Reproducible error
                      MNAE = Mean Normalsied Absolute Error
                      MSE = Mean Square Error
                      RMSE = Root Mean Sqaure Error
                      SAD =  Sum of Absolute Differences
                      SSD = Sum of Squared Differences
                      CC = Cross Correlation
                      NCC = Normalised Cross Correlation
                      NMI = Normalised Mutual Information
                      ALL =  all the above
        mask:     path to mask image to compute metric only within mask
    """
    eps = np.finfo(float).eps

    # load images
    img1 = nib.load(imageStr1).get_fdata()
    img2 = nib.load(imageStr2).get_fdata()

    assert img1.shape==img2.shape, "Input images have different dimensions, must have same."

    # check for mask input, otherwise create mask
    if maskStr:
        mask = nib.load(maskStr).get_fdata()

        assert len(np.unique(mask))==2, "Input mask seems to be non-binary."
        # TODO assert mask values = 0 1
    
    else:
        mask = np.ones(img1.shape)

    # get mask indices
    ind = np.where(mask==1) 

    # compute similarity metric
    # Mean Absolute Error
    if metric=="MAE": 
        tmp = np.abs(img1-img2) 
        similarity = np.mean(tmp[ind])
    # Mean Reproducible Error
    elif metric=="MRE":
        tmp1 = np.abs(img1-img2)
        tmp2 = (img1 + img2) / 2. + eps
        tmp = np.mean(tmp1/tmp2)
        similarity = np.mean(tmp[ind])
    # Mean Normalised Absolute Error:
    elif metric=="MNAE":
        tmp = np.abs(img1-img2) / (img1+eps)
        similarity = np.mean(tmp[ind])
    # MeanSquare error
    elif metric=="MSE": 
        tmp = (img1-img2)**2
        similarity = np.mean(tmp[ind])
    # Root Mean Square Error
    elif  metric=="RMSE":
        tmp = (img1-img2)**2
        similarity = np.sqrt(np.mean(tmp[ind]))
    
    # Sum of Absolute Differences
    elif metric=="SAD":
        tmp = np.abs(img1-img2)
        similarity = np.sum(tmp[ind])
        
    # Sum of Squared Differences
    elif metric=="SSD":
        tmp = (img1-img2)**2
        similarity = np.sum(tmp[ind])
        
    # Cross Correlation
    elif metric=="CC":
        mean1 = np.mean(img1[ind]) 
        mean2 = np.mean(img2[ind]) 
        tmp = (img1 - mean1) * (img2 - mean2)
        similarity = np.mean(tmp[ind])
    
    # Normalised Cross Correlation
    elif metric=="NCC":
        mean1 = np.mean(img1[ind]) 
        mean2 = np.mean(img2[ind]) 
        std1 = np.std(img1[ind]) 
        std2 = np.std(img2[ind]) 
        tmp = (img1 - mean1) * (img2 - mean2)
        similarity = np.mean(tmp[ind]) / (std1 * std2)

    # Mutual Information
    elif metric=="NMI":
                SL1 = img1[ind].ravel()
                SL2 = img2[ind].ravel()
                hist2D,x,y = np.histogram2d(SL1,SL2,bins=300,density=True)
                similarity = mutual_information(hist2D)

    # compute all metrics
    elif metric=="ALL":
        tmp_abs = np.abs(img1-img2)
        tmp_sqr = (img1-img2)**2    
        tmp_mean = (img1+img2)/2. 
        tmp_normdiff = tmp_abs/(img1+eps)

        mean1 = np.mean(img1[ind])
        mean2 = np.mean(img2[ind]) 
        std1 = np.std(img1[ind])
        std2 = np.std(img2[ind])
        tmp_cc = (img1 - mean1) * (img2 - mean2)

        MAE = np.mean(tmp_abs[ind])
        MRE = np.mean(tmp_abs[ind] / (tmp_mean[ind] +eps)) 
        MNAE = np.mean(tmp_normdiff[ind])
        MSE = np.mean(tmp_sqr[ind])
        RMSE = np.sqrt(MSE)
        SAD = np.sum(tmp_abs[ind])
        SSD = np.sum(tmp_sqr[ind])
        CC = np.mean(tmp_cc[ind])
        NCC = CC / (std1 * std2)
        SL1 = img1[ind].ravel()
        SL2 = img2[ind].ravel()
        hist2D,x,y = np.histogram2d(SL1,SL2,bins=300,density=True)
        similarity = mutual_information(hist2D)

    else:
        print("Unknown metric <%s>. Use MAE, MRE, MNAE, MSE, RMSE, SAD, SSD, CC, NCC, NMI or ALL." % metric)
    if metric=="ALL":
        print("MAE = %.4f | MRE = %.4f | MNAE = %.4f | MSE = %.4f | RMSE = %.4f | SAD = %.4f | SSD = %.4f | CC = %.4f | NCC = %.4f | NMI = %.4f" % (MAE, MRE, MNAE, MSE, RMSE, SAD, SSD, CC, NCC, NMI))
    else:
        print("%s = %.4f" % (metric, similarity))
        return similarity

     
