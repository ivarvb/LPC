#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Ivar Vargas Belizario
# Copyright (c) 2021
# E-mail: ivar@usp.br

from os import stat
import numpy as np
import ujson
import SimpleITK as sitk

from shapely import geometry
import cv2
#import sys
import itk

class ROI:

    """
    input: titt color image
    output: binary image (mask)
    """
    @staticmethod
    def execute(self, image):

        mask = []
        
        # read image as grayscale

        # pre-processing
        
        # smoothing the image

        # countour extraction
        
        # countour growing
        return mask


def gaussianfilter(fileimage):
    colorimage = sitk.ReadImage(fileimage)
    #imshow(sitk.GetArrayFromImage(image));
    image = sitk.VectorMagnitude(colorimage)

    smooth = sitk.SmoothingRecursiveGaussianImageFilter()  
    smooth.SetSigma(1.0)
    #smooth.SetNormalizeAcrossScale(1)
    smoothedimage  = smooth.Execute(image)  
    smoothedimage = sitk.Normalize(smoothedimage)
    
    imagefilter = sitk.GetArrayFromImage(smoothedimage)
    #imagefilter = sitk.VectorMagnitude(imagefilter)

    #imagefilter = imagefilter.max() / (imagefilter.max() - imagefilter.min())
    imagefilter = (imagefilter)*255
    
    print("imagefilter", imagefilter)

    #Foreground, Background = 210, 0
    im_size = imagefilter.shape
    mask = np.zeros(im_size, dtype=int)
    #mask[np.where(imagefilter >= Foreground)] = Background
    #mask[np.where(imagefilter < Foreground)] = Foreground
    
    #out_x = np.where(x<=40,0, np.searchsorted([40,50,60,70,80,90], x)+3)

    #sitk.GetArrayFromImage()
    #sitk.WriteImage(sitk.GetImageFromArray(mask), 'pout.png', True)
    
    #cv2.imwrite('pout.jpg', mask)
    cv2.imwrite('pout.jpg', imagefilter)


def gaussianfilteritk(inputImage, outputImage):

    sigma = 1.0

    PixelType = itk.UC
    Dimension = 2

    ImageType = itk.Image[PixelType, Dimension]

    reader = itk.ImageFileReader[ImageType].New()
    reader.SetFileName(inputImage)

    smoothFilter = itk.SmoothingRecursiveGaussianImageFilter[
            ImageType,
            ImageType].New()
    smoothFilter.SetInput(reader.GetOutput())
    smoothFilter.SetSigma(sigma)

    writer = itk.ImageFileWriter[ImageType].New()
    writer.SetFileName(outputImage)
    writer.SetInput(smoothFilter.GetOutput())

    writer.Update()    




if __name__ == "__main__":        
    filein = "/mnt/sda6/software/frameworks/data/lha/dataset_3/images/45.tiff"
    fileout = "./outimgfilter.png"
    
    #image = cv2.imread(filein, cv2.IMREAD_GRAYSCALE)
    #print(image)
    
    #gaussianfilter(filein)
    gaussianfilteritk(filein, fileout)

    pass


