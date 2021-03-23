import matplotlib.pyplot as plt
import numpy as np
import os

import solaris.preproc.pipesegment as pipesegment
import solaris.preproc.image as image
import solaris.preproc.sar as sar
import solaris.preproc.optical as optical
import solaris.preproc.label as label

class MergeImages(pipesegment.PipeSegment):
    def __init__(self, im1_path, im2_path, output_path):
        super().__init__()
        load_im1 = image.LoadImage(im1_path)
        #resize_ms = image.Resize(600, 600)
        #color_ms = optical.RGBToHSV(rband=2, gband=1, bband=0)
        load_im2 = image.LoadImage(im2_path)
        stack = image.MergeToStack()
        save_output = image.SaveImage(output_path)
        self.feeder = (load_im1 + load_im2) * stack * save_output