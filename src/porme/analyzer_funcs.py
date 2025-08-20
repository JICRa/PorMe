# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 11:07:56 2022

@author: MPardo
"""
# %% defs
from copy import deepcopy as copy
from concurrent.futures import ThreadPoolExecutor as tpe
import functools
import warnings
import time
import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt


def kill_cv2():
    cv2.destroyAllWindows()


def kill_cv2_on_error(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            wrap_out = func(*args, **kwargs)
        except:
            kill_cv2()
            raise
        return wrap_out
    return wrapper
    cv2.destroyAllWindows()


def sorter(filename):
    """Helper function to sort filenames."""
    out = filename[filename.index("layer")+5:]
    out = out[:2]
    if out[1] == "-":
        out = out[0]
    return int(out)


@kill_cv2_on_error
def roi2(src, fac):
    """
    Allows to select the region of interest of a scaled image.

    Parameters
    ----------
    src : np.ndarray
        Single grayscale or color image.
    fac : float
        Scaling factor for the selection in percentage.

    Returns
    -------
    roi : tuple
        Dimensions of the ROI. (x0,y0,width,height), measured from upper
        left corner.

    """
    src = resize(src, fac)
    roi = cv2.selectROI(src)
    cv2.destroyAllWindows()
    roi = tuple(int(x/fac*100) for x in roi)
    return roi


def resize(src, fac=30):
    """Scales a single image by a certain percentage."""
    im_y = int(src.shape[0]*fac/100)
    im_x = int(src.shape[1]*fac/100)
    dim = (im_x, im_y)
    src = cv2.resize(src, dim, interpolation=cv2.INTER_NEAREST)
    return src


@kill_cv2_on_error
def show(src, fac=100, i_arg=0):
    """
    Shows a scaled image or sequence of images. Also prints the current index.

    Parameters
    ----------
    src : np.ndarray
        Single image or array of images.
    fac : float, optional
        Scaling factor for the selection in percentage. By default no scaling
        is performed.
    i_arg : TYPE, optional
        If src is an array of images, this index will be printed to the
        console when changing images. The default is 0.

    Returns
    -------
    None.

    """
    im_y = int(src.shape[-2]*fac/100)
    im_x = int(src.shape[-1]*fac/100)
    dim = (im_x, im_y)
    i_loc = copy(i_arg)

    if ((src.ndim == 3 and src.shape[-1] != 3) or src.ndim == 4):
        change = True
        while 1:
            if change:
                print(i_loc)
                # cv2.imshow("asdf", cv2.resize(
                #     src[i_loc], dim, interpolation=cv2.INTER_NEAREST))
                cv2.imshow("asdf", src[i_loc])
            change = False

            key = cv2.waitKeyEx(1) & 0xFF
            if key == 27:
                break
            if key == ord("z"):
                change = True
                if i_loc > i_arg:
                    i_loc -= 1
                else:
                    i_loc = i_arg+len(src)-1
            elif key == ord("x"):
                change = True
                if i_loc < i_arg+len(src)-1:
                    i_loc += 1
                else:
                    i_loc = i_arg

    else:
        src2 = cv2.resize(src, dim, interpolation=cv2.INTER_NEAREST)
        cv2.imshow("asdf", src2)
        cv2.waitKey()

    cv2.destroyAllWindows()


def k(ksize):
    """Returns array of ones with size (ksize,ksize)."""
    ker = np.ones((ksize, ksize), np.uint8)
    return ker


def k2(ksize):
    """Returns elliptical array of ones with size (ksize,ksize)."""
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return ker


def hplot(arr, bin_width=1, i_arg=0, ymin=0):
    """
    Histogram plotter.

    Plots histogram of a 8-bit grayscale image. Input array(s) must have
    two dimensions: the first one for the bin range and the second for the
    bin COUNT

    Parameters
    ----------
    arr : np.ndarray
        Single histogram or array of histograms.
    bin_width : int
        Width of the bars, should be set the same as the bin width.
        By default it is 1.
    i_arg : int, optional
        If arr is an array of histograms, this index will be printed to the
        legend of the plot.
    ymin : TYPE, optional
        Bottom limit of the y axis. The default is 0.

    Returns
    -------
    None.

    """
    if arr.ndim == 2:
        fig, axis = plt.subplots()
        fig.set_dpi(150)
        axis.set_ylim((ymin, 1.5))
        axis.set_xlim((0, 255))
        axis.bar(arr[0], arr[1], width=bin_width, label=i_arg)
        plt.legend()
    else:
        for loc_count, loc_val in enumerate(arr):
            fig, axis = plt.subplots()
            fig.set_dpi(150)
            axis.set_ylim((ymin, 1.5))
            axis.set_xlim((0, 255))
            axis.bar(
                loc_val[0],
                loc_val[1],
                width=bin_width,
                label=loc_count+i_arg
            )
            plt.legend()


def hsmooth(arr, smooth_val):
    """
    Smoothens out histograms obtained with histogram().

    Also crops the smooth_val-1 outermost values

    Parameters
    ----------
    arr : np.ndarray
        Single histogram or array of histograms.
    smooth_val : TYPE
        Smoothening value. Should be an odd value.

    Returns
    -------
    arr : np.ndarray
        Single histogram or array of histograms. First and last smooth_val//2
        values are cropped on either side.

    """
    if smooth_val > 1:
        if arr.ndim == 2:
            kernel = np.ones(smooth_val)/smooth_val
            temp = np.convolve(arr[1], kernel, mode="valid")
            arr = arr[:, smooth_val//2:-(smooth_val//2)]
            arr[1] = temp
        else:
            arr = np.array([hsmooth(arr2, smooth_val) for arr2 in arr])
    return arr


def hdiff(arr, n):
    """
    Returns n-order derivative of histogram obatined from histogram().

    Parameters
    ----------
    arr : np.ndarray
        Single histogram or array of histograms.
    n : int
        Order of the derivative. 0 returns the original input.

    Returns
    -------
    arr : np.ndarray
        Output histogram(s). Each iteration of n reduces last dimension by one

    """
    if n != 0:
        if arr.ndim == 2:
            temp = np.diff(arr[1], n)
            hist = arr[:, :-n]
            hist[1] = temp
        else:
            arr = np.array([hdiff(arr2, n) for arr2 in arr])
    return arr


def histogram(arr):
    """
    Produces the histogram of an array of 8-bit grayscale images.

    Parameters
    ----------
    arr : np.ndarray
        Array of grayscale image.

    Returns
    -------
    hist : np.ndarray
        2D array of the histogram. First dimension contains the bin index and
        second one contains the normalized bin COUNT in 0-1 range.

    """
    out = []
    add = np.arange(256, dtype=np.float32)
    for _img in arr:
        hist = cv2.calcHist([_img], [0], None, [256], [0, 256]).ravel()
        temp = np.stack((add, hist))
        out.append(temp)

    return np.array(out)


def nonan(arr):
    """Returns only non-nan values of a given contour."""
    return arr[~np.isnan(arr).reshape((-1, 2))].reshape((-1, 2)).astype(int)


# %% analyzer
def analyzer(localpath, minishow=False, fast=False, img_og=None, COUNT=None,
             th=None, CROP=None, tries=0, verbose=False, single=False):
    # %% Local init
    warnings.simplefilter('ignore', RuntimeWarning)

    if not fast and verbose:
        if img_og == None:
            print(localpath + "\n    Loading images")
        else:
            print(localpath + "\n    Retrying analysis")

    localpath = os.path.normpath(localpath)
    localpath_show = "/".join(localpath.split("\\"))
    localpath_og = localpath
    localpath = localpath.lower()

    # os.chdir(localpath)
    img_list = [image for image in os.listdir(localpath)
                if image[-3:] == "png"]
    img_list.sort(key=sorter)

    if img_list == []:
        if verbose:
            print("Empty folder")
        return ("empty", localpath_og)
    if fast:
        img_list = img_list[:min(len(img_list)-1, 10)]

    if not (fast or single):
        tags = [None] * 5
        if "ace" in localpath:
            tags[0] = ("ACE")
        elif "dcm" in localpath:
            tags[0] = ("DCM")
        elif "mini" in localpath:
            tags[0] = ("MINI")
        elif "pcl" in localpath:
            tags[0] = ("PCL")

        if "90" in localpath:
            tags[1] = (90)
        elif "110" in localpath:
            tags[1] = (110)
        elif "130" in localpath:
            tags[1] = (130)

        if "printing log" in localpath:
            tags[2] = os.listdir("\\".join(localpath.split("\\")[:-2]))
            tags[2] = tags[2].index(localpath.split("\\")[-2]) + 1
        else:
            _temp = "\\".join(localpath.split("\\")[:-2])
            _type = os.path.dirname(localpath).split(" ")[-1]
            _similar = [_dir.lower() for _dir in os.listdir(_temp) if _type in _dir.lower()]
            tags[2] = _similar.index(localpath.split("\\")[-2]) + 1

        if "\\l" in localpath:
            tags[3] = ("L")
        elif "\\r" in localpath:
            tags[3] = ("R")

        if " 10" in localpath:
            tags[4] = 10
        elif " 20" in localpath:
            tags[4] = 20
        else:
            tags[4] = 0

        if None in tags and verbose:
            print(localpath + "\n    ERROR: Problem with folder tagging")
            return("Problem with folder tagging", tags, localpath_og)

    max_layers = sorter(img_list[-1])
    valid_layers = np.zeros(max_layers, dtype=bool)
    for _img_name in img_list:
        valid_layers[sorter(_img_name)-1] = True

    if img_og == None:
        img_og = [None] * max_layers

        def read(i_loc):
            if valid_layers[i_loc] is False:
                return
            img_i = sorter(img_list[i_loc]) - 1
            _res = cv2.imread(os.path.join(localpath, img_list[i_loc]), -1)
            if _res is None:
                valid_layers[img_i] = False
                return
            img_og[img_i] = _res

        with tpe() as read_exe:
            read_exe.map(read, range(len(img_list)))

    img_og = [image for image in img_og if image is not None]

    PRE_PORE_DIST = 85
    if CROP == None:
        print("Select ROI")
        CROP = roi2(img_og[-1], 40)

    # It was found that the images had some darkening in the edges,
    # so the average local brigtness was corrected.
    # Hence, it was necessary to select one extra pore around the selected ROI
    img = [image[CROP[1]-PRE_PORE_DIST:CROP[1]+CROP[3]+PRE_PORE_DIST,
                 CROP[0]-PRE_PORE_DIST:CROP[0]+CROP[2]+PRE_PORE_DIST]
           for image in img_og]

    _blur_amount = PRE_PORE_DIST // 2 * 2 + 1
    img_vig = np.array([cv2.blur(image, (_blur_amount, _blur_amount)) for image in img])

    img0 = img[0][PRE_PORE_DIST:-PRE_PORE_DIST, PRE_PORE_DIST:-PRE_PORE_DIST]/255

    # Automatic detection of printing direction
    img0_sum_v = np.sum(np.abs((cv2.blur(img0, (1, 301)) - img0)))
    img0_sum_h = np.sum(np.abs((cv2.blur(img0, (301, 1)) - img0)))
    EVEN_IS_UPDOWN = img0_sum_v < img0_sum_h

    img = np.array(img)
    img = cv2.divide(img.ravel(), img_vig.max()/255).reshape(img_vig.shape)
    img = img[:, PRE_PORE_DIST:-PRE_PORE_DIST, PRE_PORE_DIST:-PRE_PORE_DIST]

    del img_vig

    # %% Calculating threshold
    if th is None:
        if verbose:
            print(localpath_show + "\n    Calculating binarization values...")
        # The histograms for the scaffold images can be thought of having two distinct
        # peaks, one for the dark background and one for the lighter printed material.
        # We are interested in selecting the rising edge of the second peak,
        # this way we eliminate the effects of the blurriness on the pictures.

        # A first binarization is performed using Otsu's method for bimodal images
        # This way we eliminate one of the two peaks on the image's histogram
        otsu = [cv2.threshold(image, 0, 255, cv2.THRESH_OTSU) for image in img]
        # otsu_vals = np.array([val[0] for val in otsu])
        otsu_img = np.array([val[1] for val in otsu])
        otsu_img_norm = img*(otsu_img//255)

        hists = histogram(otsu_img_norm)[..., 1:-1].astype(float)

        # The histograms are then smoothed and differentiated in order to get
        # the value of the second peak's rising edge.
        # We only select the positive parts of the derivatives, as we don't care
        # about the negative values.
        # Also, to eliminate the floor values we select the 15th smallest percentile
        # for the histogram distribution.

        hists = hsmooth(hists, 3)

        hists_min_vals = np.where(hists[:, 1] != 0, hists[:, 1], np.nan)
        hists_min_vals = np.nanpercentile(hists_min_vals, 15, axis=1)
        hists_norm = hists.copy()
        hists_norm[:, 1] -= hists_min_vals[:, None]
        hists_norm = np.clip(hists_norm, 0, 1000)

        hists2 = hsmooth(hists_norm.copy(), 15)
        hists2 = hsmooth(hists2, 3)
        hists2 = hsmooth(hists2, 3)
        hists2 = hsmooth(hists2, 3)
        hists3 = hdiff(hists2.copy(), 1)
        hists4 = hists3[..., :-1]
        hists4[:, 1] = np.divide(
            hists4[:, 1], np.max(hists4[:, 1], axis=1)[:, None])
        hists4 = np.clip(hists4, 0, 1000)

        hists4_min_vals = np.where(hists4[:, 1] != 0, hists4[:, 1], np.nan)
        hists4_min_vals = np.nanpercentile(hists4_min_vals, 10, axis=1)
        hists5 = hists4.copy()
        hists5[:, 1] -= hists4_min_vals[:, None]
        hists5 = np.clip(hists5, 0, 1000)

        # We are interested only in the first significant value, corresponding to the
        # second peak's shoulder.

        th_sig_vals = np.array([np.where(hist[1] > 0.05, True, False) for hist in hists5], dtype=int)

        th_vals = np.argmax(np.maximum(np.diff(th_sig_vals), 0), 1)
        th_vals = hists5[0, 0, th_vals]

        th_vals = th_vals * 0.9
    else:
        th_vals = th

    # %% Applying thresholds
    if verbose:
        print(localpath_show + "\n    Applying thresholds...")

    # Once we have the proper threshold values, we can proceed to actually binarize the images.
    img_th = 1 - np.array([cv2.threshold(img[i], th_vals[i], 255, cv2.THRESH_BINARY)[1]
                           for i in range(len(img))], dtype=np.float32) / 255

    # Discard first two layers
    img_th = img_th[2:]
    valid_layers = valid_layers[2:]

    # [0] is horizontal, [1] vertical
    img_edge_pre = [None] * 2
    for _j in range(2):
        img_edge_pre[_j] = np.where(
            np.sum(img_th, axis=-1-_j) / img_th.shape[-2+_j] > 0.15, 1.0, 0.0)
        img_edge_pre[_j] = np.array([cv2.morphologyEx(image, cv2.MORPH_OPEN, k(7))
                                     for image in img_edge_pre[_j]])
        img_edge_pre[_j] = np.array([cv2.morphologyEx(image, cv2.MORPH_CLOSE, k(15))
                                     for image in img_edge_pre[_j]])
        img_edge_pre[_j] = np.array([cv2.morphologyEx(image, cv2.MORPH_ERODE, k(33))
                                     for image in img_edge_pre[_j]])
        img_edge_pre[_j] = np.array([cv2.morphologyEx(image, cv2.MORPH_DILATE, k(39))
                                     for image in img_edge_pre[_j]])
        img_edge_pre[_j] = np.squeeze(img_edge_pre[_j])
        img_edge_pre[_j] = cv2.morphologyEx(img_edge_pre[_j], cv2.MORPH_CLOSE, k(9))

    img_edge_h = img_edge_pre[0][:, :, None]
    img_edge_v = img_edge_pre[1][:, None, :]

    img_edge = (img_edge_v * img_edge_h)

    img_edge_med_v = np.squeeze(np.percentile(img_edge_v, 85, axis=0, method='closest_observation'))
    img_edge_med_h = np.squeeze(np.percentile(img_edge_h, 85, axis=0, method='closest_observation'))

    if (img_edge_med_v[0] + img_edge_med_v[-1] + img_edge_med_h[0] + img_edge_med_h[-1]):

        if fast:
            print("There was a problem with the initial ROI")
            CROP, COUNT = analyzer(localpath, fast=True, verbose=False)
            return

        if tries == 0 and not fast:
            roll_v = roll_h = 0
            img_edge_roll_v = np.where(img_edge_med_v == 0)[0]
            roll_v_max = np.max(np.diff(img_edge_roll_v))
            img_edge_roll_v = np.array([img_edge_roll_v[0],
                                        img_edge_roll_v[-1] - len(img_edge_med_v) + 1])

            img_edge_roll_h = np.where(img_edge_med_h == 0)[0]
            roll_h_max = np.max(np.diff(img_edge_roll_h))
            img_edge_roll_h = np.array([img_edge_roll_h[0],
                                        img_edge_roll_h[-1] - len(img_edge_med_h) + 1])

            if img_edge_roll_v.any():
                roll_v = np.zeros(2, dtype=int)
                roll_v[0] = (roll_v_max - img_edge_roll_v[0] +
                             (PRE_PORE_DIST - roll_v_max) / 2)
                roll_v[1] = (PRE_PORE_DIST - roll_v_max + img_edge_roll_h[1] +
                             (PRE_PORE_DIST - roll_v_max) / 2)

                if not (roll_v[0] / roll_v[1] > 1.1 or
                        roll_v[0] / roll_v[1] < 0.9):
                    pass
                elif ((roll_v[0] > roll_v[1] and roll_v[0] < roll_v_max*0.85)
                      or roll_v[1] > roll_v_max*0.85):
                    roll_v = int(roll_v[0])
                else:
                    roll_v = int(-roll_v[1])

            if img_edge_roll_h.any():
                roll_h = np.zeros(2, dtype=int)
                roll_h[0] = (roll_h_max - img_edge_roll_h[0] +
                             (PRE_PORE_DIST - roll_h_max) / 2)
                roll_h[1] = (PRE_PORE_DIST - roll_h_max + img_edge_roll_h[1] +
                             (PRE_PORE_DIST - roll_h_max) / 2)

                if not (roll_h[0] / roll_h[1] > 1.1 or
                        roll_h[0] / roll_h[1] < 0.9):
                    pass
                elif ((roll_h[0] > roll_h[1] and roll_h[0] < roll_h_max*0.85)
                      or roll_h[1] > roll_h_max*0.85):
                    roll_h = int(roll_h[0])
                else:
                    roll_h = int(-roll_h[1])

            if isinstance(roll_v, int) and isinstance(roll_h, int):
                if verbose:
                    print("----------")
                if verbose:
                    print("WARNING: Retrying crop")
                return analyzer(localpath_og, img_og=img_og, th=th_vals, tries=1, COUNT=COUNT,
                                CROP=(CROP[0]-roll_v, CROP[1]-roll_h, CROP[2], CROP[3]),
                                verbose=verbose
                                )
            else:
                if verbose:
                    print(localpath + "\n    Rescheduled for next run")
                return ("crop retry", tags, localpath_og)

        elif tries == 1:
            if verbose:
                print(localpath + "\n    Rescheduled for next run")
            return ("crop retry", tags, localpath_og)

    img_th2 = img_th * img_edge

    # The following morphology operations fills in any small holes and bright
    # spots. Then, it closes medium sized gaps and holes for the final images
    img_th4 = np.empty_like(img_th2)
    for _num, _img in enumerate(img_th2):
        img_th4[_num] = cv2.morphologyEx(_img, cv2.MORPH_OPEN,  k2(9))
        img_th4[_num] = cv2.morphologyEx(_img, cv2.MORPH_CLOSE, k2(9))

    # Re-add missing images
    for _layer_num in np.nonzero(~valid_layers)[0]:
        img_th4 = np.insert(img_th4, _layer_num, 0, 0)
        img_edge = np.insert(img_edge, _layer_num, 0, 0)
        img_edge_v = np.insert(img_edge_v, _layer_num, 0, 0)
        img_edge_h = np.insert(img_edge_h, _layer_num, 0, 0)

    # %% Extracting contours
    if verbose:
        print(localpath_show + "\n    Extracting contours...")

    # We properly sort the position of the pores:
    # Contour information will be stored in a 5-dimensional array in this way:
    # 1st dim corresponds to each layer
    # 2nd dim corresponds to rows
    # 3rd dim corresponds to columns
    # 4th dim actually stores each contour's boundary points
    # 5th dim corresponds to X/Y values of the boundary points
    # Also, if we detect a clogged pore, we will mark it as such

    cnt_edge = [cv2.findContours(image.astype(np.uint8),
                                 cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)[0]
                for image in img_edge]

    CNT_COUNT = np.array([len(_layer) for _layer in cnt_edge])
    MAX_COUNT = np.max(CNT_COUNT)
    if MAX_COUNT == 0:
        return ("Empty images", tags, localpath_og)

    CNT_SQRT = int(MAX_COUNT ** 0.5)

    pore_nan = np.full((len(cnt_edge), CNT_SQRT, CNT_SQRT), "ok", "<U11")
    pore_nan_valid = np.ones(len(cnt_edge), dtype=bool)
    layer_shape = (CNT_SQRT, CNT_SQRT, 4, 2)
    _failed_layer = np.ones(layer_shape, dtype=float) * np.nan

    pore_nan[np.nonzero(~valid_layers)[0]] = "missing img"
    pore_nan_valid[np.nonzero(~valid_layers)[0]] = False
    for _layer in np.nonzero(~valid_layers)[0]:
        cnt_edge[_layer] = _failed_layer

    if (MAX_COUNT != int(MAX_COUNT**0.5)**2 or (COUNT != None and MAX_COUNT != COUNT)):
        if COUNT == None:
            raise RuntimeError("There were problems with the initial image set. "
                               "Code only works with square pore regions (NxN). "
                               "Try again with different ROI")
        CNT_SQRT = int(COUNT ** 0.5)
        for _num, _layer_count in enumerate(CNT_COUNT):
            if pore_nan_valid[_num] and _layer_count != COUNT:
                cnt_edge[_num] = _failed_layer
                pore_nan_valid[_num] = False
                pore_nan[_num] = "cnt fail"

    if fast:
        if verbose:
            print("-------\n")
        return CROP, MAX_COUNT

    for _num, _layer in enumerate(cnt_edge):
        if (pore_nan[_num] != "ok").any():
            continue
        try:
            cnt_edge[_num] = np.array(cnt_edge[_num], dtype=float).reshape(layer_shape)
        except ValueError:
            cnt_edge[_num] = _failed_layer
            pore_nan_valid[_num] = False
            pore_nan[_num] = "th fail"

    cnt_edge = np.array(cnt_edge)
    cnt_edge = np.flip(cnt_edge, axis=(1, 2, 3, 4))

    cnt_pre = [cv2.findContours(image.astype(np.uint8),
                                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
               for image in img_th4]
    CNT_MAX_SIZE = 0
    for layer in cnt_pre:
        for cont_loop in layer:
            CNT_MAX_SIZE = max(CNT_MAX_SIZE, len(cont_loop))

    contours = []
    for _lay, layer in enumerate(img_th4.astype(np.uint8)):
        contours_row = []
        for _row, row in enumerate(cnt_edge[_lay]):
            contours_column = []
            for _col, column in enumerate(row):
                if pore_nan[_lay, _row, _col] != "ok":
                    contour_temp = np.ones((CNT_MAX_SIZE, 2)) * np.nan
                    contours_column.append(contour_temp)
                    continue

                crop_y0 = int(min(column[:, 0]) - 10)
                crop_y1 = int(max(column[:, 0]) + 10)
                crop_x0 = int(min(column[:, 1]) - 10)
                crop_x1 = int(max(column[:, 1]) + 10)
                crop_region = layer[crop_y0:crop_y1, crop_x0:crop_x1]

                contour_temp = cv2.findContours(
                    crop_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

                if len(contour_temp) > 1:
                    contour_temp = np.ones((CNT_MAX_SIZE, 2)) * np.nan
                    pore_nan[_lay, _row, _col] = "SPLIT"
                else:
                    contour_temp = np.squeeze(contour_temp)
                    if contour_temp.shape[0] == 0 or contour_temp.ndim == 1:
                        contour_temp = np.ones((CNT_MAX_SIZE, 2)) * np.nan
                        pore_nan[_lay, _row, _col] = "EMPTY"
                    else:
                        # print(_lay, _col, _row)
                        contour_temp[:, 1] += crop_y0
                        contour_temp[:, 0] += crop_x0
                        contour_temp = np.pad(contour_temp.astype(float),
                                              ((0, CNT_MAX_SIZE-len(contour_temp)), (0, 0)),
                                              mode="constant", constant_values=np.nan)
                contours_column.append(contour_temp)
            contours_row.append(contours_column)
        contours.append(contours_row)
    contours = np.array(contours)

    # %% Calculating pore information
    if verbose:
        print(localpath_show + "\n    Calculating pore information...")

    # Once we have indexed all of the pores positions, we can finally extract
    # information from them in a layer-row-column fashion.
    # For the pore distance information, we take in consideration the direction
    # of the printing as previously calculated.


    # Just in case any pore touches the edge
    img_edge_h[:, -1] = 0
    img_edge_h[:, 1] = 0
    img_edge_v[..., -1] = 0
    img_edge_v[..., 1] = 0

    img_edge_v_diff = np.diff(img_edge_v[pore_nan_valid].squeeze())
    img_edge_v_diff = np.nonzero(img_edge_v_diff)[1].reshape(-1, CNT_SQRT, 2)
    img_edge_v_dist = np.diff(img_edge_v_diff, axis=1)

    img_edge_h_diff = np.diff(img_edge_h[pore_nan_valid].squeeze())
    img_edge_h_diff = np.nonzero(img_edge_h_diff)[1].reshape(-1, CNT_SQRT, 2)
    img_edge_h_dist = np.diff(img_edge_h_diff, axis=1)

    sample_conv_fac = ((img_edge_v_dist.mean() + img_edge_h_dist.mean()) / 2) / 0.8

    pore_area = np.empty((len(contours), CNT_SQRT, CNT_SQRT))
    pore_perim = np.empty((len(contours), CNT_SQRT, CNT_SQRT))
    pore_circ = np.empty((len(contours), CNT_SQRT, CNT_SQRT))
    for layer, _item in enumerate(contours):
        for i in range(CNT_SQRT):
            for j in range(CNT_SQRT):
                _p = layer, i, j
                pore_area[_p] = cv2.contourArea(nonan(contours[_p])) / sample_conv_fac**2
                if pore_area[_p] < 0.09:
                    if pore_nan[_p] == "ok":
                        pore_nan[_p] = "SMALL"
                    pore_area[_p] = np.nan
                    contours[_p] = np.nan

                pore_perim[_p] = cv2.arcLength(nonan(contours[_p]), True) / sample_conv_fac
                if pore_perim[_p] == 0:
                    pore_perim[_p] = np.nan

    pore_circ = (4 * np.pi * pore_area) / pore_perim**2
    SQUARE_CIRC = (4 * np.pi) / 4**2
    _lo_circ = pore_circ < SQUARE_CIRC / 1.08
    _hi_circ = pore_circ > SQUARE_CIRC * 1.08
    _circ_nan = np.nonzero(_lo_circ | _hi_circ)
    pore_nan[(_lo_circ | _hi_circ) & (pore_nan == "ok")] = "CIRC"
    pore_area[_circ_nan] = np.nan
    pore_perim[_circ_nan] = np.nan
    contours[_circ_nan] = np.nan

    contours_vertical_bool = np.zeros(len(contours), dtype=bool)
    if EVEN_IS_UPDOWN:
        contours_vertical_bool[0::2] = True
    else:
        contours_vertical_bool[1::2] = True

    pore_dist = np.empty((len(contours), CNT_SQRT, CNT_SQRT))
    pore_size = np.empty((len(contours), CNT_SQRT, CNT_SQRT))
    pore_dist[..., -1] = np.nan
    pore_size[..., -1] = np.nan
    _perc10, _perc90 = np.nanpercentile(contours, (10, 90), axis=3)
    for layer, _item in enumerate(contours):
        for u in range(CNT_SQRT):
            for v in range(CNT_SQRT-1):
                if contours_vertical_bool[layer]:
                    pore_actual_max_loc = _perc90[layer, u, v, 0]
                    pore_next_min_loc = _perc10[layer, u, v+1, 0]

                    pore_near = _perc10[layer, u, v, 0]
                    pore_far = _perc90[layer, u, v, 0]

                else:
                    pore_actual_max_loc = _perc90[layer, v, u, 1]
                    pore_next_min_loc = _perc10[layer, v+1, u, 1]

                    pore_near = _perc10[layer, u, v, 1]
                    pore_far = _perc90[layer, u, v, 1]

                pore_dist[layer, u, v] = (pore_next_min_loc - pore_actual_max_loc) / sample_conv_fac
                pore_size[layer, u, v] = (pore_far - pore_near) / sample_conv_fac

        if not contours_vertical_bool[layer]:
            pore_dist[layer] = pore_dist[layer].T
            pore_size[layer] = pore_size[layer].T

    # %% Show

    # This section is a small animation showing the binarized images, with each
    # pore being highlighted in sequence. It is useful to check any problems
    # with the original images, as well as remembering the layer-row-column order.
    # This section is optional and can be commented out if not needed.

    if minishow:
        # img2_gray = cv2.merge((img2[2:], img2[2:], img2[2:]))
        cv2.namedWindow("testshow")
        break_all = False
        while 1:
            img_th_draw = (img_th4 * 255).astype(np.uint8)
            zeros = np.zeros_like(img_th_draw, dtype=np.uint8)
            img_th_draw = cv2.merge((zeros, zeros, img_th_draw))
            for layer, _item in enumerate(contours):
                testshowcnt = list(contours[layer])
                print("Layer: " + str(layer))
                for z, _item in enumerate(testshowcnt):
                    testshowcnt[z] = [
                        nonan(arr).astype(int) for arr in testshowcnt[z]]

                for i in range(CNT_SQRT):
                    for j in range(CNT_SQRT):
                        if testshowcnt[i][j].shape[0] == 0:
                            continue
                        key = cv2.waitKeyEx(1) & 0xFF
                        if key == 27:
                            break_all = True
                            break
                        draw = cv2.drawContours(
                            img_th_draw[layer],
                            testshowcnt[i], j, (0, 255, 0), -1)
                        cv2.imshow("testshow", draw)
                    if break_all:
                        break
                    time.sleep(0)
                #  time.sleep(0.2)
                if break_all:
                    break
            if break_all:
                break
            break
        cv2.destroyAllWindows()

    # %% Returns

    pore_shape = pore_area.shape
    layer_count = len(pore_area)
    pore_count = pore_area.size

    ret_mat = np.tile(tags[0], pore_count)
    ret_temp = np.tile(tags[1], pore_count)
    ret_bg = np.tile(tags[4], pore_count)
    ret_rep = np.tile(tags[2], pore_count)
    ret_loc = np.tile(tags[3], pore_count)
    ret_dir = np.where(
        np.broadcast_to(contours_vertical_bool[:, None, None], pore_shape).flatten(),
        "Vertical", "Horizontal")
    ret_col = np.broadcast_to(np.arange(CNT_SQRT)[None, None, :], pore_shape).flatten()
    ret_row = np.broadcast_to(np.arange(CNT_SQRT)[None, :, None], pore_shape).flatten()
    ret_layer = np.broadcast_to(np.arange(3, layer_count+3)[:, None, None], pore_shape).flatten()
    ret_area = pore_area.flatten()
    ret_perim = pore_perim.flatten()
    ret_dist = pore_dist.flatten()
    ret_size = pore_size.flatten()
    ret_nan = pore_nan.flatten()
    ret_conv = np.tile(sample_conv_fac, pore_count)

    ret = np.vstack((
        ret_mat,
        ret_temp,
        ret_bg,
        ret_rep,
        ret_loc,
        ret_dir,
        ret_layer,
        ret_col,
        ret_row,
        ret_area,
        ret_perim,
        ret_dist,
        ret_size,
        ret_nan,
        ret_conv
    )).T

    headers = [
        "Material",
        "Temperature",
        "%BG",
        "Sample",
        "Location",
        "Print direction",
        "Layer",
        "Column",
        "Row",
        "Area",
        "Perimeter",
        "Pore distance",
        "Pore size",
        "Pore validity",
        "Conversion factor"]

    ret = pd.DataFrame(ret, columns=headers)
    ret["Temperature"] = ret["Temperature"].astype(float)
    ret["Sample"] = ret["Sample"].astype(int)
    ret["Layer"] = ret["Layer"].astype(int)
    ret["Column"] = ret["Column"].astype(int)
    ret["Row"] = ret["Row"].astype(int)
    ret["Area"] = ret["Area"].astype(float)
    ret["Perimeter"] = ret["Perimeter"].astype(float)
    ret["Pore distance"] = ret["Pore distance"].astype(float)
    ret["Pore size"] = ret["Pore size"].astype(float)
    ret["Conversion factor"] = ret["Conversion factor"].astype(float)
    # ret["Pore validity"] = ret["Pore validity"].astype(bool)

    if verbose:
        print(localpath_show + "\n    Done")

    return (ret, tags, localpath_og)
