# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 13:30:13 2021

@author: mpardo
"""

# from time import sleep
from subprocess import run as sub_run
import cv2
import numpy as np
import rawpy as rp
import warnings
import psutil
import functools

_out_types = {
    None: -1,
    np.uint8: 0,
    np.int8: 1,
    np.uint16: 2,
    np.int16: 3,
    np.int32: 4,
    np.float32: 5,
    np.float64: 6,
    float: 6,
}


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


def kill_process(name):
    for proc in psutil.process_iter():
        if proc.name() == name:
            proc.kill()


def cmd(comando, printed=False):
    out = sub_run(comando, shell=True, text=True, capture_output=True)
    if out.stdout != "":
        if printed: print(out.stdout)
        return out.stdout
    elif out.stderr != "":
        if printed: print(out.stderr)
        return out.stderr


@kill_cv2_on_error
def show(src, fac=100, dim1: int = None, dim2: int = None, verbose=False, near=False):
    """
    Shows a scaled image or sequence of images. Also prints the current index.

    Parameters
    ----------
    src : np.ndarray or list
        Single image, array of images or list of images.
    fac : float, optional
        Scaling factor for the selection in percentage. By default no scaling
        is performed.
    """
    if isinstance(src, np.ndarray):
        if src.ndim == 2:
            src = src[None, ...]
        elif src.ndim == 3:
            if src.shape[-1] == 3:
                src = src[None, ...]

        if src.dtype in (float, np.float16, np.float32):
            if src.max() > 50 and src.max() < 300:
                src /= 255
            elif src.max() >= 300:
                src /= 65535
            src = src.astype(float)

        if src.dtype == bool:
            src = src.astype(float)
    elif not isinstance(src, list):
        raise TypeError("Input was neither a list or a numpy array")

    ilen = len(src)
    i_actual = 0

    change = True
    while 1:
        if change:
            if verbose:
                print(i_actual)
            cv2.imshow("asdf", resize(src[i_actual], fac, dim1, dim2, near=near))
        change = False

        key = cv2.waitKeyEx(1) & 0xFF
        if key == 27:
            break
        if key == ord("z"):
            change = True
            i_actual -= 1
            i_actual %= ilen
        elif key == ord("x"):
            change = True
            i_actual += 1
            i_actual %= ilen

    cv2.destroyAllWindows()


def sho2(src, *args, **kwargs):
    show(src, fac=None, dim1=None, dim2=720, *args, **kwargs)


@kill_cv2_on_error
def show_crop(src, fac=100, dim1: int = None, dim2: int = None,
              verbose=False, coords=None) -> tuple:
    # crop is (y1,y2,x1,x2)
    # cv2 rectangle is (x1,y1), (x2,y2)
    _pressed = mode = y_check = x_check = x0 = y0 = None

    def mouse_click(event, x, y, flags, param):
        nonlocal coord_change
        nonlocal _pressed
        nonlocal mode
        nonlocal y_check, x_check
        nonlocal x0, y0

        if event == cv2.EVENT_LBUTTONDOWN:
            _pressed = True
            x_check = 0
            y_check = 0

            if np.abs(y - coords[0]) < 10:
                y_check = 1
            elif np.abs(y - coords[1]) < 10:
                y_check = 2
            if np.abs(x - coords[2]) < 10:
                x_check = 1
            elif np.abs(x - coords[3]) < 10:
                x_check = 2

            if x_check and y_check:
                mode = "vertex"
            elif x_check:
                mode = "col"
            elif y_check:
                mode = "row"
            else:
                mode = "drag"
                x0 = x
                y0 = y

        elif event == cv2.EVENT_MOUSEMOVE:
            if _pressed:
                if mode == "vertex":
                    coords[-1+y_check] = y
                    coords[1+x_check] = x

                elif mode == "col":
                    coords[1+x_check] = x

                elif mode == "row":
                    coords[-1+y_check] = y

                elif mode == "drag":
                    coords[0:2] += (y - y0)
                    coords[2:4] += (x - x0)
                    x0 = x
                    y0 = y

        elif event == cv2.EVENT_LBUTTONUP:
            _pressed = False
            coords[:2] = np.clip(coords[:2], 2, temp.shape[0] - 2)
            coords[2:] = np.clip(coords[2:], 2, temp.shape[1] - 2)

    # crop is (y1,y2,x1,x2)
    # cv2 rectangle is (x1,y1), (x2,y2)
    if src.ndim == 2:
        src = src[None, ...]
    elif src.ndim == 3:
        if src.shape[-1] == 3:
            src = src[None, ...]

    src = f2u(src.copy(), 8)

    ilen = len(src)
    i_actual = 0

    if coords is None:
        src_check = resize(src[0], fac, dim1, dim2)
        coords = np.array(
            (src_check.shape[0] * 0.02, src_check.shape[0] * 0.98,
             src_check.shape[1] * 0.02, src_check.shape[1] * 0.98)).astype(int)
    else:
        coords = np.array(coords, dtype=int)
        if dim1 is not None:
            fac = dim1 / src.shape[1] * 100
        elif dim2 is not None:
            fac = dim2 / src.shape[2] * 100

        coords = (coords * (fac / 100)).astype(int)

    i_change = True
    cv2.namedWindow("Crop Select")
    cv2.setMouseCallback('Crop Select', mouse_click)
    while 1:
        if i_change:
            if verbose:
                print(i_actual)
            temp = resize(src[i_actual], fac, dim1, dim2)
            i_change = False
            coord_change = True

        vertex1 = (coords[2], coords[0])
        vertex2 = (coords[3], coords[1])
        temp2 = temp.copy()
        cv2.rectangle(temp2, vertex1, vertex2, (20, 20, 255), 2)
        cv2.imshow("Crop Select", temp2)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == 13:
            break
        if key == ord("z"):
            i_change = True
            i_actual -= 1
            i_actual %= ilen
        elif key == ord("x"):
            i_change = True
            i_actual += 1
            i_actual %= ilen

    cv2.destroyAllWindows()

    if dim1 is not None:
        fac = dim1 / src.shape[2] * 100
    elif dim2 is not None:
        fac = dim2 / src.shape[1] * 100

    coords = (coords / (fac / 100)).astype(int)
    coords[:2] = np.clip(coords[:2], 0, src.shape[1])
    coords[2:] = np.clip(coords[2:], 0, src.shape[2])

    return np.array((
        min(coords[0], coords[1]), max(coords[0], coords[1]),
        min(coords[2], coords[3]), max(coords[2], coords[3]),
    ))


def resize(src: np.ndarray, fac: float = None,
           dim1: int = None, dim2: int = None,
           near: bool = False, no_upsize: bool = False) -> np.ndarray:
    inter = cv2.INTER_NEAREST if near else None

    if isinstance(src, np.ndarray) and (src.ndim == 2 or (src.ndim == 3 and src.shape[-1] == 3)):
        if fac is not None and (dim1, dim2) == (None, None):
            if fac == 100:
                return src
            if fac > 100 and no_upsize:
                return src
            return cv2.resize(src, None, None, fac/100, fac/100, interpolation=inter)

        elif dim1 or dim2:
            if dim1 and dim2:
                warnings.warn(
                    "As dim1 and dim2 are provided, stretching might occur",
                    SyntaxWarning)
                dims = (dim1, dim2)
                if np.sign(src.shape[1] - src.shape[0]) != np.sign(dim1 - dim2):
                    dims = (dim2, dim1)
            elif dim1:
                fac = src.shape[1] / dim1
                dim2 = src.shape[0] / fac
                dims = (dim1, dim2)
            elif dim2:
                fac = src.shape[0] / dim2
                dim1 = src.shape[1] / fac
                dims = (dim1, dim2)

            if fac > 100 and no_upsize:
                return src

            return cv2.resize(src, [int(dim) for dim in dims], interpolation=inter)

        else:
            raise AttributeError("No dimensions specified")

    else:
        ret_type = type(src)
        if ret_type in (np.ndarray, list):
            ret_type = np.array
            return np.array([resize(_img, fac, dim1, dim2, near) for _img in src])
        else:
            raise TypeError("Input was neither a list or a numpy array")


def CCM(src_arr, ccm):
    src_arr = r2b(src_arr)
    src_arr = np.matmul(src_arr, ccm.T)
    src_arr = r2b(src_arr)
    return src_arr


def gamma(src, gammaA, gammaB=None, gammaG=None, gammaR=None):
    src_out = np.empty_like(src)
    if gammaB and gammaG and gammaR and src.shape[-1] == 3:
        vec_gamma = np.array([gammaB, gammaG, gammaR]) * gammaA
        vec_gamma = 1 / vec_gamma
        for _j in range(3):
            src_out[..., _j] = cv2.pow(src[..., _j], vec_gamma[_j])

    else:
        src_out = cv2.pow(src, 1 / gammaA)

    src_out[np.isnan(src_out)] = 0
    return src_out


@kill_cv2_on_error
def ccmGamma(src_arr: np.ndarray, fac: float = None, dim1=None, dim2=720,
             init_gamma: np.ndarray = None, show_hist: bool = True) -> dict:
    def nothing(p): pass

    def gammaChange(p):
        global gamma_update
        gamma_update = True

    if src_arr.ndim == 3:
        src_arr = src_arr[None, ...]

    i = 0

    if 1:  # create trackbars
        cv2.namedWindow('image')
        cv2.namedWindow('histogram')
        cv2.namedWindow('tracks', cv2.WINDOW_NORMAL)

        cv2.createTrackbar('Clipping', 'tracks', 0, 1, nothing)
        cv2.createTrackbar('Black point', 'tracks', 25, 100, gammaChange)
        cv2.createTrackbar('White point', 'tracks', 50, 100, gammaChange)

        for char in ("R", "G", "B"):
            cv2.createTrackbar(f'Black {char}', 'tracks', 50, 100, gammaChange)
        for char in ("R", "G", "B"):
            cv2.createTrackbar(f'White {char}', 'tracks', 50, 100, gammaChange)
        if init_gamma is None:
            for chan in ("All", "R", "G", "B"):
                cv2.createTrackbar(f'{chan} gamma', 'tracks', 80, 100, gammaChange)
        else:
            init_gamma = init_gamma.copy()
            init_gamma[0] = 100/2 * (np.log(init_gamma[0])/np.log(4) + 1.6)
            init_gamma[1:] = 100/2 * (np.log(init_gamma[1:])/np.log(3) + 1.6)
            init_gamma = init_gamma.astype(int)
            for num, chan in enumerate(("All", "R", "G", "B")):
                cv2.createTrackbar(f'{chan} gamma','tracks', init_gamma[(4-num)%4], 100, gammaChange)

        cv2.createTrackbar('Disable CCM', 'tracks', 0, 1, nothing)
        cv2.createTrackbar('Reset',       'tracks', 0, 1, nothing)

        cv2.createTrackbar('R-R', 'tracks', 100, 250, nothing)
        cv2.createTrackbar('R-G/B', 'tracks', 100, 200, nothing)
        cv2.createTrackbar('G-G', 'tracks', 100, 250, nothing)
        cv2.createTrackbar('G-R/B', 'tracks', 100, 200, nothing)
        cv2.createTrackbar('B-B', 'tracks', 100, 250, nothing)
        cv2.createTrackbar('B-R/G', 'tracks', 100, 200, nothing)

    src = resize(src_arr[i], fac, dim1, dim2)

    gamma_update = True
    while 1:
        k = cv2.waitKeyEx(1) & 0xFF
        gamma_update = True
        if k == 27 or k == 13:
            break
        elif k == ord("z"):
            i -= 1
            i %= len(src_arr)
            src = resize(src_arr[i], fac, dim1, dim2)
        elif k == ord("x"):
            i += 1
            i %= len(src_arr)
            src = resize(src_arr[i], fac, dim1, dim2)

        if 1:  # Getting positions
            black = np.empty(4, dtype=float)
            white = np.empty(4, dtype=float)
            black[0] = cv2.getTrackbarPos('Black point', 'tracks') / 1000 * 15 - 0.375
            white[0] = cv2.getTrackbarPos('White point', 'tracks') / 1000 * 4 - 0.2
            for num, char in enumerate(("B", "G", "R")):
                black[num + 1] = cv2.getTrackbarPos(f'Black {char}', 'tracks') / 1000 * 4 - 0.2
                white[num + 1] = cv2.getTrackbarPos(f'White {char}', 'tracks') / 1000 * 2 - 0.1

            gammaa = cv2.getTrackbarPos('All gamma', 'tracks')
            gammar = cv2.getTrackbarPos('R gamma', 'tracks')
            gammag = cv2.getTrackbarPos('G gamma', 'tracks')
            gammab = cv2.getTrackbarPos('B gamma', 'tracks')

            gammaa = 4**((gammaa / 100) * 2 - 1.6)
            gammar = 3**((gammar / 100) * 2 - 1.6)
            gammag = 3**((gammag / 100) * 2 - 1.6)
            gammab = 3**((gammab / 100) * 2 - 1.6)

            clipping = cv2.getTrackbarPos('Clipping', 'tracks')
            disable = cv2.getTrackbarPos('Disable CCM', 'tracks')
            reset = cv2.getTrackbarPos('Reset', 'tracks')

            r1 = cv2.getTrackbarPos('R-R', 'tracks') / 100
            r2 = cv2.getTrackbarPos('R-G/B', 'tracks') / 100 - 1
            g1 = cv2.getTrackbarPos('G-G', 'tracks') / 100
            g2 = cv2.getTrackbarPos('G-R/B', 'tracks') / 100 - 1
            b1 = cv2.getTrackbarPos('B-B', 'tracks') / 100
            b2 = cv2.getTrackbarPos('B-R/G', 'tracks') / 100 - 1

        if reset:
            reset = 0
            cv2.setTrackbarPos('R-R', 'tracks', 100)
            cv2.setTrackbarPos('R-G/B', 'tracks', 100)
            cv2.setTrackbarPos('G-G', 'tracks', 100)
            cv2.setTrackbarPos('G-R/B', 'tracks', 100)
            cv2.setTrackbarPos('B-B', 'tracks', 100)
            cv2.setTrackbarPos('B-R/G', 'tracks', 100)
            cv2.setTrackbarPos('Reset', 'tracks', 0)

        if disable:
            ccm = np.eye(3)
        else:
            ccm = np.array([
                [r1, 0.5 + 0.5 * r2 - 0.5 * r1, 0.5 - 0.5 * r2 - 0.5 * r1],
                [0.5 + 0.5 * g2 - 0.5 * g1, g1, 0.5 - 0.5 * g2 - 0.5 * g1],
                [0.5 + 0.5 * b2 - 0.5 * b1, 0.5 - 0.5 * b2 - 0.5 * b1, b1]])

        # if gamma_update:
        if 1:
            temp = (src.copy() + black[0]) / (1 + black[0])
            temp = (temp + black[1:]) / (1 + black[1:])

            _white = (1 + white[0]) * (1 + white[1:])
            temp = temp * _white

            temp = gamma(temp, gammaa, gammab, gammag, gammar)
            temp = CCM(temp, ccm)
            gamma_update = False

            if show_hist:
                hist = histogram(temp)

        if clipping:
            temp[temp >= 1] = 0.001
            temp[temp <= 0] = 1

        cv2.imshow("image", temp)
        if show_hist:
            cv2.imshow("histogram", hist)

    cv2.destroyAllWindows()

    out = {"ccm": ccm,
           "white": white,
           "black": black,
           "gamma": (gammaa, gammab, gammag, gammar),
           }

    return out


def ccmGamma_apply(src: np.ndarray, params_in: dict) -> np.ndarray:
    black = params_in["black"]
    white = params_in["white"]
    gammas = params_in["gamma"]
    ccm = params_in["ccm"]

    src = (src + black[0]) / (1 + black[0])
    src = (src + black[1:]) / (1 + black[1:])

    _white = (1 + white[0]) * (1 + white[1:])
    src = src * _white

    src = gamma(src, *gammas)
    src = CCM(src, ccm)

    return src


def diffshow(src, fac):
    src = np.interp(src, (src.min(), src.max()), (0, 1))
    show(src, fac)


def bgr2gray(src):
    return src[..., 2] * 0.299 + src[..., 1] * 0.587 + src[..., 0] * 0.114


def mask_edge(src):
    src = bgr2gray(src.astype(np.float64))
    src = u2f(src)
    short = src.shape[0]

    a = 0.5
    src_norm = src / cv2.blur(src, k2(short*a))
    src_norm /= np.percentile(src_norm, 99.5)
    gray = src_norm[..., 0] ** 0.5 * \
        src_norm[..., 1] ** 2 * src_norm[..., 2] ** 0.5
    gray = np.clip(gray, 0, 5)

    b = 0.075
    blur = cv2.blur(gray, k2(short*b/3))
    for __ in range(3):
        blur = cv2.blur(blur, k2(short*b/3))
    diff = gray - blur

    c = 0.05
    diffblur = cv2.blur(np.abs(diff), k2(short*c/3))
    for __ in range(3):
        diffblur = cv2.blur(diffblur, k2(short*c/3))
    diffblur /= np.percentile(diffblur, 99.9)

    d = 2
    diff3 = diff - diffblur * d

    e = 0.15
    diff4 = np.where(diff3 > e, 255, 0)

    return diff4.astype(np.uint8)


def mask_thres(src, thres):
    c = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    __, c = cv2.threshold(c, thres, 255, cv2.THRESH_BINARY)
    return c.astype(np.uint8)


def k(ksize):
    ksize = max(ksize, 1)
    ker = np.ones((ksize, ksize), np.uint8)
    return ker


def k2(ksize):
    ksize = max(ksize, 1)
    return (int(ksize), int(ksize))


def sponerMsk(img1, img2, fac):
    if len(img2.shape) == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    img2[:, :, 0] = np.multiply(img2[:, :, 0], (fac[0] / 255))
    img2[:, :, 1] = np.multiply(img2[:, :, 1], (fac[1] / 255))
    img2[:, :, 2] = np.multiply(img2[:, :, 2], (fac[2] / 255))

    out = (img1).astype(np.uint16) | img2.astype(np.uint16)

    return out.astype(np.uint8)


def norm(src: np.ndarray,
         th_lo: float = 0.1, th_hi: float = 99.9,
         val_lo: float = None, val_hi: float = None,
         skip: int = 5, dtype=np.float32,
         rgb: bool = True) -> np.ndarray:

    if dtype == np.uint8:
        dtype_max = 255
    elif dtype == np.uint16:
        dtype_max = 65535
    elif dtype in (float, np.float16, np.float32, np.float64):
        dtype_max = 1
    else:
        raise AttributeError("Allowed dtypes are np.uint8, np.uint16 or a float type")

    if ((src.shape[-1] == 3 and src.ndim == 3) or src.ndim == 2):
        src = src[None, ...]

    src_out = np.empty_like(src)
    src_perc = src[:, ::skip, ::skip]
    if src_out.shape[-1] == 3 and rgb:
        if val_lo is None or val_hi is None:
            perc_lo, perc_hi = np.percentile(src_perc, (th_lo, th_hi), axis=(0, 1, 2)) / dtype_max
            val_lo = perc_lo if val_lo is None else val_lo
            val_hi = perc_hi if val_hi is None else val_hi
        for i in range(3):
            src_out[..., i] = cv2.addWeighted(
                src1=src[..., i],
                alpha=1/(val_hi[i] - val_lo[i]),
                src2=fast_like(dtype_max, src[..., i]),
                beta=-val_lo[i] / (val_hi[i] - val_lo[i]),
                gamma=0,
                dtype=_out_types[dtype])

    else:
        if val_lo is None or val_hi is None:
            perc_lo, perc_hi = np.percentile(src_perc, (th_lo, th_hi)) / dtype_max
            val_lo = perc_lo if val_lo is None else val_lo
            val_hi = perc_hi if val_hi is None else val_hi
        src_out = cv2.addWeighted(
            src1=src,
            alpha=1/(val_hi - val_lo),
            src2=fast_like(1, src),
            beta=-val_lo / (val_hi - val_lo),
            gamma=0,
            dtype=_out_types[dtype])

    return src_out


# OPTMIZE: with cv2.addWeighted
def float2uint(src, out_bits=16, copy=False, displace=0):
    if copy:
        src = src.copy()

    if out_bits == 16:
        out_dt = np.uint16
    elif out_bits == 8:
        out_dt = np.uint8
    else:
        raise AttributeError("'out' must be either 8 or 16")

    out = np.empty_like(src, out_dt)

    if src.dtype in (float, np.float64, np.float32, np.float16):
        src.clip(0, 1, out=src)
        if out_bits == 16:
            src *= 65535
        elif out_bits == 8:
            src *= 255
        out[:] = src
        return out

    if src.dtype == np.uint16:
        if out_bits == 16:
            return src
        elif out_bits == 8:
            out[:] = src >> 8
            return out

    if src.dtype == np.uint8:
        if out_bits == 8:
            return src
        elif out_bits == 16:
            out[:] = src << 8
            return out

    if src.dtype == int:
        src.clip(0, out_dt**2-1, out=src)
        out[:] = src
        return out


def f2u(*args, **kwargs):
    return float2uint(*args, **kwargs)


# OPTMIZE with cv2.addWeighted
def uint2float(src):
    if src.dtype in (float, np.float16, np.float32, np.float64):
        return src

    out = np.empty_like(src, float)
    if src.dtype == int:
        out[:] = src
        return out
    if src.dtype == np.uint16:
        out[:] = src
        out /= 65535
        return out
    if src.dtype == np.uint8:
        out[:] = src
        out /= 255
        return out


def u2f(src):
    return uint2float(src)


def r2b(src):
    return np.flip(src, axis=-1)


def compress_shadows(src, fixed, fac):
    if fac < 0:
        raise ValueError("fac must be greater than 0")
    if fixed <= 0 or fixed >= 1:
        raise ValueError("Fixed point must be between 0 and 1")
    gamma_fac = (np.log(fixed + fac) - np.log(1 + fac)) / np.log(fixed)

    src_out = (src + fac) / (1 + fac)
    src_out = gamma(src_out, gamma_fac)

    return src_out


def compress_highlights(src, fixed, fac):
    if fac < 0:
        raise ValueError("fac must be greater than 0")
    if fixed <= 0 or fixed >= 1:
        raise ValueError("Fixed point must be between 0 and 1")
    fixed = 1 - fixed
    gamma_fac = (np.log(fixed + fac) - np.log(1 + fac)) / np.log(fixed)

    src_out = (1 - src + fac) / (1 + fac)
    src_out = 1 - gamma(src_out, gamma_fac)

    return src_out


def inpaint(src, msk, radius=6):
    msk = cv2.dilate(msk, k(radius))

    if src.dtype in (float, np.float16):
        src = src.astype(np.float32)
    src_out = np.empty_like(src)
    for _j in range(3):
        src_out[..., _j] = cv2.inpaint(src[..., _j], msk, 10, cv2.INPAINT_TELEA)

    return src_out


def histogram(src: np.ndarray, h_res: int = 256, v_res: int = 100,
              col_range: tuple = (0.2, 0.9), rgb: bool = True,
              in_fac: float = 100, out_fac: float = 100) -> np.ndarray:

    src = resize(src, in_fac)
    if not rgb and src.ndim == 3:
        src = bgr2gray(src)
    if src.ndim == 2:
        src = src[..., None]

    src = u2f(src).clip(0, 1)
    src *= (h_res - 1)
    src = src.astype(int)

    out = []
    bins = []
    for channel in range(src.shape[-1]):
        bins.append(np.bincount(src[..., channel].ravel(), minlength=h_res))

    bins = np.array(bins)
    bins = bins/np.percentile(bins, 99) * v_res
    bins = bins.astype(int)

    for channel in range(src.shape[-1]):
        draw = np.zeros((v_res, h_res))
        for _col, _height in enumerate(bins[channel]):
            draw[:_height, _col] = 1
        draw = np.flipud(draw)
        draw *= (col_range[1] - col_range[0])
        draw += col_range[0]

        out.append(draw)

    out = np.squeeze(np.array(out))
    if out.ndim == 3:
        out = np.rollaxis(out, 0, 3)
    out = resize(out, out_fac)

    return out


def calc_gamma(src, target_gamma: float = 0.5):
    if src.shape[-1] != 3:
        raise ValueError("Input image was not in color")

    if src.ndim == 2:
        src = src[None, ...]

    src = u2f(src)
    gammas = np.log(src.mean((0, 1, 2))) / np.log(target_gamma)
    gamma_max = gammas.max()
    gammas /= gamma_max

    return np.array((gamma_max, *gammas))


def thumb(filename, fac=100, dim1=None, dim2=None):
    with rp.imread(filename) as raw:
        thumb = raw.extract_thumb().data
    src = cv2.imdecode(np.frombuffer(thumb, dtype=np.uint8), -1)
    src = resize(src, fac, dim1, dim2)

    return src


def rawread(filename, **kwargs):
    with rp.imread(filename) as raw:
        out = np.ascontiguousarray(raw.postprocess(**kwargs))
        return r2b(out)


def fast_array(init_val, shape, dtype=None):
    return np.broadcast_to(np.array((init_val), dtype=dtype), shape)


def fast_like(init_val, arr_like, dtype=None):
    if dtype is None:
        return np.broadcast_to(np.array((init_val), dtype=arr_like.dtype), arr_like.shape)

    return np.broadcast_to(np.array((init_val), dtype=dtype), arr_like.shape)


def rotate(src, angle: int):
    # if isinstance(src, list)
    # if src.ndim == 2 or (src.ndim == 3 and src.shape[-1] == 3):
    center = (src.shape[1]//2, src.shape[0]//2)
    scale = 1
    rot_mat = cv2.getRotationMatrix2D(center, angle, scale)
    out = cv2.warpAffine(src, rot_mat, (src.shape[1], src.shape[0]))

    return out


def mult(src, scalar, dtype=None, inplace=False):
    if inplace:
        cv2.addWeighted(src, scalar, fast_like(0, src), 0, 0, src[:], _out_types[dtype])
        return

    return cv2.addWeighted(src, scalar, fast_like(0, src), 0, 0, None, _out_types[dtype])


def add(src, scalar, dtype=None, inplace=False):
    if inplace:
        cv2.addWeighted(src, 1, fast_like(1, src), scalar, 0, src[:], _out_types[dtype])
        return

    return cv2.addWeighted(src, 1, fast_like(1, src), scalar, 0, None, _out_types[dtype])


def merge_rgb(file0, file1: str = None, file2: str = None,
              order="rgb",
              args: dict = {},
              exp_correct=True,
              dtype=np.float32,
              et_path=None,
              et_instance=None,
              ):

    if not all([char in order.lower() for char in ("r", "g", "b")]) or len(order) != 3:
        raise AttributeError("Order not recognized")

    if dtype == np.uint8:
        dtype_max = 255
    elif dtype == np.uint16:
        dtype_max = 65535
    elif dtype in (float, np.float16, np.float32, np.float64):
        dtype_max = 1
    else:
        raise AttributeError("Allowed dtypes are np.uint8, np.uint16 or a float type")

    if None in (file1, file2):
        file0, file1, file2 = file0

    _files = [file0, file1, file2]
    files = [
        _files[order.index("b")],
        _files[order.index("g")],
        _files[order.index("r")],
    ]

    if exp_correct:
        if et_instance is None:
            from exiftool import ExifTool
            with ExifTool(et_path) as et:
                exif = et.execute_json(*files)
        else:
            exif = et_instance.execute_json(*files)
        try:
            exp_shutter = np.array([float(item["EXIF:ExposureTime"]) for item in exif])
            exp_aperture = np.array([float(item["EXIF:FNumber"]) for item in exif])
            exp_iso = np.array([float(item["EXIF:ISO"]) for item in exif])
        except NameError:
            warnings.warn("Hubo un error leyendo las etiquetas EXIF, "
                          "se continuará sin corregir luminosidad RGB")
            exp_correct = False

        if exp_correct:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('error')
                    exp_ev = np.log2(exp_aperture**2 / (exp_shutter * exp_iso/100))
            except RuntimeWarning:
                warnings.warn("No se pudo calcular la exposición automáticamente "
                              "porque al menos una de las etiquetas EXIF es inválida, "
                              "se continuará sin corregir luminosidad RGB")
                exp_correct = False

    raw_b = rp.imread(files[0])
    raw_g = rp.imread(files[1])
    raw_r = rp.imread(files[2])

    raw_g.raw_image[0::2, 0::2] = raw_r.raw_image[0::2, 0::2]
    raw_g.raw_image[1::2, 1::2] = raw_b.raw_image[1::2, 1::2]

    raw = r2b(raw_g.postprocess(**args))
    white_level = raw_g.camera_white_level_per_channel[0] - raw_g.black_level_per_channel[0]

    raw = mult(raw, dtype_max/white_level, dtype=dtype)

    if exp_correct:
        raw *= 2 ** (exp_ev - exp_ev.max())
        return raw, np.flip(exp_ev)

    return raw, None


def flatten(list_in: list) -> list:
    return [item for sublist in list_in for item in sublist]


def sRGB_to_XYZ(src):
    ccm_srgb_XYZ = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])

    return CCM(src, ccm_srgb_XYZ)


def XYZ_to_Adobe(src):
    ccm_XYZ_adobe = np.array([
        [2.0413690, -0.5649464, -0.3446944],
        [-0.9692660,  1.8760108,  0.0415560],
        [0.0134474, -0.1183897,  1.0154096],
    ])

    return CCM(src, ccm_XYZ_adobe)


def rebayer(src: np.ndarray, order="0123") -> np.ndarray:
    '''Rebayers into RGGB pattern'''
    if not isinstance(src, np.ndarray):
        raise AttributeError("Input was not numpy array")

    if src.ndim == 3:
        out = np.empty((src.shape[-3]*2, src.shape[-2]*2), np.uint16)
        channels = {
            "0": (slice(0, None, 2), slice(0, None, 2)),
            "1": (slice(0, None, 2), slice(1, None, 2)),
            "3": (slice(1, None, 2), slice(0, None, 2)),
            "2": (slice(1, None, 2), slice(1, None, 2)),
        }
    elif src.ndim == 4:
        out = np.empty((src.shape[-4], src.shape[-3]*2, src.shape[-2]*2), np.uint16)
        channels = {
            "0": (slice(None), slice(0, None, 2), slice(0, None, 2)),
            "1": (slice(None), slice(0, None, 2), slice(1, None, 2)),
            "3": (slice(None), slice(1, None, 2), slice(0, None, 2)),
            "2": (slice(None), slice(1, None, 2), slice(1, None, 2)),
        }
    else:
        raise AttributeError("Input was not 3D array or array of 3D arrays")

    for num, c_out in enumerate(order):
        num = 2 if num == 3 and src.shape[-1] == 3 else num
        out[channels[c_out]] = src[..., num]

    return out


def read(filename: str, options: str):
    with open(filename, options) as file:
        return file.read()


def write(filename: str, options: str):
    with open(filename, options) as file:
        return file.write()


args_proxy = dict(
    demosaic_algorithm=rp.DemosaicAlgorithm.LINEAR,
    half_size=True,
    fbdd_noise_reduction=rp.FBDDNoiseReductionMode.Off,
    use_camera_wb=False,
    use_auto_wb=False,
    user_wb=[1, 1, 1, 1],
    output_color=rp.ColorSpace.raw,
    output_bps=16,
    user_sat=65535,
    # user_black=0,
    no_auto_bright=True,
    no_auto_scale=True,
    gamma=(1, 1),
    user_flip=0,
    # bright=4
)

args_full = dict(
    # demosaic_algorithm=rp.DemosaicAlgorithm.DHT, #now set by setup.ini
    # half_size=True,
    fbdd_noise_reduction=rp.FBDDNoiseReductionMode.Off,
    use_camera_wb=False,
    use_auto_wb=False,
    user_wb=[1, 1, 1, 1],
    output_color=rp.ColorSpace.raw,
    output_bps=16,
    user_sat=65535,
    # user_black=0,
    no_auto_bright=True,
    no_auto_scale=True,
    gamma=(1, 1),
    # user_flip=0
    # bright=4
)
