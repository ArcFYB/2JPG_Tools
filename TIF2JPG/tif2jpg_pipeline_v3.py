# -*- coding: utf-8 -*-
"""
tif2jpg_pipeline.py

功能概述
1) 支持 1 波段 / 3 波段 GeoTIFF(.tif/.tiff) 裁剪：按超参数 crop_size/stride 生成 patch，保存到 folder1(裁剪输出目录)
2) 将 tif 转为 jpg：保存到 folder2(输出目录)，同时输出：
   - raw：仅做 16bit/float -> 8bit 映射（可选 percent-stretch）
   - enhanced：在 raw 基础上做增强（按超参数选择）
3) 图像增强：提供多种增强算法，可按超参数选择；并在代码内区分“在量化前增强”和“在量化后增强”的合理阶段

依赖：
- GDAL (osgeo.gdal)
- numpy
- opencv-python (cv2)
- pillow (PIL) 仅在 --enhance pil 时需要
- tqdm (可选，用于进度条)

用法示例：
python tif2jpg_pipeline.py ^
  --input_dir "D:\data\tif" ^
  --crop_out_dir "D:\out\crop_tif" ^
  --jpg_out_dir "D:\out\jpg" ^
  --do_crop 1 --crop_size 256 --stride 256 ^
  --do_convert 1 ^
  --enhance autolevel --stretch_low 0.001 --stretch_high 0.999 ^
  --jpg_quality 95

增强方法（--enhance）：
- none        : 不增强
- autolevel   : 分位数拉伸/动态范围拉伸（类似 AutoLevel / percent stretch）
- equalize    : 直方图均衡（3通道逐通道），可选白平衡
- msr         : Multi-Scale Retinex（逐通道）
- pil         : PIL 对比度+色彩增强（适合快速增强）

注意：
- 对 3 波段图像默认按 (1,2,3) 读取并输出 RGB；如你的 tif 波段顺序不同，可用 --band_order 调整。
- 裁剪输出默认保存为 tif patch（保留地理参考）；如仅做视觉 patch 训练，也可改为输出 jpg（见 --crop_as_jpg）。
"""

import os
import argparse
import numpy as np

from osgeo import gdal

import cv2

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# -------------------------
# I/O helpers
# -------------------------
def list_tif_files(input_dir: str):
    exts = {".tif", ".tiff"}
    out = []
    for root, _, files in os.walk(input_dir):
        for fn in files:
            if os.path.splitext(fn)[1].lower() in exts:
                out.append(os.path.join(root, fn))
    return sorted(out)


def imwrite_unicode(path: str, img: np.ndarray, jpg_quality: int = 95):
    """
    OpenCV 在 Windows/中文路径下可能 cv2.imwrite 失败，使用 imencode + tofile 更稳。
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".jpg", ".jpeg"):
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)]
    else:
        params = []
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise RuntimeError(f"cv2.imencode failed: {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    buf.tofile(path)


# -------------------------
# Geo helpers (preserve georef when cropping)
# -------------------------
def _window_geotransform(gt, xoff, yoff):
    """
    根据原始 GeoTransform 与窗口偏移计算 patch 的 GeoTransform
    gt = (originX, pixelWidth, rot1, originY, rot2, pixelHeight)
    """
    # affine:
    # Xgeo = gt[0] + x*gt[1] + y*gt[2]
    # Ygeo = gt[3] + x*gt[4] + y*gt[5]
    new_gt0 = gt[0] + xoff * gt[1] + yoff * gt[2]
    new_gt3 = gt[3] + xoff * gt[4] + yoff * gt[5]
    return (new_gt0, gt[1], gt[2], new_gt3, gt[4], gt[5])


def write_tif_patch(out_path: str, patch: np.ndarray, ref_ds, xoff: int, yoff: int):
    """
    写出 tif patch，尽量保留投影与仿射变换
    patch: (H,W) 或 (H,W,3)
    """
    driver = gdal.GetDriverByName("GTiff")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if patch.ndim == 2:
        h, w = patch.shape
        bands = 1
    else:
        h, w, bands = patch.shape

    dtype_map = {
        np.dtype("uint8"): gdal.GDT_Byte,
        np.dtype("int16"): gdal.GDT_Int16,
        np.dtype("uint16"): gdal.GDT_UInt16,
        np.dtype("int32"): gdal.GDT_Int32,
        np.dtype("uint32"): gdal.GDT_UInt32,
        np.dtype("float32"): gdal.GDT_Float32,
        np.dtype("float64"): gdal.GDT_Float64,
    }
    gdal_dtype = dtype_map.get(patch.dtype, gdal.GDT_Float32)

    out_ds = driver.Create(out_path, w, h, bands, gdal_dtype, options=["COMPRESS=LZW"])
    if out_ds is None:
        raise RuntimeError(f"Failed to create: {out_path}")

    # georef
    gt = ref_ds.GetGeoTransform(can_return_null=True)
    if gt:
        out_ds.SetGeoTransform(_window_geotransform(gt, xoff, yoff))
    proj = ref_ds.GetProjection()
    if proj:
        out_ds.SetProjection(proj)

    if bands == 1:
        out_ds.GetRasterBand(1).WriteArray(patch)
    else:
        # gdal band index starts from 1
        for b in range(bands):
            out_ds.GetRasterBand(b + 1).WriteArray(patch[:, :, b])

    out_ds.FlushCache()
    out_ds = None


# -------------------------
# Scaling: tif -> uint8
# -------------------------
def percent_stretch(arr: np.ndarray, low=0.001, high=0.999, mask_zero=True, per_channel=True):
    """
    分位数拉伸到 [0,1]（float32），支持 2D/3D.
    low/high 为分位数（0~1）。例如 low=0.001 意味着丢弃 0.1% 的低端。
    """
    arr = arr.astype(np.float32, copy=False)

    def _stretch_2d(a2):
        if mask_zero:
            m = a2 != 0
            if np.any(m):
                v = a2[m]
            else:
                v = a2.reshape(-1)
        else:
            v = a2.reshape(-1)

        lo = np.quantile(v, low)
        hi = np.quantile(v, high)
        if hi <= lo:
            # fallback: min/max
            lo, hi = float(np.min(v)), float(np.max(v))
            if hi <= lo:
                return np.zeros_like(a2, dtype=np.float32)

        out = (a2 - lo) / (hi - lo)
        return np.clip(out, 0.0, 1.0)

    if arr.ndim == 2 or not per_channel:
        return _stretch_2d(arr)
    else:
        out = np.zeros_like(arr, dtype=np.float32)
        for c in range(arr.shape[2]):
            out[:, :, c] = _stretch_2d(arr[:, :, c])
        return out


def to_uint8(arr: np.ndarray, low=0.001, high=0.999, mask_zero=True, per_channel=True):
    """
    将 2D/3D 数据映射到 uint8。
    先做 percent_stretch -> [0,1]，再乘 255。
    """
    stretched = percent_stretch(arr, low=low, high=high, mask_zero=mask_zero, per_channel=per_channel)
    return (stretched * 255.0 + 0.5).astype(np.uint8)


# -------------------------
# Enhancement methods
# -------------------------
def white_balance_bgr(img_bgr: np.ndarray):
    """
    Lab 空间简单白平衡（来自你提供的 equalizeHist 脚本思路）。
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    L, a, b = cv2.split(lab)
    avg_a = np.mean(a)
    avg_b = np.mean(b)
    a = np.uint8(np.clip(a + (128 - avg_a), 0, 255))
    b = np.uint8(np.clip(b + (128 - avg_b), 0, 255))
    balanced = cv2.merge([L, a, b])
    return cv2.cvtColor(balanced, cv2.COLOR_Lab2BGR)


def equalize_hist_color(img_bgr: np.ndarray, do_white_balance: bool = True):
    """
    逐通道直方图均衡（对彩色来说更像“增强对比”，不保证颜色自然）。
    """
    x = img_bgr.copy()
    if do_white_balance:
        x = white_balance_bgr(x)
    for c in range(3):
        x[:, :, c] = cv2.equalizeHist(x[:, :, c])
    return x


def _replace_zeroes(data: np.ndarray):
    nz = data[np.nonzero(data)]
    if nz.size == 0:
        return data
    mn = float(nz.min())
    data = data.copy()
    data[data == 0] = mn
    return data


def msr_single_channel(img_u8: np.ndarray, scales=(15, 101, 301)):
    """
    Multi-Scale Retinex (MSR) 单通道版本（参考你提供的 MSR 脚本逻辑）。
    输入/输出 uint8。
    """
    img = img_u8.astype(np.float32)
    h, w = img.shape
    log_r = np.zeros((h, w), dtype=np.float32)
    weight = 1.0 / max(1, len(scales))

    for s in scales:
        img2 = _replace_zeroes(img)
        blur = cv2.GaussianBlur(img2, (int(s), int(s)), 0)
        blur = _replace_zeroes(blur)
        dst_img = cv2.log(img2 / 255.0)
        dst_blur = cv2.log(blur / 255.0)
        # 注意：你原代码用 multiply(dst_img, dst_l_blur) 再 subtract
        # 这里保持同等结构
        dst_ixl = cv2.multiply(dst_img, dst_blur)
        log_r += weight * cv2.subtract(dst_img, dst_ixl)

    out = cv2.normalize(log_r, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.convertScaleAbs(out)


def msr_color(img_bgr: np.ndarray, scales=(15, 101, 301)):
    """
    逐通道 MSR
    """
    b, g, r = cv2.split(img_bgr)
    b2 = msr_single_channel(b, scales=scales)
    g2 = msr_single_channel(g, scales=scales)
    r2 = msr_single_channel(r, scales=scales)
    return cv2.merge([b2, g2, r2])


def pil_contrast_color(img_bgr: np.ndarray, contrast_factor=1.5, color_factor=1.5):
    """
    PIL 对比度 + 色彩增强（参考你提供的 tif2jpg_enhance.py 思路）。
    """
    from PIL import Image, ImageEnhance
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    pil = ImageEnhance.Contrast(pil).enhance(float(contrast_factor))
    pil = ImageEnhance.Color(pil).enhance(float(color_factor))
    out = np.array(pil)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


def apply_enhancement(
    img_u8: np.ndarray,
    method: str,
    *,
    do_white_balance: bool = True,
    msr_scales=(15, 101, 301),
    pil_contrast=1.5,
    pil_color=1.5,
):
    """
    输入 uint8 图像，返回增强后 uint8 图像
    img_u8: (H,W) gray 或 (H,W,3) BGR
    """
    method = (method or "none").lower()

    if method == "none":
        return img_u8

    if img_u8.ndim == 2:
        # gray
        if method == "equalize":
            return cv2.equalizeHist(img_u8)
        if method == "msr":
            return msr_single_channel(img_u8, scales=msr_scales)
        if method == "pil":
            # PIL 对灰度没有 color 增强意义，这里只做对比度
            from PIL import Image, ImageEnhance
            pil = Image.fromarray(img_u8, mode="L")
            pil = ImageEnhance.Contrast(pil).enhance(float(pil_contrast))
            return np.array(pil)
        if method == "autolevel":
            # autolevel 对灰度可直接再做一次拉伸
            return to_uint8(img_u8, low=0.001, high=0.999, mask_zero=False, per_channel=False)
        raise ValueError(f"Unknown enhance method for gray: {method}")

    # color (BGR)
    if method == "equalize":
        return equalize_hist_color(img_u8, do_white_balance=do_white_balance)
    if method == "msr":
        return msr_color(img_u8, scales=msr_scales)
    if method == "pil":
        return pil_contrast_color(img_u8, contrast_factor=pil_contrast, color_factor=pil_color)
    if method == "autolevel":
        # autolevel：对 uint8 再做一次分位拉伸（一般对比度会更强）
        # 注意：这里是“量化后”增强版本；若想“量化前”增强，用主流程里的 pre_uint8_stretch
        return to_uint8(img_u8, low=0.001, high=0.999, mask_zero=False, per_channel=True)

    raise ValueError(f"Unknown enhance method: {method}")


# -------------------------
# Reading tif
# -------------------------
def read_tif_as_array(ds, band_order=(1, 2, 3)):
    """
    读取 tif 为 numpy 数组：
    - 1 band -> (H,W)
    - >=3 band -> (H,W,3) 以 band_order 组合
    返回：arr, is_color
    """
    bands = ds.RasterCount
    w, h = ds.RasterXSize, ds.RasterYSize
    if bands == 1:
        a = ds.GetRasterBand(1).ReadAsArray(0, 0, w, h)
        return a, False

    if bands < 3:
        raise ValueError(f"RasterCount={bands} (<3) 不支持按 3 通道输出")

    idxs = list(band_order)
    if len(idxs) != 3:
        raise ValueError("--band_order 必须给 3 个 band index，如 1,2,3")

    chans = []
    for i in idxs:
        if i < 1 or i > bands:
            raise ValueError(f"band index {i} out of range 1..{bands}")
        chans.append(ds.GetRasterBand(i).ReadAsArray(0, 0, w, h))

    # 这里先返回 RGB 排列的数组；但 OpenCV 习惯 BGR，我们在保存前会转换
    rgb = np.stack(chans, axis=-1)
    return rgb, True


def rgb_to_bgr(x):
    return x[:, :, ::-1].copy()


# -------------------------
# Cropping
# -------------------------
def crop_tif(
    tif_path: str,
    crop_out_dir: str,
    crop_size: int,
    stride: int,
    band_order=(1, 2, 3),
    skip_zero: bool = True,
    cover_edges: bool = False,
    crop_as_jpg: bool = False,
    jpg_quality: int = 95,
    stretch_low: float = 0.001,
    stretch_high: float = 0.999,
):
    """
    裁剪 tif 成 patch：
    - 默认输出 tif patch（保留地理参考）
    - 可选输出 jpg patch（适合训练数据），会进行 uint8 映射
    """
    ds = gdal.Open(tif_path)
    if ds is None:
        raise RuntimeError(f"无法打开: {tif_path}")

    base = os.path.splitext(os.path.basename(tif_path))[0]
    w, h = ds.RasterXSize, ds.RasterYSize
    bands = ds.RasterCount

    if bands == 1:
        is_color = False
        # window read
        def read_win(xoff, yoff, xsize, ysize):
            return ds.GetRasterBand(1).ReadAsArray(xoff, yoff, xsize, ysize)

    else:
        is_color = True
        idxs = list(band_order)
        def read_win(xoff, yoff, xsize, ysize):
            chans = []
            for i in idxs:
                chans.append(ds.GetRasterBand(i).ReadAsArray(xoff, yoff, xsize, ysize))
            return np.stack(chans, axis=-1)

    # iterate windows
    xs = list(range(0, w - crop_size + 1, stride))
    ys = list(range(0, h - crop_size + 1, stride))
    if cover_edges:
        if len(xs) == 0 or xs[-1] != w - crop_size:
            xs.append(max(0, w - crop_size))
        if len(ys) == 0 or ys[-1] != h - crop_size:
            ys.append(max(0, h - crop_size))

    total = len(xs) * len(ys)
    iterator = ((x, y) for y in ys for x in xs)
    if tqdm is not None:
        iterator = tqdm(iterator, total=total, desc=f"Crop {base}")
    idx = 0

    for xoff, yoff in iterator:
        patch = read_win(xoff, yoff, crop_size, crop_size)
        if patch is None:
            continue

        if skip_zero:
            if np.any(patch == 0):
                continue

        if crop_as_jpg:
            # 映射到 uint8
            if is_color:
                patch_u8 = to_uint8(patch, low=stretch_low, high=stretch_high, mask_zero=True, per_channel=True)
                # RGB -> BGR for cv2
                patch_bgr = rgb_to_bgr(patch_u8)
                out_path = os.path.join(crop_out_dir, f"{base}_{idx}.jpg")
                imwrite_unicode(out_path, patch_bgr, jpg_quality=jpg_quality)
                idx += 1
            else:
                patch_u8 = to_uint8(patch, low=stretch_low, high=stretch_high, mask_zero=True, per_channel=False)
                out_path = os.path.join(crop_out_dir, f"{base}_{idx}.jpg")
                imwrite_unicode(out_path, patch_u8, jpg_quality=jpg_quality)
                idx += 1
        else:
            out_path = os.path.join(crop_out_dir, f"{base}_{idx}.tif")
            write_tif_patch(out_path, patch, ds, xoff=xoff, yoff=yoff)
            idx += 1

    ds = None


# -------------------------
# Convert full tif -> jpg (raw + enhanced)
# -------------------------
def convert_tif_to_jpg(
    tif_path: str,
    jpg_out_dir: str,
    *,
    band_order=(1, 2, 3),
    enhance: str = "none",
    # scaling (tif -> uint8)
    stretch_low: float = 0.001,
    stretch_high: float = 0.999,
    # enhancement params
    do_white_balance: bool = True,
    msr_scales=(15, 101, 301),
    pil_contrast: float = 1.5,
    pil_color: float = 1.5,
    # stage control
    pre_uint8_stretch: bool = True,
    # output
    jpg_quality: int = 95,
):
    """
    输出两张：
    - raw: 仅映射到 uint8
    - enhanced: raw 上做增强（某些方法也会在 uint8 前做一次 autolevel）
    """
    ds = gdal.Open(tif_path)
    if ds is None:
        raise RuntimeError(f"无法打开: {tif_path}")

    base = os.path.splitext(os.path.basename(tif_path))[0]

    arr, is_color = read_tif_as_array(ds, band_order=band_order)

    # 关键点（对应你的问题3）：
    # - 大多数遥感 tif 是 16bit/float，先做“量化/拉伸”到 8bit 是必须步骤；
    # - 对于“动态范围/AutoLevel”这类算法，最好在量化前(高位深)做一次拉伸，再量化；
    # - 对于 equalizeHist/MSR 这类典型基于 8bit 的 CV 算法，通常在量化后做更合适；
    # 本实现中：总是先做一次 to_uint8(含分位拉伸)，作为 raw；
    # autolevel 可通过 pre_uint8_stretch=True 在量化前起到主作用（raw 的 stretch 已经承担这一点）；
    # 其他增强统一在 raw(uint8) 上做，保证输出稳定可控。

    # raw uint8
    if is_color:
        # arr 是 RGB
        raw_u8_rgb = to_uint8(arr, low=stretch_low, high=stretch_high, mask_zero=True, per_channel=True) if pre_uint8_stretch else \
                     np.clip(arr, 0, 255).astype(np.uint8)
        raw_bgr = rgb_to_bgr(raw_u8_rgb)
    else:
        raw_u8 = to_uint8(arr, low=stretch_low, high=stretch_high, mask_zero=True, per_channel=False) if pre_uint8_stretch else \
                 np.clip(arr, 0, 255).astype(np.uint8)

    # save raw
    raw_dir = os.path.join(jpg_out_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    raw_path = os.path.join(raw_dir, base + ".jpg")
    if is_color:
        imwrite_unicode(raw_path, raw_bgr, jpg_quality=jpg_quality)
    else:
        imwrite_unicode(raw_path, raw_u8, jpg_quality=jpg_quality)

    # enhanced
    enh_dir = os.path.join(jpg_out_dir, f"enhanced_{enhance.lower()}")
    os.makedirs(enh_dir, exist_ok=True)
    enh_path = os.path.join(enh_dir, base + f"_{enhance.lower()}.jpg")

    if enhance.lower() == "none":
        # 仍然输出一份，便于管线一致
        if is_color:
            imwrite_unicode(enh_path, raw_bgr, jpg_quality=jpg_quality)
        else:
            imwrite_unicode(enh_path, raw_u8, jpg_quality=jpg_quality)
        ds = None
        return raw_path, enh_path

    if is_color:
        enh_bgr = apply_enhancement(
            raw_bgr,
            enhance,
            do_white_balance=do_white_balance,
            msr_scales=msr_scales,
            pil_contrast=pil_contrast,
            pil_color=pil_color,
        )
        imwrite_unicode(enh_path, enh_bgr, jpg_quality=jpg_quality)
    else:
        enh_u8 = apply_enhancement(
            raw_u8,
            enhance,
            msr_scales=msr_scales,
            pil_contrast=pil_contrast,
            pil_color=pil_color,
        )
        imwrite_unicode(enh_path, enh_u8, jpg_quality=jpg_quality)

    ds = None
    return raw_path, enh_path


# -------------------------
# Main
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", type=str, required=True, help="输入 tif/tiff 文件夹（递归）")

    # task switches
    p.add_argument("--do_crop", type=int, default=1, help="是否裁剪：1/0")
    p.add_argument("--do_convert", type=int, default=1, help="是否转 jpg：1/0")

    # outputs
    p.add_argument("--crop_out_dir", type=str, default=None, help="裁剪 patch 输出目录（folder1）。默认：与 input_dir 同级的 crop_out_dir（若 --crop_as_jpg=1 则默认同级的 crop_as_jpg）")
    p.add_argument("--jpg_out_dir", type=str, default=None, help="jpg 输出目录（folder2）。默认：与 input_dir 同级的 jpg_out_dir")

    # crop hypers
    p.add_argument("--crop_size", type=int, default=256)
    p.add_argument("--stride", type=int, default=256)
    p.add_argument("--skip_zero", type=int, default=1, help="裁剪时若 patch 含0像素则跳过：1/0")
    p.add_argument("--cover_edges", type=int, default=0, help="是否覆盖边缘（最后一行/列补齐窗口）：1/0")
    p.add_argument("--crop_as_jpg", type=int, default=0, help="裁剪 patch 是否直接输出 jpg：1/0")

    # band order
    p.add_argument("--band_order", type=str, default="1,2,3", help="3波段读取顺序，例如 1,2,3 或 3,2,1")

    # scaling hypers
    p.add_argument("--stretch_low", type=float, default=0.001)
    p.add_argument("--stretch_high", type=float, default=0.999)

    # enhancement hypers
    p.add_argument("--enhance", type=str, default="none",
                   choices=["none", "autolevel", "equalize", "msr", "pil"],
                   help="增强方法")
    p.add_argument("--pre_uint8_stretch", type=int, default=1,
                   help="量化到8bit前是否做分位拉伸（推荐=1）：1/0")

    p.add_argument("--do_white_balance", type=int, default=1, help="equalize 时是否先白平衡：1/0")
    p.add_argument("--msr_scales", type=str, default="15,101,301", help="MSR scales，例如 15,101,301")
    p.add_argument("--pil_contrast", type=float, default=1.5)
    p.add_argument("--pil_color", type=float, default=1.5)

    # jpg output
    p.add_argument("--jpg_quality", type=int, default=95)

    return p.parse_args()


def main():
    args = parse_args()
    # ---------- default output dirs ----------
    # 默认输出目录：与 input_dir 同级（即 input_dir 的父目录下）
    in_dir = os.path.abspath(args.input_dir)
    parent_dir = os.path.dirname(in_dir.rstrip(os.sep))
    # crop_out_dir: 若用户未指定，则默认使用 parent_dir/crop_out_dir；
    # 若 --crop_as_jpg=1 且用户未指定，则默认使用 parent_dir/crop_as_jpg（便于区分裁剪输出类型）
    if args.crop_out_dir is None or str(args.crop_out_dir).strip() == "":
        if bool(args.crop_as_jpg):
            args.crop_out_dir = os.path.join(parent_dir, "crop_as_jpg")
        else:
            args.crop_out_dir = os.path.join(parent_dir, "crop_out_dir")

    # jpg_out_dir: 若用户未指定，则默认使用 parent_dir/jpg_out_dir
    if args.jpg_out_dir is None or str(args.jpg_out_dir).strip() == "":
        args.jpg_out_dir = os.path.join(parent_dir, "jpg_out_dir")

    tif_files = list_tif_files(args.input_dir)
    if len(tif_files) == 0:
        print(f"[WARN] No tif/tiff found in: {args.input_dir}")
        return

    band_order = tuple(int(x.strip()) for x in args.band_order.split(",") if x.strip())
    msr_scales = tuple(int(x.strip()) for x in args.msr_scales.split(",") if x.strip())

    os.makedirs(args.crop_out_dir, exist_ok=True)
    os.makedirs(args.jpg_out_dir, exist_ok=True)

    iterator = tif_files
    if tqdm is not None:
        iterator = tqdm(tif_files, desc="Files")

    for tif_path in iterator:
        try:
            if args.do_crop:
                crop_tif(
                    tif_path,
                    args.crop_out_dir,
                    crop_size=int(args.crop_size),
                    stride=int(args.stride),
                    band_order=band_order,
                    skip_zero=bool(args.skip_zero),
                    cover_edges=bool(args.cover_edges),
                    crop_as_jpg=bool(args.crop_as_jpg),
                    jpg_quality=int(args.jpg_quality),
                    stretch_low=float(args.stretch_low),
                    stretch_high=float(args.stretch_high),
                )

            if args.do_convert:
                convert_tif_to_jpg(
                    tif_path,
                    args.jpg_out_dir,
                    band_order=band_order,
                    enhance=args.enhance,
                    stretch_low=float(args.stretch_low),
                    stretch_high=float(args.stretch_high),
                    do_white_balance=bool(args.do_white_balance),
                    msr_scales=msr_scales,
                    pil_contrast=float(args.pil_contrast),
                    pil_color=float(args.pil_color),
                    pre_uint8_stretch=bool(args.pre_uint8_stretch),
                    jpg_quality=int(args.jpg_quality),
                )

        except Exception as e:
            print(f"[ERROR] {tif_path} -> {e}")

    print("Done.")


if __name__ == "__main__":
    main()

'''

CUDA_VISIBLE_DEVICES=0 python TIF2JPG/tif2jpg_pipeline_v3.py \
--input_dir '/home/fiko/UT/Dataset/test/input_dir' \
--crop_out_dir '/home/fiko/UT/Dataset/test/crop_out_dir' \
--jpg_out_dir '/home/fiko/UT/Dataset/test/jpg_out_dir' \
--crop_as_jpg 0 \
--enhance pil


'''