import os
import cv2
import numpy as np
from osgeo import gdal

def normalization(data):
    _min = np.min(data)
    _max = np.max(data)
    _range = _max - _min
    if _range == 0:
        return np.zeros_like(data, dtype=np.float32)
    return (data - _min) / _range

def imgto8bit(img):
    img_nrm = normalization(img)
    img_8 = cv2.equalizeHist(np.uint8(255 * img_nrm))
    return img_8

def tif_to_jpg_3band(rasterfile):
    """3波段：按(1,2,3)当作(B,G,R)合成彩色"""
    in_ds = gdal.Open(rasterfile)
    if in_ds is None:
        raise RuntimeError(f"GDAL无法打开文件: {rasterfile}")

    xsize = in_ds.RasterXSize
    ysize = in_ds.RasterYSize
    bands = in_ds.RasterCount
    if bands < 3:
        raise RuntimeError(f"要求>=3波段，但当前只有 {bands} 波段: {rasterfile}")

    B = in_ds.GetRasterBand(1).ReadAsArray(0, 0, xsize, ysize).astype(np.float32)
    G = in_ds.GetRasterBand(2).ReadAsArray(0, 0, xsize, ysize).astype(np.float32)
    R = in_ds.GetRasterBand(3).ReadAsArray(0, 0, xsize, ysize).astype(np.float32)

    B1 = imgto8bit(B)
    G1 = imgto8bit(G)
    R1 = imgto8bit(R)

    return cv2.merge([B1, G1, R1])  # BGR for cv2

def tif_to_jpg_1band(rasterfile, as_3channel=False):
    """
    1波段：输出灰度JPG（默认单通道）
    as_3channel=True 时，把灰度复制到3通道，便于某些下游只吃RGB
    """
    in_ds = gdal.Open(rasterfile)
    if in_ds is None:
        raise RuntimeError(f"GDAL无法打开文件: {rasterfile}")

    xsize = in_ds.RasterXSize
    ysize = in_ds.RasterYSize
    bands = in_ds.RasterCount
    if bands < 1:
        raise RuntimeError(f"文件无波段: {rasterfile}")

    band1 = in_ds.GetRasterBand(1).ReadAsArray(0, 0, xsize, ysize).astype(np.float32)
    gray8 = imgto8bit(band1)  # (H,W) uint8

    if as_3channel:
        return cv2.merge([gray8, gray8, gray8])  # (H,W,3)
    return gray8  # (H,W)

def tif_to_jpg(rasterfile, band_mode=3):
    """统一入口：按 band_mode=1 或 3 分流"""
    in_ds = gdal.Open(rasterfile)
    if in_ds is None:
        raise RuntimeError(f"GDAL无法打开文件: {rasterfile}")
    bands = in_ds.RasterCount

    if band_mode == 3:
        # 如果用户指定3波段但实际不足，给出明确错误
        if bands < 3:
            raise RuntimeError(f"BAND_MODE=3 但该tif只有 {bands} 个波段: {rasterfile}")
        return tif_to_jpg_3band(rasterfile)

    if band_mode == 1:
        if bands < 1:
            raise RuntimeError(f"BAND_MODE=1 但该tif无波段: {rasterfile}")
        return tif_to_jpg_1band(rasterfile, as_3channel=False)

    raise ValueError("band_mode 只能是 1 或 3")

if __name__ == '__main__':
    path = '/home/fiko/UT/dinov3/data/masks'

    # 超参数：选择按1波段还是3波段处理
    BAND_MODE = 1          # 1 或 3

    # 是否保留原始 tif/tiff
    KEEP_TIF = True

    # 是否覆盖已有 jpg
    OVERWRITE_JPG = True

    for fname in os.listdir(path):
        lower = fname.lower()
        if not (lower.endswith('.tif') or lower.endswith('.tiff')):
            continue

        src = os.path.join(path, fname)
        base, _ = os.path.splitext(fname)
        dst = os.path.join(path, base + ".jpg")

        if (not OVERWRITE_JPG) and os.path.exists(dst):
            print("Skip (exists):", dst)
            continue

        try:
            img = tif_to_jpg(src, band_mode=BAND_MODE)

            # imencode 对灰度/彩色都支持
            cv2.imencode('.jpg', img)[1].tofile(dst)
            print("Saved:", dst)

            if not KEEP_TIF:
                os.remove(src)
                print("Removed source:", src)

        except Exception as e:
            print("Failed:", src)
            print("Error:", repr(e))
