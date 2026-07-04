"""
太阳暗区检测器 - 用CV找到日面上真实的暗区位置

核心策略：
1. CV检测所有真实暗区（作为位置真相）
2. AI特征描述（类型、标签）与CV暗区匹配
3. 所有CV暗区都生成特征，确保不遗漏
"""

import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from scipy import ndimage
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)


@dataclass
class DarkRegion:
    """暗区信息"""
    center_x: float  # 归一化坐标
    center_y: float  # 归一化坐标
    size: float      # 相对大小（占日面直径比例）
    darkness: float  # 暗度（0-1，越暗越大）
    pixel_count: int # 像素数
    
    def to_dict(self) -> Dict:
        return {
            "center_x": self.center_x,
            "center_y": self.center_y,
            "size": self.size,
            "darkness": self.darkness,
            "pixel_count": self.pixel_count,
        }


def detect_dark_regions(
    image_path: str,
    disk_info: Optional[Dict] = None,
    min_size_ratio: float = 0.005,
    max_regions: int = 30,
) -> List[DarkRegion]:
    """
    检测日面上的暗区
    
    使用双重策略：
    1. 全局阈值（P15百分位）捕捉最暗核心
    2. 局部对比度（高斯模糊背景减除）捕捉半影区域
    
    返回像素坐标（pixel coordinates）
    """
    try:
        img = Image.open(image_path).convert("RGB")
        img_array = np.array(img)
        h, w = img_array.shape[:2]
        
        # 转灰度
        gray = np.array(img.convert("L")).astype(float)
        
        # 如果提供了日面信息，创建日面掩码
        mask = np.ones((h, w), dtype=bool)
        if disk_info:
            cx = disk_info.get("center_x", w / 2)
            cy = disk_info.get("center_y", h / 2)
            r = disk_info.get("radius", min(w, h) / 2)
            
            # 创建日面掩码 - 排除边缘15%（避免临边昏暗误检）
            yy, xx = np.ogrid[:h, :w]
            dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
            mask = dist <= r * 0.85  # 只检测日面内部85%区域
        
        # 计算日面平均亮度（用于阈值）
        disk_pixels = gray[mask]
        if len(disk_pixels) == 0:
            logger.warning("No pixels in disk mask!")
            return []
        
        mean_brightness = np.mean(disk_pixels)
        std_brightness = np.std(disk_pixels)
        
        logger.debug(f"Brightness stats: mean={mean_brightness:.1f}, std={std_brightness:.1f}")
        
        # 策略1: 全局阈值 - 捕捉最暗核心
        threshold_global = np.percentile(disk_pixels, 15)
        dark_mask_core = (gray < threshold_global) & mask
        
        # 策略2: 局部对比度 - 用高斯模糊得到局部背景，找比局部背景暗的区域
        # 只处理日面内部（外部设为均值避免边缘效应）
        gray_for_blur = gray.copy()
        gray_for_blur[~mask] = mean_brightness
        local_background = gaussian_filter(gray_for_blur, sigma=20)
        
        # 局部暗度：比局部背景暗30以上的像素
        local_dark = (local_background - gray) > 30
        dark_mask_local = local_dark & mask
        
        # 合并两种mask
        dark_mask = dark_mask_core | dark_mask_local
        
        # 形态学膨胀：合并邻近的暗像素
        dark_mask_dilated = ndimage.binary_dilation(dark_mask, iterations=4)
        
        # 连通区域分析（在膨胀后的mask上）
        labeled, num_features = ndimage.label(dark_mask_dilated)
        if num_features == 0:
            return []
        
        # 计算每个区域的属性
        regions = []
        for i in range(1, num_features + 1):
            region_mask_dilated = (labeled == i)
            
            # 用原始mask（未膨胀）计算真实暗像素
            region_mask_original = region_mask_dilated & dark_mask
            pixel_count = np.sum(region_mask_original)
            
            # 过滤太小的区域（降低到10像素）
            if pixel_count < 10:
                continue
            
            # 计算质心（用膨胀后的mask）- 像素坐标
            ys, xs = np.where(region_mask_dilated)
            cy_px = float(np.mean(ys))
            cx_px = float(np.mean(xs))
            
            # 计算平均暗度（用原始mask）
            darkness = 1.0 - (np.mean(gray[region_mask_original]) / 255.0)
            
            # 计算尺寸（占日面直径比例）
            if disk_info:
                disk_diameter = disk_info.get("radius", min(w, h) / 2) * 2
            else:
                disk_diameter = min(w, h) * 0.7
            
            # 用等效直径估算
            area = pixel_count
            equivalent_diameter = np.sqrt(4 * area / np.pi)
            size_ratio = equivalent_diameter / disk_diameter
            
            # 过滤太小的区域
            if size_ratio < min_size_ratio:
                continue
            
            # 过滤太大的区域（可能是背景伪影）
            if size_ratio > 0.25:
                logger.debug(f"Filtered oversized region: size={size_ratio:.3f}, pixels={pixel_count}")
                continue
            
            # 存储像素坐标（pixel coordinates）
            regions.append(DarkRegion(
                center_x=cx_px,
                center_y=cy_px,
                size=size_ratio,
                darkness=darkness,
                pixel_count=pixel_count,
            ))
        
        # 按暗度排序，取最暗的max_regions个
        regions.sort(key=lambda r: r.darkness, reverse=True)
        return regions[:max_regions]
        
    except Exception as e:
        logger.error(f"暗区检测失败: {e}")
        return []


def match_features_to_dark_regions(
    features: List[Dict],
    dark_regions: List[DarkRegion],
    disk_info: Optional[Dict] = None,
    max_distance_ratio: float = 0.15,
    image_width: int = 0,
    image_height: int = 0,
) -> List[Dict]:
    """
    将AI特征与CV暗区匹配，并补充AI遗漏的特征
    
    坐标系统：
    - CV暗区：像素坐标 (pixel coordinates)
    - AI特征：0-1归一化坐标 (normalized coordinates)
    - 输出：像素坐标 (pixel coordinates)，供标注函数直接使用
    
    策略：
    1. 尝试将AI特征匹配到最近的CV暗区（修正坐标）
    2. 未匹配的CV暗区生成新特征（补充AI遗漏）
    3. 未匹配的AI特征保留（可能是亮特征如谱斑）
    """
    if not dark_regions:
        logger.warning("未检测到真实暗区，保留AI原始坐标")
        return features
    
    used_regions = set()
    matched_features = []
    
    # 分离暗特征和非暗特征
    dark_features = []
    non_dark_features = []
    for feat in features:
        ftype = feat.get("type", "")
        if ftype in ["sunspot", "filament", "coronal_hole"]:
            dark_features.append(feat)
        else:
            non_dark_features.append(feat)
    
    # 辅助函数：AI归一化坐标转像素坐标
    def ai_to_pixel(ai_x: float, ai_y: float) -> Tuple[float, float]:
        if image_width > 0 and image_height > 0:
            return ai_x * image_width, ai_y * image_height
        return ai_x, ai_y
    
    # 辅助函数：像素坐标转归一化坐标
    def pixel_to_norm(px_x: float, px_y: float) -> Tuple[float, float]:
        if image_width > 0 and image_height > 0:
            return px_x / image_width, px_y / image_height
        return px_x, px_y
    
    # 辅助函数：计算归一化距离（用于匹配阈值）
    def norm_distance(ai_x: float, ai_y: float, px_x: float, px_y: float) -> float:
        ai_px_x, ai_px_y = ai_to_pixel(ai_x, ai_y)
        dist_px = np.sqrt((ai_px_x - px_x)**2 + (ai_px_y - px_y)**2)
        # 转换为归一化距离
        if image_width > 0 and image_height > 0:
            return dist_px / np.sqrt(image_width**2 + image_height**2)
        return dist_px
    
    # 策略1: 基于距离匹配AI暗特征到CV暗区
    for feat in dark_features:
        pos = feat.get("position", {})
        ai_x = pos.get("x", 0.5)
        ai_y = pos.get("y", 0.5)
        
        best_region = None
        best_distance = float("inf")
        best_idx = None
        
        for idx, region in enumerate(dark_regions):
            if idx in used_regions:
                continue
            
            dist = norm_distance(ai_x, ai_y, region.center_x, region.center_y)
            
            if dist < best_distance:
                best_distance = dist
                best_region = region
                best_idx = idx
        
        if best_region and best_distance < max_distance_ratio:
            # 距离匹配成功 - 保持归一化坐标，同时存储像素坐标
            used_regions.add(best_idx)
            # 转换像素坐标为归一化坐标
            norm_x, norm_y = pixel_to_norm(best_region.center_x, best_region.center_y)
            feat["position"] = {"x": norm_x, "y": norm_y}
            feat["pixel_position"] = {"x": best_region.center_x, "y": best_region.center_y}
            feat["cv_matched"] = True
            feat["cv_darkness"] = best_region.darkness
            feat["cv_size"] = best_region.size
            logger.info(f"距离匹配: '{feat.get('label', '')}' -> norm({norm_x:.3f},{norm_y:.3f}) pixel({best_region.center_x:.0f},{best_region.center_y:.0f})")
            matched_features.append(feat)
    
    # 策略2: 对未匹配的AI暗特征，用最佳分配
    unmatched_ai = [f for f in dark_features if not f.get("cv_matched")]
    
    if unmatched_ai:
        # 按置信度降序
        unmatched_ai.sort(key=lambda f: f.get("confidence", 0), reverse=True)
        
        # 获取未使用的CV暗区，按暗度降序
        available = [(i, r) for i, r in enumerate(dark_regions) if i not in used_regions]
        available.sort(key=lambda x: x[1].darkness, reverse=True)
        
        for feat in unmatched_ai:
            if not available:
                break
            
            idx, region = available.pop(0)
            used_regions.add(idx)
            
            old_pos = feat["position"]
            # 转换像素坐标为归一化坐标
            norm_x, norm_y = pixel_to_norm(region.center_x, region.center_y)
            feat["position"] = {"x": norm_x, "y": norm_y}
            feat["pixel_position"] = {"x": region.center_x, "y": region.center_y}
            feat["cv_matched"] = True
            feat["cv_darkness"] = region.darkness
            feat["cv_size"] = region.size
            feat["cv_fallback"] = True
            
            logger.info(f"最佳分配: '{feat.get('label', '')}' ({old_pos.get('x',0):.3f},{old_pos.get('y',0):.3f}) -> norm({norm_x:.3f},{norm_y:.3f}) pixel({region.center_x:.0f},{region.center_y:.0f})")
            matched_features.append(feat)
    
    # 策略3: 为未匹配的CV暗区生成新特征（补充AI遗漏）
    unmatched_regions = [(i, r) for i, r in enumerate(dark_regions) if i not in used_regions]
    
    for idx, region in unmatched_regions:
        # 根据暗度和大小生成标签
        if region.size > 0.05:
            label = f"大型暗区 (CV检测)"
        elif region.size > 0.02:
            label = f"中型暗区 (CV检测)"
        else:
            label = f"小型暗区 (CV检测)"
        
        # 存储归一化坐标和像素坐标
        norm_x, norm_y = pixel_to_norm(region.center_x, region.center_y)
        new_feat = {
            "type": "sunspot",
            "label": label,
            "position": {"x": norm_x, "y": norm_y},
            "pixel_position": {"x": region.center_x, "y": region.center_y},
            "size_relative": region.size,
            "confidence": 0.6,  # CV检测的置信度较低
            "cv_only": True,
            "cv_darkness": region.darkness,
            "cv_size": region.size,
            "additional_params": {"source": "cv"},
        }
        matched_features.append(new_feat)
        logger.info(f"CV补充: '{label}' at norm({norm_x:.3f},{norm_y:.3f}) pixel({region.center_x:.0f},{region.center_y:.0f})")
    
    # 合并所有特征
    final_features = matched_features + non_dark_features
    
    return final_features
