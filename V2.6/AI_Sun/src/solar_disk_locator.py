"""
太阳圆盘定位与坐标验证模块

功能：
1. 自动检测太阳圆盘边界（圆心和半径）
2. 验证特征坐标是否在日面内
3. 精修AI返回的坐标位置
4. 提供日面约束的可视化信息
"""

import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SolarDiskInfo:
    """太阳圆盘信息"""
    center_x: float  # 圆心x（像素）
    center_y: float  # 圆心y（像素）
    radius: float    # 半径（像素）
    confidence: float  # 检测置信度 0-1
    image_width: int
    image_height: int
    
    def to_dict(self) -> Dict:
        return {
            "center_x": self.center_x,
            "center_y": self.center_y,
            "radius": self.radius,
            "confidence": self.confidence,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "normalized_center_x": self.center_x / self.image_width,
            "normalized_center_y": self.center_y / self.image_height,
            "normalized_radius": self.radius / max(self.image_width, self.image_height),
        }
    
    def is_point_inside(self, x: float, y: float, tolerance: float = 0.0) -> bool:
        """检查点是否在日面内（归一化坐标）"""
        px = x * self.image_width
        py = y * self.image_height
        dist = np.sqrt((px - self.center_x)**2 + (py - self.center_y)**2)
        return dist <= (self.radius + tolerance)
    
    def clamp_to_disk(self, x: float, y: float) -> Tuple[float, float]:
        """将坐标约束到日面内（归一化坐标）"""
        px = x * self.image_width
        py = y * self.image_height
        dist = np.sqrt((px - self.center_x)**2 + (py - self.center_y)**2)
        
        if dist <= self.radius:
            return x, y  # 已在日面内
        
        # 投影到日面边界
        ratio = self.radius / dist
        new_px = self.center_x + (px - self.center_x) * ratio
        new_py = self.center_y + (py - self.center_y) * ratio
        
        return new_px / self.image_width, new_py / self.image_height


def detect_solar_disk(image_path: str, method: str = "auto") -> Optional[SolarDiskInfo]:
    """
    检测太阳圆盘
    
    方法：
    - auto: 自动选择最佳方法
    - hough: Hough圆检测
    - threshold: 阈值+轮廓检测
    - radial: 径向亮度分析
    
    Returns:
        SolarDiskInfo 或 None（检测失败）
    """
    try:
        img = Image.open(image_path).convert("RGB")
        img_array = np.array(img)
        h, w = img_array.shape[:2]
        
        # 转灰度
        gray = np.array(img.convert("L"))
        
        if method == "auto":
            # 尝试多种方法，返回置信度最高的
            results = []
            
            # 方法1: 阈值检测（适用于有明显边界的图像）
            r1 = _detect_by_threshold(gray, w, h)
            if r1:
                results.append(r1)
            
            # 方法2: 径向亮度分析（适用于渐变边缘）
            r2 = _detect_by_radial(gray, w, h)
            if r2:
                results.append(r2)
            
            if results:
                return max(results, key=lambda r: r.confidence)
        elif method == "threshold":
            return _detect_by_threshold(gray, w, h)
        elif method == "radial":
            return _detect_by_radial(gray, w, h)
        elif method == "hough":
            return _detect_by_hough(gray, w, h)
        
    except Exception as e:
        logger.error(f"日面检测失败: {e}")
    
    return None


def _detect_by_threshold(gray: np.ndarray, w: int, h: int) -> Optional[SolarDiskInfo]:
    """阈值法检测日面"""
    try:
        # 自适应阈值
        mean_val = np.mean(gray)
        std_val = np.std(gray)
        threshold = mean_val - 0.5 * std_val
        
        # 二值化
        binary = (gray > threshold).astype(np.uint8)
        
        # 找最大连通区域
        from scipy import ndimage
        labeled, num = ndimage.label(binary)
        if num == 0:
            return None
        
        sizes = ndimage.sum(binary, labeled, range(1, num+1))
        largest = np.argmax(sizes) + 1
        
        # 获取该区域的边界
        mask = (labeled == largest)
        ys, xs = np.where(mask)
        
        if len(xs) < 100:
            return None
        
        # 拟合圆
        cx = np.mean(xs)
        cy = np.mean(ys)
        
        # 计算半径（使用面积）
        area = len(xs)
        radius = np.sqrt(area / np.pi)
        
        # 验证：检查圆形度
        perimeter = 0
        # 简化：用边界框估算
        bbox_w = xs.max() - xs.min()
        bbox_h = ys.max() - ys.min()
        circularity = 4 * np.pi * area / (bbox_w * bbox_h + 1e-6)
        
        confidence = min(1.0, circularity / 3.0)  # 理想圆=π≈3.14
        
        return SolarDiskInfo(
            center_x=cx,
            center_y=cy,
            radius=radius,
            confidence=confidence,
            image_width=w,
            image_height=h,
        )
    except Exception as e:
        logger.debug(f"阈值法检测失败: {e}")
        return None


def _detect_by_radial(gray: np.ndarray, w: int, h: int) -> Optional[SolarDiskInfo]:
    """径向亮度分析法 - 先用亮度质心定位日面中心"""
    try:
        # Step 1: 用亮度阈值找日面大致区域，计算质心
        mean_val = np.mean(gray)
        std_val = np.std(gray)
        threshold = mean_val - 0.3 * std_val
        bright_mask = gray > threshold
        
        ys, xs = np.where(bright_mask)
        if len(xs) < 1000:
            return None
        
        # 用亮度加权质心
        weights = gray[bright_mask].astype(float)
        cx_guess = int(np.average(xs, weights=weights))
        cy_guess = int(np.average(ys, weights=weights))
        
        # Step 2: 从质心向外径向扫描
        angles = np.linspace(0, 2*np.pi, 72, endpoint=False)
        radii = []
        
        for angle in angles:
            dx, dy = np.cos(angle), np.sin(angle)
            max_r = min(w, h) // 2
            
            sample_r = np.arange(5, max_r, 1)
            intensities = []
            valid_r = []
            
            for r in sample_r:
                x = int(cx_guess + r * dx)
                y = int(cy_guess + r * dy)
                if 0 <= x < w and 0 <= y < h:
                    intensities.append(float(gray[y, x]))
                    valid_r.append(r)
            
            if len(intensities) < 20:
                continue
            
            intensities = np.array(intensities)
            
            # 找亮度从亮到暗的突变点（日面边缘）
            # 用滑动窗口平均找下降沿
            window = 5
            smoothed = np.convolve(intensities, np.ones(window)/window, mode='valid')
            
            if len(smoothed) < 10:
                continue
            
            # 找最大负梯度
            gradient = np.diff(smoothed)
            min_idx = np.argmin(gradient)
            
            if gradient[min_idx] < -5:
                radii.append(valid_r[min_idx + window // 2])
        
        if len(radii) < 18:  # 至少一半方向有效
            return None
        
        # 用中位数和一致性过滤异常值
        median_r = np.median(radii)
        filtered = [r for r in radii if 0.5 * median_r < r < 1.5 * median_r]
        
        if len(filtered) < 10:
            return None
        
        radius = np.median(filtered)
        radius_std = np.std(filtered)
        confidence = max(0.3, 1.0 - radius_std / (radius + 1e-6))
        
        return SolarDiskInfo(
            center_x=cx_guess,
            center_y=cy_guess,
            radius=radius,
            confidence=confidence,
            image_width=w,
            image_height=h,
        )
    except Exception as e:
        logger.debug(f"径向分析法检测失败: {e}")
        return None


def _detect_by_hough(gray: np.ndarray, w: int, h: int) -> Optional[SolarDiskInfo]:
    """Hough圆检测（需要OpenCV）"""
    try:
        import cv2
        
        # 高斯模糊
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Hough圆检测
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=min(w, h) // 4,
            param1=50,
            param2=30,
            minRadius=min(w, h) // 4,
            maxRadius=min(w, h) // 2,
        )
        
        if circles is None or len(circles) == 0:
            return None
        
        # 取最大的圆
        circles = np.round(circles[0, :]).astype("int")
        largest = max(circles, key=lambda c: c[2])
        
        cx, cy, r = largest
        
        return SolarDiskInfo(
            center_x=cx,
            center_y=cy,
            radius=r,
            confidence=0.9,
            image_width=w,
            image_height=h,
        )
    except ImportError:
        logger.debug("OpenCV不可用，跳过Hough检测")
        return None
    except Exception as e:
        logger.debug(f"Hough检测失败: {e}")
        return None


def validate_and_refine_features(
    features: List[Dict],
    disk_info: SolarDiskInfo,
    image_path: str,
    refinement_enabled: bool = True,
) -> List[Dict]:
    """
    验证并精修特征坐标
    
    Args:
        features: AI返回的特征列表（归一化坐标）
        disk_info: 日面信息
        image_path: 图像路径（用于精修）
        refinement_enabled: 是否启用位置精修
    
    Returns:
        验证后的特征列表（已过滤日面外的特征）
    """
    validated = []
    
    for feat in features:
        pos = feat.get("position", {})
        x = pos.get("x", 0.5)
        y = pos.get("y", 0.5)
        
        # 1. 检查是否在日面内
        if not disk_info.is_point_inside(x, y, tolerance=0.02):  # 2%容差
            logger.info(f"过滤日面外特征: {feat.get('label', 'unknown')} at ({x:.3f}, {y:.3f})")
            continue
        
        # 2. 位置精修（可选）
        if refinement_enabled:
            x_refined, y_refined = _refine_position(image_path, x, y, disk_info, feat.get("type", ""))
            if x_refined is not None:
                x, y = x_refined, y_refined
        
        # 3. 更新坐标
        feat["position"] = {"x": x, "y": y}
        feat["validated"] = True
        validated.append(feat)
    
    return validated


def _refine_position(
    image_path: str,
    x: float,
    y: float,
    disk_info: SolarDiskInfo,
    feature_type: str,
) -> Tuple[Optional[float], Optional[float]]:
    """
    使用图像处理精修特征位置
    
    对于黑子：在AI坐标附近找最暗的区域
    对于亮区：在AI坐标附近找最亮的区域
    """
    try:
        img = Image.open(image_path).convert("L")
        img_array = np.array(img)
        h, w = img_array.shape[:2]
        
        # AI坐标转像素
        px = int(x * w)
        py = int(y * h)
        
        # 搜索窗口（日面半径的5%）
        window = int(disk_info.radius * 0.05)
        window = max(10, min(window, 50))  # 限制在10-50像素
        
        # 提取搜索区域
        y1, y2 = max(0, py - window), min(h, py + window)
        x1, x2 = max(0, px - window), min(w, px + window)
        
        region = img_array[y1:y2, x1:x2]
        
        if region.size == 0:
            return None, None
        
        # 根据特征类型找极值点
        if feature_type in ["sunspot", "filament", "coronal_hole"]:
            # 暗特征：找最小值
            idx = np.argmin(region)
        else:
            # 亮特征：找最大值
            idx = np.argmax(region)
        
        # 转回坐标
        local_y, local_x = np.unravel_index(idx, region.shape)
        refined_px = x1 + local_x
        refined_py = y1 + local_y
        
        # 转回归一化坐标
        refined_x = refined_px / w
        refined_y = refined_py / h
        
        # 验证：精修后仍需在日面内
        if disk_info.is_point_inside(refined_x, refined_y, tolerance=0.02):
            return refined_x, refined_y
        
    except Exception as e:
        logger.debug(f"位置精修失败: {e}")
    
    return None, None


def get_disk_overlay_info(disk_info: SolarDiskInfo) -> Dict:
    """获取日面叠加层信息（用于前端可视化）"""
    return {
        "type": "solar_disk",
        "center": {
            "x": disk_info.normalized_center_x,
            "y": disk_info.normalized_center_y,
        },
        "radius": disk_info.normalized_radius,
        "confidence": disk_info.confidence,
    }
