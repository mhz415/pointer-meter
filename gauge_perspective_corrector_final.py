import os
import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class CorrectionResult:
    corrected_bgr: np.ndarray
    debug_bgr: np.ndarray
    affine_matrix: np.ndarray
    ellipse: tuple
    method: str


class GaugePerspectiveCorrector:
    """
    面向“已裁剪单个表盘图像”的稳健透视/仿射校正模块。

    这版不是简单画圆，而是：
    1. 先用 Hough 圆获得表盘中心和半径的粗估计；
    2. 只在外圈环带中提取边缘点；
    3. 对环带点拟合椭圆；
    4. 用椭圆 -> 圆的仿射校正，把斜拍表盘拉成正视近似；
    5. 再增加一个“方向保持”旋转，使结果尽量接近原图正常朝向；
    6. 圆外背景统一设为白色。

    说明：
    - 对你的这类工业表盘图，这个方法通常比四点单应变换更稳定。
    - 它属于“椭圆到圆的稳健近似校正”，不是严格相机标定下的精确单应恢复。
    """

    def __init__(self, output_size=512, circle_fill=0.88):
        self.output_size = int(output_size)
        self.circle_fill = float(circle_fill)

    def _find_rough_circle(self, image_bgr):
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 2)

        h, w = gray.shape
        min_side = min(h, w)
        img_center = np.array([w / 2.0, h / 2.0], dtype=np.float32)

        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(20, min_side // 3),
            param1=120,
            param2=30,
            minRadius=int(0.26 * min_side),
            maxRadius=int(0.52 * min_side)
        )

        if circles is None:
            return float(w / 2.0), float(h / 2.0), float(0.40 * min_side)

        circles = np.squeeze(circles, axis=0)

        best = None
        best_score = -1e18
        target_r = 0.40 * min_side

        for x, y, r in circles:
            center = np.array([x, y], dtype=np.float32)
            center_dist = np.linalg.norm(center - img_center)
            radius_penalty = abs(r - target_r)

            score = -8.0 * center_dist - 8.0 * radius_penalty
            if score > best_score:
                best_score = score
                best = (float(x), float(y), float(r))

        if best is None:
            return float(w / 2.0), float(h / 2.0), float(0.40 * min_side)

        return best

    def _fit_ellipse_from_ring(self, image_bgr, cx, cy, r0):
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        h, w = gray.shape
        min_side = min(h, w)

        edges = cv2.Canny(gray, 50, 140)

        border = max(8, int(0.02 * min_side))
        edges[:border, :] = 0
        edges[-border:, :] = 0
        edges[:, :border] = 0
        edges[:, -border:] = 0

        ys, xs = np.where(edges > 0)
        if len(xs) < 80:
            return None, edges

        pts = np.stack([xs, ys], axis=1).astype(np.float32)
        center = np.array([cx, cy], dtype=np.float32)
        d = np.linalg.norm(pts - center[None, :], axis=1)

        # 只保留表盘外圈附近的环带点
        band = (d > 0.76 * r0) & (d < 1.08 * r0)
        ring_pts = pts[band]
        if len(ring_pts) < 80:
            return None, edges

        # 角度均匀采样，避免某一侧边缘点过多
        rel = ring_pts - center[None, :]
        ang = np.arctan2(rel[:, 1], rel[:, 0])
        bins = np.linspace(-np.pi, np.pi, 73)
        picked = []

        for i in range(len(bins) - 1):
            mask = (ang >= bins[i]) & (ang < bins[i + 1])
            part = ring_pts[mask]
            if len(part) == 0:
                continue
            part_d = np.linalg.norm(part - center[None, :], axis=1)
            idx = np.argsort(np.abs(part_d - r0))[:4]
            picked.append(part[idx])

        if len(picked) == 0:
            return None, edges

        fit_pts = np.concatenate(picked, axis=0).astype(np.float32)
        if len(fit_pts) < 40:
            return None, edges

        try:
            ellipse = cv2.fitEllipse(fit_pts.reshape(-1, 1, 2))
        except cv2.error:
            return None, edges

        (ex, ey), (d1, d2), _ = ellipse
        a = max(d1, d2) / 2.0
        b = min(d1, d2) / 2.0
        ratio = b / max(a, 1e-6)

        # 合理性检查
        if ratio < 0.58:
            return None, edges
        if a < 0.25 * min_side or a > 0.55 * min_side:
            return None, edges
        if np.linalg.norm(np.array([ex, ey], dtype=np.float32) - center) > 0.20 * min_side:
            return None, edges

        return ellipse, edges

    def _build_affine_matrix(self, ellipse):
        """
        核心步骤：
        用椭圆 -> 圆的线性变换进行校正，并额外加入一个旋转，
        使“原图的上方方向”在校正后仍尽量朝上。
        """
        (cx, cy), (d1, d2), angle_deg = ellipse
        a = d1 / 2.0
        b = d2 / 2.0
        theta = np.deg2rad(angle_deg)

        # 椭圆参数矩阵：x = c + A s, 其中 s 在单位圆上
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ], dtype=np.float32)
        A = R @ np.diag([a, b]).astype(np.float32)
        A_inv = np.linalg.inv(A)

        # 选择一个额外旋转 Q，使校正后“图像的上方向”尽量保持为上
        up = np.array([0.0, -1.0], dtype=np.float32)
        v = A_inv @ up
        ang = np.arctan2(v[1], v[0])
        target = -np.pi / 2.0
        alpha = target - ang
        Q = np.array([
            [np.cos(alpha), -np.sin(alpha)],
            [np.sin(alpha),  np.cos(alpha)]
        ], dtype=np.float32)

        s = self.output_size
        out_center = np.array([(s - 1) / 2.0, (s - 1) / 2.0], dtype=np.float32)
        out_radius = (s * self.circle_fill) / 2.0

        M = np.zeros((2, 3), dtype=np.float32)
        M[:, :2] = out_radius * (Q @ A_inv)
        M[:, 2] = out_center - out_radius * (Q @ A_inv) @ np.array([cx, cy], dtype=np.float32)
        return M

    def correct(self, image_bgr, debug=True):
        cx, cy, r0 = self._find_rough_circle(image_bgr)
        ellipse, edges = self._fit_ellipse_from_ring(image_bgr, cx, cy, r0)

        method = "ellipse_affine_oriented"
        if ellipse is None:
            # 回退：直接用粗圆，至少完成居中和尺度标准化
            ellipse = ((cx, cy), (2.0 * r0, 2.0 * r0), 0.0)
            method = "circle_fallback"

        M = self._build_affine_matrix(ellipse)

        corrected = cv2.warpAffine(
            image_bgr,
            M,
            (self.output_size, self.output_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)
        )

        # 圆外区域设为白色
        s = self.output_size
        mask = np.zeros((s, s), dtype=np.uint8)
        c = (s - 1) // 2
        rr = int(0.44 * s)
        cv2.circle(mask, (c, c), rr, 255, -1)
        white = np.full_like(corrected, 255)
        corrected = np.where(mask[:, :, None] == 255, corrected, white)

        debug_img = None
        if debug:
            debug_img = image_bgr.copy()
            cv2.ellipse(debug_img, ellipse, (0, 255, 0), 2)
            cv2.circle(debug_img, (int(round(cx)), int(round(cy))), 4, (255, 255, 0), -1)
            cv2.putText(
                debug_img,
                method,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

        return CorrectionResult(
            corrected_bgr=corrected,
            debug_bgr=debug_img,
            affine_matrix=M,
            ellipse=ellipse,
            method=method
        )


def batch_correct(input_dir, output_dir, output_size=512, debug=True):
    os.makedirs(output_dir, exist_ok=True)
    if debug:
        os.makedirs(os.path.join(output_dir, "debug"), exist_ok=True)

    corrector = GaugePerspectiveCorrector(output_size=output_size)

    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    for fname in os.listdir(input_dir):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in valid_ext:
            continue

        in_path = os.path.join(input_dir, fname)
        img = cv2.imread(in_path)
        if img is None:
            print(f"[跳过] 无法读取图像: {fname}")
            continue

        try:
            result = corrector.correct(img, debug=debug)
        except Exception as e:
            print(f"[失败] {fname}: {e}")
            continue

        stem = os.path.splitext(fname)[0]
        out_path = os.path.join(output_dir, f"{stem}_corrected{ext}")
        cv2.imwrite(out_path, result.corrected_bgr)

        if debug and result.debug_bgr is not None:
            dbg_path = os.path.join(output_dir, "debug", f"{stem}_debug{ext}")
            cv2.imwrite(dbg_path, result.debug_bgr)

        print(f"[OK] {fname} ({result.method})")


if __name__ == "__main__":
    img_path = "test.jpg"
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        corrector = GaugePerspectiveCorrector(output_size=512)
        result = corrector.correct(img, debug=True)
        cv2.imwrite("test_corrected.jpg", result.corrected_bgr)
        if result.debug_bgr is not None:
            cv2.imwrite("test_debug.jpg", result.debug_bgr)
        print("单张测试完成：", result.method)
