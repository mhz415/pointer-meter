import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import os
import math
import easyocr
from tqdm import tqdm
from skimage.morphology import skeletonize

# 从你的模型文件导入
from model import DeepLab


class Predictor:
    def __init__(self, weight_path, num_classes=3, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"正在加载模型至: {self.device}...")
        self.model = DeepLab(num_classes=num_classes, backbone='mobilenetv4')

        checkpoint = torch.load(weight_path, map_location=self.device)
        state_dict = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint

        # 这里保持你原来的加载方式
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device).eval()

        print("正在初始化 OCR 引擎...")
        self.reader = easyocr.Reader(['en'])

        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # 零位/满量程吸附角度容差，先给一个稳定初值 2.5°
        self.endpoint_eps = math.radians(2.5)

    def _safe_thinning(self, binary_mask):
        """
        安全细化：
        1) 优先使用 cv2.ximgproc.thinning
        2) 如果当前 OpenCV 没有该接口，则退回 skimage skeletonize
        """
        if hasattr(cv2, 'ximgproc') and hasattr(cv2.ximgproc, 'thinning'):
            return cv2.ximgproc.thinning(binary_mask)

        sk = skeletonize(binary_mask > 0)
        return (sk.astype(np.uint8) * 255)

    @staticmethod
    def _circ_dist(a, b):
        """
        计算两个角度在圆上的最小夹角，范围 [0, pi]
        """
        d = abs(a - b) % (2 * math.pi)
        return min(d, 2 * math.pi - d)

    def _read_value_from_angle(self, ua, start_angle, end_angle, total_angle_range, max_val):
        """
        更稳的角度->读数映射：
        1) 零位附近吸附到 0
        2) 满量程附近吸附到 max_val
        3) 若角度落在有效量程弧段外，则按离起点/终点最近原则决定归零还是归满
        """
        eps = self.endpoint_eps
        rel = (ua - start_angle + 2 * math.pi) % (2 * math.pi)

        # 先做端点吸附
        if self._circ_dist(ua, start_angle) <= eps:
            return 0.0
        if self._circ_dist(ua, end_angle) <= eps:
            return max_val

        # 正常落在有效量程弧段内
        if rel <= total_angle_range:
            return (rel / max(total_angle_range, 1e-6)) * max_val

        # 落在弧段外，按最近端点处理
        if self._circ_dist(ua, start_angle) < self._circ_dist(ua, end_angle):
            return 0.0
        else:
            return max_val

    def _fit_dial_circle(self, s_mask):
        """
        核心改进 1：利用刻度区域拟合表盘圆心和半径
        """
        h, w = s_mask.shape
        mask_clean = s_mask.copy()

        # 抹除中心区域，减少文字和中心结构干扰
        cv2.circle(mask_clean, (w // 2, h // 2), int(min(w, h) * 0.30), 0, -1)

        pts = np.column_stack(np.where(mask_clean > 0))
        if len(pts) < 10:
            return None, None

        pts = pts[:, [1, 0]]  # (row, col) -> (x, y)
        ellipse = cv2.fitEllipse(pts)
        (ox, oy), (ma, MA), angle = ellipse

        return (int(ox), int(oy)), int(MA / 2)

    def _get_range_by_spatial_constraint(self, image_np, O, R, end_angle):
        """
        在终点扇区附近提取 OCR 量程值
        """
        results = self.reader.readtext(image_np)
        candidates = []

        standards = [0.1, 0.16, 0.25, 0.4, 0.6, 1.0, 1.6, 2.5, 4.0, 6.0,
                     10, 16, 25, 40, 60, 100, 250, 400, 600]

        for (bbox, text, prob) in results:
            clean = ''.join(c for c in text if c.isdigit() or c == '.')
            try:
                val = float(clean)
                if 0 < val < 1000:
                    box_center = np.mean(bbox, axis=0)

                    dist = math.sqrt((box_center[0] - O[0]) ** 2 + (box_center[1] - O[1]) ** 2)
                    ocr_angle = (math.atan2(box_center[1] - O[1], box_center[0] - O[0]) + 2 * math.pi) % (2 * math.pi)

                    angle_diff = abs(ocr_angle - end_angle)
                    if angle_diff > math.pi:
                        angle_diff = 2 * math.pi - angle_diff

                    if 0.6 * R < dist < 1.4 * R and angle_diff < (math.pi / 6):
                        candidates.append(val)
            except Exception:
                continue

        if not candidates:
            return 1.6

        raw_max = max(candidates)
        return min(standards, key=lambda x: abs(x - raw_max))

    def _process_single_image(self, image_path, save_dir):
        raw_img_pil = Image.open(image_path).convert('RGB')
        orig_w, orig_h = raw_img_pil.size
        img_np = cv2.cvtColor(np.array(raw_img_pil), cv2.COLOR_RGB2BGR)

        # 1. 模型推理获取掩膜
        input_tensor = self.transform(raw_img_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor)
            output = torch.nn.functional.interpolate(
                output,
                size=(orig_h, orig_w),
                mode='bilinear',
                align_corners=True
            )
            pred_mask = output.argmax(1).squeeze(0).cpu().numpy()

        # 类别约定：
        # 0 = 背景
        # 1 = 指针
        # 2 = 刻度
        p_mask = np.where(pred_mask == 1, 255, 0).astype(np.uint8)
        s_mask = np.where(pred_mask == 2, 255, 0).astype(np.uint8)

        # 可选：保存中间掩膜，便于排查
        cv2.imwrite(os.path.join(save_dir, 'pointer_mask.png'), p_mask)
        cv2.imwrite(os.path.join(save_dir, 'scale_mask.png'), s_mask)

        # 2. 拟合圆心并提取所有刻度角度
        O, R = self._fit_dial_circle(s_mask)
        if O is None:
            print(f"跳过 {os.path.basename(image_path)}: 未能拟合出圆心。")
            return

        s_cnts, _ = cv2.findContours(s_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        angles = []

        for c in s_cnts:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                dist = math.sqrt((cx - O[0]) ** 2 + (cy - O[1]) ** 2)

                # 过滤明显不在表盘刻度环附近的轮廓
                if 0.5 * R < dist < 1.5 * R:
                    ang = (math.atan2(cy - O[1], cx - O[0]) + 2 * math.pi) % (2 * math.pi)
                    angles.append(ang)

        if len(angles) < 5:
            print(f"跳过 {os.path.basename(image_path)}: 识别刻度太少。")
            return

        # 3. 通过最大角间隙估计起点和终点
        angles.sort()
        max_gap = 0
        start_idx = 0

        for i in range(len(angles)):
            gap = (angles[(i + 1) % len(angles)] - angles[i] + 2 * math.pi) % (2 * math.pi)
            if gap > max_gap:
                max_gap = gap
                start_idx = (i + 1) % len(angles)

        start_angle = angles[start_idx]
        end_angle = angles[start_idx - 1]
        total_angle_range = (end_angle - start_angle + 2 * math.pi) % (2 * math.pi)

        if total_angle_range < math.radians(10):
            print(f"跳过 {os.path.basename(image_path)}: 刻度有效角域异常。")
            return

        # 4. OCR 提取量程
        max_val = self._get_range_by_spatial_constraint(img_np, O, R, end_angle)

        # 5. 指针提取：细化 + 霍夫直线
        skeleton = self._safe_thinning(p_mask)
        cv2.imwrite(os.path.join(save_dir, 'pointer_skeleton.png'), skeleton)

        lines = cv2.HoughLinesP(
            skeleton,
            1,
            np.pi / 180,
            threshold=30,
            minLineLength=max(R // 3, 20),
            maxLineGap=20
        )

        if lines is None:
            print(f"跳过 {os.path.basename(image_path)}: 未能检测到指针。")
            return

        valid_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # 线到圆心的距离，过滤掉离圆心太远的线段
            d = abs((y2 - y1) * O[0] - (x2 - x1) * O[1] + x2 * y1 - y2 * x1) / (
                math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2) + 1e-5
            )

            if d < R / 3:
                valid_lines.append(line[0])

        if not valid_lines:
            print(f"跳过 {os.path.basename(image_path)}: 指针线段过滤后为空。")
            return

        # 取最长线段作为候选指针
        best_l = max(valid_lines, key=lambda l: (l[2] - l[0]) ** 2 + (l[3] - l[1]) ** 2)

        dist1 = (best_l[0] - O[0]) ** 2 + (best_l[1] - O[1]) ** 2
        dist2 = (best_l[2] - O[0]) ** 2 + (best_l[3] - O[1]) ** 2

        # U = 离圆心更远的端点，作为指针尖端
        U = (best_l[0], best_l[1]) if dist1 > dist2 else (best_l[2], best_l[3])

        # 6. 计算读数（修复零位重合时容易跳到满量程的问题）
        ua = (math.atan2(U[1] - O[1], U[0] - O[0]) + 2 * math.pi) % (2 * math.pi)
        reading = self._read_value_from_angle(
            ua=ua,
            start_angle=start_angle,
            end_angle=end_angle,
            total_angle_range=total_angle_range,
            max_val=max_val
        )

        # 7. 增强可视化
        f_scale = orig_w / 600.0
        vis = img_np.copy()

        cv2.rectangle(vis, (0, 0), (int(520 * f_scale), int(95 * f_scale)), (0, 0, 0), -1)
        cv2.putText(
            vis,
            f"Range:{max_val} Read:{reading:.3f}",
            (10, int(40 * f_scale)),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(f_scale * 0.8, 0.5),
            (0, 255, 255),
            2
        )
        cv2.putText(
            vis,
            f"Start:{math.degrees(start_angle):.1f} End:{math.degrees(end_angle):.1f}",
            (10, int(78 * f_scale)),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(f_scale * 0.45, 0.4),
            (180, 255, 180),
            1
        )

        # 圆心、指针
        cv2.circle(vis, O, int(max(8 * f_scale, 3)), (0, 0, 255), -1)
        cv2.line(vis, O, U, (255, 0, 0), 3)

        # 起点/终点方向线：这里改成用 R，而不是 O[0]/O[1]
        start_pt = (
            int(O[0] + R * 0.5 * math.cos(start_angle)),
            int(O[1] + R * 0.5 * math.sin(start_angle))
        )
        end_pt = (
            int(O[0] + R * 0.5 * math.cos(end_angle)),
            int(O[1] + R * 0.5 * math.sin(end_angle))
        )
        cv2.line(vis, O, start_pt, (0, 255, 0), 2)
        cv2.line(vis, O, end_pt, (0, 0, 255), 2)

        # 指针端点
        cv2.circle(vis, U, int(max(6 * f_scale, 2)), (255, 255, 0), -1)

        cv2.imwrite(os.path.join(save_dir, 'final_option2_vis.jpg'), vis)

    def predict_folder(self, input_folder, base_save_dir):
        if not os.path.exists(input_folder):
            print(f"输入目录不存在: {input_folder}")
            return

        files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
        if not files:
            print("输入目录中没有可处理图片。")
            return

        for f in tqdm(files):
            out = os.path.join(base_save_dir, os.path.splitext(f)[0])
            os.makedirs(out, exist_ok=True)
            self._process_single_image(os.path.join(input_folder, f), out)


if __name__ == "__main__":
    WEIGHT = r"D:\det-read-pointer-meter-main\test79\best_meter_model_optimized.pth"
    INPUT = r"D:\det-read-pointer-meter-main\network\test_images"
    OUTPUT = r"D:\det-read-pointer-meter-main\test79\runs\final_run1"

    predictor = Predictor(WEIGHT)
    predictor.predict_folder(INPUT, OUTPUT)