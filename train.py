import os
import csv
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# 导入你的项目模块
from model import DeepLab
from dataloader import MeterDataset
from utils import SegmentationLoss, Evaluator

# ==============================================================================
# 1. 训练配置 (针对优化版模型提速)
# ==============================================================================
DATA_DIR = r"D:/det-read-pointer-meter-main/dataset"
INPUT_SIZE = (512, 512)

BATCH_SIZE = 8

ACCUMULATION_STEPS = 1

EPOCHS = 120
LEARNING_RATE = 1e-4
NUM_CLASSES = 3
CSV_PATH = "training_log_final_optimized.csv"


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #  确认模式：优化版 SPCE
    print(f" 设备已就绪: {device} | 模式: MSH-PPNet Optimized (空洞条形卷积 + 提速版 SPCE)")

    # 1. 数据加载
    train_dataset = MeterDataset(DATA_DIR, subset='train', input_size=INPUT_SIZE, augment=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)

    val_dataset = MeterDataset(DATA_DIR, subset='val', input_size=INPUT_SIZE, augment=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    # 2. 模型初始化
    # 使用默认的 output_stride=16 配合空洞卷积获得高性能
    model = DeepLab(num_classes=NUM_CLASSES, output_stride=16)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"🛠️ 模型总参数量: {total_params:.2f} M")

    # 3. 优化器与损失函数
    criterion = SegmentationLoss(num_classes=NUM_CLASSES)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    evaluator = Evaluator(NUM_CLASSES)
    best_miou = 0.0

    # 4. 初始化 CSV 表头
    with open(CSV_PATH, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train_Loss', 'Val_Loss', 'mIoU', 'Pixel_Acc',
                         'Class_Acc', 'FWIoU', 'FPS', 'Params_M', 'LR'])

    # ==========================================================================
    # 5. 训练主循环
    # ==========================================================================
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for i, (images, masks) in enumerate(pbar):
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            # 如果 ACCUMULATION_STEPS > 1 则启用
            loss = loss / ACCUMULATION_STEPS
            loss.backward()

            if (i + 1) % ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item() * ACCUMULATION_STEPS
            pbar.set_postfix({'loss': f"{loss.item() * ACCUMULATION_STEPS:.4f}"})

        avg_train_loss = train_loss / len(train_loader)

        # ==========================
        # 6. 验证环节
        # ==========================
        model.eval()
        evaluator.reset()
        val_loss_sum = 0.0
        inference_times = []

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)

                torch.cuda.synchronize()
                start = time.time()
                outputs = model(images)
                torch.cuda.synchronize()
                inference_times.append(time.time() - start)

                val_loss_sum += criterion(outputs, masks).item()
                preds = torch.argmax(outputs, dim=1)
                evaluator.add_batch(masks.cpu().numpy(), preds.cpu().numpy())

        # 指标计算
        mIoU = evaluator.Mean_Intersection_over_Union()
        acc = evaluator.Pixel_Accuracy()
        acc_class = evaluator.Pixel_Accuracy_Class()
        fwiou = evaluator.Frequency_Weighted_Intersection_over_Union()

        avg_val_loss = val_loss_sum / len(val_loader)
        fps = 1.0 / np.mean(inference_times[5:]) if len(inference_times) > 5 else 0
        current_lr = optimizer.param_groups[0]['lr']

        print(f"\n Epoch {epoch + 1} 总结:")
        print(f"   mIoU: {mIoU:.4f} | Class_Acc: {acc_class:.4f} | FPS: {fps:.2f}")

        # 7. 保存最佳模型
        if mIoU > best_miou:
            best_miou = mIoU
            torch.save(model.state_dict(), "best_meter_model_optimized.pth")
            print(f" 最佳记录更新: {best_miou:.4f}")

        # 8. 写入日志
        with open(CSV_PATH, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_train_loss, avg_val_loss, mIoU, acc,
                             acc_class, fwiou, fps, total_params, current_lr])

        scheduler.step()

    print(f"训练已完成！所有指标已保存在 {CSV_PATH}")


if __name__ == '__main__':
    train()