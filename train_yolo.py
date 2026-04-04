# train_yolo.py
from ultralytics import YOLO

def main():
    # 1. 加载模型
    model = YOLO("yolo26s.pt")

    # 2. 开始训练
    model.train(
        data="meter.yaml",
        epochs=300,
        imgsz=640,
        optimizer="MuSGD",
        batch=16,
        workers=0,  # 💡 设为 0 可以避开 Windows 的多进程文件占用问题
        device=0  # 💡 既然装了 GPU 版，继续尝试用显卡
    )

if __name__ == '__main__':
    main()