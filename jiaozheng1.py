import cv2
from perspective_correction_meter import batch_correct
if __name__ == "__main__":
    batch_correct(
        input_dir=r"D:\det-read-pointer-meter-main\jiaozheng\picture",
        output_dir=r"D:\det-read-pointer-meter-main\jiaozheng\output1",
        output_size=512,
        debug=True
    )