import cv2
import torch


def load_video(path, max_frame=100, size=(480, 480)):
    capture = cv2.VideoCapture(path)
    all_frames = []
    while True:
        ret, frame = capture.read()
        if len(all_frames) == max_frame:
            break
        if not ret:
            break
        frame = get_center_square(frame)
        frame = cv2.resize(frame, size)
        frame = frame[:, :, [2, 1, 0]]  # ảnh đọc ra bởi cv2 ở dạng BGR, chuyển về RGB
        frame = torch.tensor(frame)
        frame = frame.permute(2, 0, 1)  # chuyển về channel, height, width
        all_frames.append(frame)
    capture.release()
    return torch.stack(all_frames, dim=0)  # dim = (100, 3, size[0], size[1])


"""Get center square of frame"""


def get_center_square(frame):
    y, x = frame.shape[0:2]
    min_lenght = min(x, y)
    start_x = (x // 2) - min_lenght // 2
    start_y = (y // 2) - min_lenght // 2
    frame = frame[start_y : (start_y + min_lenght), start_x : (start_x + min_lenght)]
    return frame
