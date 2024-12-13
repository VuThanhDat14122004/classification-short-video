import torch
from torchvision import transforms


def augmentation_picture(frame):
    transform_a_frame = transforms.Compose(
        [
            # transforms.RandomRotation(degrees=(90,90)), # xoay ảnh 90 90
            transforms.GaussianBlur(
                kernel_size=11, sigma=1
            ),  # làm mờ ảnh nhẹ với vùng ảnh là 11x11
            transforms.RandomErasing(
                scale=(0.01, 0.1), ratio=(1, 1)
            ),  # scale là phạm vi tỷ lệ (min, max) cho vùng xóa so với ảnh ban đầu,
            # ratio là tỉ lệ chiều rộng và cao
        ]
    )
    result = transform_a_frame(frame)
    return result


def augmentation_video(frames):
    new_frames = []
    for frame in frames:
        new_frame = augmentation_picture(frame)
        new_frames.append(new_frame)
    return torch.stack(new_frames, dim=0)
