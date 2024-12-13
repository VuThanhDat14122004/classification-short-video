# Giới thiệu

Dự án nghiên cứu bài toán phân loại video sử dụng mô hình học sâu kết hợp giữa CNN và RNN.

# Mô tả

Nhóm đã thử cài đặt kết hợp các mô hình pre-train với mô hình RNN truyền thống và mô hình LSTM để thực hiện bài toán phân loại video.

Trong đó:

- Sử dụng bộ dữ liệu [abdallahwagih/ucf101-videos](https://www.kaggle.com/datasets/abdallahwagih/ucf101-videos)
- Sử dụng các mô hình Pre-train: [ResNet50](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html), [ResNet101](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet101.html) và [ResNet152](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet152.html)

# Kết quả

Sau quá trình huấn luyện với các bộ tham số khác nhau, nhóm đã đạt được 1 số kết quả sơ bộ như sau:

### Kết hợp với mô hình RNN truyền thống

| Mô hình/Tham số | Learning rate cố định | Learning rate thay đổi |
| --------------- | --------------------- | ---------------------- |
| ResNet50        | 0.6339                | 0.7188                 |
| ResNet101       | 0.5804                | 0.7723                 |
| ResNet152       | 0.6295                | 0.7500                 |

### Kết hợp với mô hình LSTM

| Tham số/Mô hình                                | ResNet50 | ResNet101 | ResNet152 |
| ---------------------------------------------- | -------- | --------- | --------- |
| Batch size = 8, lr=0.0001                      | 0.6964   | 0.808     | 0.6651    |
| Batch size = 8, lr=0.00005                     | 0.8616   | -         | -         |
| Batch size = 16, lr=0.0001                     | 0.7946   | 0.683     | 0.7366    |
| Batch size = 16, lr=0.0001 + ReduceLROnPlateau | 0.8705   | -         | -         |
| Batch size = 16, lr=0.00005                    | 0.7946   | -         | -         |

Kết quả chi tiết xem tại: [report.pdf](./report.pdf)

# Thành viên

<table>
    <th>
        <td>Full Name</td>
        <td>Email</td>
    </th>
    <tr>
        <td>1</td>
        <td>Vũ Thành Đạt</td>
        <td><a href="mailto:22022620@vnu.edu.vn">22022620@vnu.edu.vn</a></td>
    </tr>
    <tr>
        <td>2</td>
        <td>Nguyễn Trần Hải Ninh</td>
        <td><a href="mailto:22022526@vnu.edu.vn">22022526@vnu.edu.vn</a></td>
    </tr>
    <tr>
        <td>3</td>
        <td>Nguyễn Quang Thao</td>
        <td><a href="mailto:22022619@vnu.edu.vn">22022619@vnu.edu.vn</a></td>
    </tr>
</table>
