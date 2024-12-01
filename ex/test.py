import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

# Đọc và chuyển đổi hình ảnh thành tensor
def load_image(image_path):
    # Sử dụng torchvision transforms để chuyển đổi hình ảnh thành tensor
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Thay đổi kích thước ảnh về 64x64 (có thể tùy chỉnh)
        transforms.ToTensor()  # Chuyển hình ảnh thành tensor (C x H x W) với các giá trị từ 0 đến 1
    ])
    
    image = Image.open(image_path)  # Mở hình ảnh
    return transform(image).unsqueeze(0)  # Chuyển thành tensor và thêm batch dimension (B x C x H x W)

# Hiển thị tensor dưới dạng hình ảnh
def show_image(tensor, title="Image"):
    np_image = tensor.squeeze().permute(1, 2, 0).numpy()  # Chuyển tensor sang numpy để hiển thị
    plt.imshow(np_image)
    plt.title(title)
    plt.show()

# Đường dẫn tới hình ảnh của bạn
image_path = "your_image_path_here.jpg"  # Thay thế bằng đường dẫn hình ảnh của bạn

# Bước 1: Load hình ảnh vào tensor
image_tensor = load_image(image_path)

# Hiển thị ảnh gốc (RGB)
show_image(image_tensor, title="Original RGB Image")

# Bước 2: Hoán đổi kênh màu từ RGB sang GBR
gbr_image_tensor = image_tensor[:, [1, 2, 0], :, :]  # Hoán đổi các kênh từ RGB sang GBR

# Hiển thị ảnh sau khi hoán đổi kênh
show_image(gbr_image_tensor, title="GBR Image")