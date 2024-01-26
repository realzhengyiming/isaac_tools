# 训练以撒模型

import json
import os
import shutil

# 导入依赖
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# 加载训练数据
# 读取道具的 爬虫的数据映射


with open("id_mapping_dict.json") as file:
    json_content = json.loads(file.read())

classes = {v['new_id']: v["zh"] + ":" + v["desc"] for i, v in json_content.items() if v.get("new_id") is not None}

# 检查总数
dir_path = "cus_data/"

images = os.listdir(dir_path)
print(len(images))

# 分出小批量进行测试验证
meta_output_dir = "mini_20meta_dataset"
os.makedirs(meta_output_dir, exist_ok=True)
for i in images[:20]:
    shutil.copy(os.path.join(dir_path, i), os.path.join(meta_output_dir, i))

# 自定义自己数据集
from torchvision.io import read_image
from torch.utils.data import Dataset


class IssacCustomDatasets(Dataset):
    def __init__(self, annotations_file,
                 img_dir, transform=None,
                 target_transform=None):
        self.img_labels = annotations_file  # 直接生成他们的顺序标签, 后面dataloader再shuffle
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, str(self.img_labels[idx]) + ".png")
        image = read_image(img_path, mode=torchvision.io.image.ImageReadMode.RGB)

        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


# 加载数据
transform = transforms.Compose(
    [
        #         transforms.ToTensor(),
        transforms.Resize((224, 224)),  # 将图片尺寸调整为224x224 大的话, 训练时间会更长, 那之前是怎么训练的, 麻了.
        # 增加噪声, 防止过拟合, 因为我还是需要一些现实的照片才可以更准确一些.
        transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1,
            hue=0.1),  # 抖动图像的亮度、对比度、饱和度和色相
        transforms.Lambda(lambda x: x.float()),
        transforms.Normalize(
            [43.11019, 42.666084, 42.702415],
            [100.52347, 99.96471, 100.45631]
        )  # 对图片数据做正则化
    ])

batch_size = 4
labels = list(range(20))
train_dataset = IssacCustomDatasets(labels, img_dir="mini_20meta_dataset/",
                                    transform=transform)
# labels = list(range(len(os.listdir("cus_data"))))
# train_dataset = IssacCustomDatasets(labels, img_dir="cus_data/",
#                                     transform=transform)

# dataloader
train_dataloader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True)
# test_loader = train_dataloader

# 使用pytorch模型进行训练
from torchvision.models import vgg16
import torch.optim as optim

# 小批量测试
net = vgg16(num_classes=20)  # 这个倒是完整的
# net = MobileNetV3(num_classes=20)  # 这个倒是完整的
net.to(device)  # 重建一个模型, 初始化一个  或者我直接用 64 不用于训练模型

criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.00001, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr=0.0001)
# 换cell 才可以好一点, 不然会出问题


