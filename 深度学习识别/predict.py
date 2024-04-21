import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from skimage import io



class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, growth_rate * 4, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(growth_rate * 4)
        self.conv2 = nn.Conv2d(growth_rate * 4, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        out = self.bn1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = torch.cat((x, out), 1)
        return out


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList([DenseLayer(in_channels + i * growth_rate, growth_rate) for i in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        out = self.bn(x)
        out = F.relu(out)
        out = self.conv(out)
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, growth_rate, block_config, num_classes=7):
        super(DenseNet, self).__init__()
        self.conv1 = nn.Conv2d(3, growth_rate * 2, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(growth_rate * 2)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        in_channels = growth_rate * 2
        self.dense_blocks = nn.ModuleList()
        self.transition_layers = nn.ModuleList()
        for i, num_layers in enumerate(block_config):
            dense_block = DenseBlock(in_channels, growth_rate, num_layers)
            self.dense_blocks.append(dense_block)
            in_channels += num_layers * growth_rate
            if i != len(block_config) - 1:
                transition_layer = TransitionLayer(in_channels, in_channels // 2)
                self.transition_layers.append(transition_layer)
                in_channels = in_channels // 2

        self.bn2 = nn.BatchNorm2d(in_channels)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for i, dense_block in enumerate(self.dense_blocks):
            x = dense_block(x)
            if i != len(self.dense_blocks) - 1:
                x = self.transition_layers[i](x)

        x = self.bn2(x)
        x = F.relu(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


net = DenseNet(growth_rate=32, block_config=[6, 12, 24, 16], num_classes=6)


# 加载模型
model_path = './model.pth'
net.load_state_dict(torch.load(model_path))
net.eval()

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 定义类别标签
class_labels = ['A', 'B', 'C', 'Five', 'Point', 'V']

# 图片文件夹路径
image_folder = './test'

# 读取文件夹中的所有图片文件
image_filenames = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

# 对每张图片进行预测并显示
for image_filename in image_filenames:
    image_path = os.path.join(image_folder, image_filename)
    image = Image.open(image_path).convert('RGB')
    input_image = transform(image).unsqueeze(0)
    output = net(input_image)
    _, predicted_label_index = torch.max(output, 1)
    predicted_label = class_labels[predicted_label_index]
    print(f'Label index: {predicted_label_index}')
    print(f'Predicted label: {predicted_label}')
    plt.imshow(io.imread(image_path))
    plt.title(f'Predicted label: {predicted_label}')
    plt.show()

    plt.close()
