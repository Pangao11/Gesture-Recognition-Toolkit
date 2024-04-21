import torch
import os
import json
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split


# Define class labels
class_labels = ['A', 'B', 'C', 'Five', 'Point', 'V']


def load_dataset(dataset_dir):
    image_list = []
    label_list = []
    class_labels = []
    gesture_folders = os.listdir(dataset_dir)

    for label, gesture_folder in enumerate(gesture_folders):
        gesture_folder_path = os.path.join(dataset_dir, gesture_folder)
        if os.path.isdir(gesture_folder_path):
            class_labels.append(gesture_folder)
            image_files = os.listdir(gesture_folder_path)
            for image_file in image_files:
                image_file_path = os.path.join(gesture_folder_path, image_file)
                if image_file_path.endswith('.png'):
                    image_list.append(image_file_path)
                    label_list.append(label - 1)  # Changed this line

    return image_list, label_list, class_labels


class HandPostureDataset(Dataset):
    def __init__(self, image_list, label_list, transform=None):
        self.image_list = image_list
        self.label_list = label_list
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(self.image_list[idx]).convert("RGB")
        label = self.label_list[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


dataset_dir = './Hand_Posture_Hard_Stu'

image_list, label_list, class_labels = load_dataset(dataset_dir)

with open('class_labels.json', 'w') as f:  # 新增这部分，保存 class_labels 到文件
    json.dump(class_labels, f)


# 数据集拆分
train_image_list, test_image_list, train_label_list, test_label_list = train_test_split(image_list, label_list, test_size=0.2, random_state=42)

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = HandPostureDataset(train_image_list, train_label_list, transform)
test_dataset = HandPostureDataset(test_image_list, test_label_list, transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


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
    def __init__(self, growth_rate, block_config, num_classes=6):
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


# 创建DenseNet模型并修改全连接层
def densenet121(num_classes=6, pretrained=False):
    # 这里创建模型并载入预训练权重的方法发生了改变
    model = torchvision.models.densenet121(pretrained=pretrained)
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.classifier.in_features
    # Replace the last layer
    model.classifier = nn.Linear(num_features, num_classes)
    # Unfreeze the last layer
    for param in model.classifier.parameters():
        param.requires_grad = True
    return model


# net = DenseNet(growth_rate=32, block_config=[6, 12, 24, 16], num_classes=6)

net = densenet121(num_classes=6, pretrained=False)

model_weight_path = "./densenet121-a639ec97.pth"
assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)

# Load pretrain weights
pretrain_dict = torch.load(model_weight_path)
model_dict = net.state_dict()

# Get the updated dict
pretrained_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == pretrain_dict[k].size()}

# Update the current model dict
model_dict.update(pretrained_dict)

# Load the updated dict into the model
net.load_state_dict(model_dict, strict=False)


# 使用CUDA进行训练
device = torch.device("mps")
print(f"Using device: {device}")
print(set(label_list))

net = net.to(device)


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, output, target):
        c = output.size(1)
        log_preds = F.log_softmax(output, dim=1)
        loss = F.nll_loss(log_preds, target, reduction=self.reduction)
        smooth_loss = -log_preds.mean(dim=1)
        if self.reduction == 'mean':
            smooth_loss = smooth_loss.mean()
        elif self.reduction == 'sum':
            smooth_loss = smooth_loss.sum()
        return loss * (1 - self.eps) + smooth_loss * self.eps


# 设置损失函数和优化器
criterion = LabelSmoothingCrossEntropy()

optimizer = optim.Adam(net.parameters(), lr=0.001)


def evaluate(net, loader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


# 训练
num_epochs = 60

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data

        # Move data to GPU
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_loss += loss.item()
        rate = (i + 1) / len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss:{:^3.0f}%[{}->{}]{:.4f}".format(int(rate * 100), a, b, loss), end="")
    print()
    train_acc = evaluate(net, train_loader, device)
    test_acc = evaluate(net, test_loader, device)
    print('Epoch %d loss: %.3f, train acc: %.3f, test acc: %.3f' % (epoch + 1, running_loss / (i + 1), train_acc, test_acc))

print('Finished Training')

# 保存模型
torch.save(net.state_dict(), './model.pth')



