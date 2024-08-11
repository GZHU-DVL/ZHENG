import torch.nn as nn
import torch.nn.functional as F

nz = 128
nc = 1


class Net_s(nn.Module):
    def __init__(self):
        super(Net_s, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class Net_m(nn.Module):
    def __init__(self):
        self.number = 0
        super(Net_m, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.conv3 = nn.Conv2d(50, 50, 3, 1, 1)
        self.fc1 = nn.Linear(2 * 2 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x, sign=0):
        if sign == 0:
            self.number += 1
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 2 * 2 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def get_number(self):
        return self.number


class Net_l(nn.Module):
    def __init__(self):
        super(Net_l, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.conv3 = nn.Conv2d(50, 50, 3, 1, 1)
        self.conv4 = nn.Conv2d(50, 50, 3, 1, 1)
        self.fc1 = nn.Linear(50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class EnsembleModel(nn.Module):
    def __init__(self, models):
        super(EnsembleModel, self).__init__()
        self.models = models

    def forward(self, x):
        outputs = [model(x) for model in self.models]
        weighted_sum = (outputs[0] + outputs[1] + outputs[2]) / 3
        return weighted_sum


class pre_conv(nn.Module):
    def __init__(self, num_class):
        super(pre_conv, self).__init__()
        self.nf = 64

        self.pre_conv = nn.Sequential(
            nn.Conv2d(nz, self.nf * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.nf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(self.nf * 2, self.nf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.nf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(self.nf * 2, self.nf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.nf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(self.nf * 2, self.nf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.nf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(self.nf * 2, self.nf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.nf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(self.nf * 2, self.nf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.nf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, input):
        output = self.pre_conv(input)
        return output


class Generator(nn.Module):
    def __init__(self, num_class):
        super(Generator, self).__init__()
        self.nf = 64
        self.num_class = num_class
        self.main = nn.Sequential(
            nn.Conv2d(self.nf * 2, self.nf * 4, 3, 1, 0, bias=False),
            nn.BatchNorm2d(self.nf * 4),
            nn.LeakyReLU(0.2, inplace=True),


            nn.Conv2d(self.nf * 4, self.nf * 8, 3, 1, 0, bias=False),
            nn.BatchNorm2d(self.nf * 8),
            nn.LeakyReLU(0.2, inplace=True),


            nn.Conv2d(self.nf * 8, self.nf * 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.nf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.nf * 4, self.nf * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.nf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.nf * 2, self.nf, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.nf),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.nf, nc, 3, 1, 1, bias=False),
            nn.BatchNorm2d(nc),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nc, nc, 3, 1, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output
