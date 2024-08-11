from __future__ import print_function
import os
import argparse
import math
import gc
import sys
import xlwt
import random
import numpy as np
import foolbox as fb
from attack import LinfBasicIterativeAttack as LinfBasicIterativeAttack_en
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.nn.functional import mse_loss
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torch.utils.data.sampler as sp
from vgg import VGG
from nets import EnsembleModel

cudnn.benchmark = True
nz = 128
target = False

SEED = 1000
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


sys.stdout = Logger('imitation_network_cifar10_L.log', sys.stdout)

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=500, help='input batch size')
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--niter', type=int, default=120, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', default=True, action='store_true', help='enables cuda')
parser.add_argument('--alpha', type=float, default=0.2, help='alpha')
parser.add_argument('--beta', type=float, default=0.0, help='set to 0.0 in the label-only scenario and to 0.1 in the '
                                                            'probability scenario')
parser.add_argument('--save_folder', type=str, default='saved_model_L', help='')
opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

transforms = transforms.Compose([transforms.ToTensor()])

testset = torchvision.datasets.CIFAR10(root='datasets/', train=False,
                                       download=True,
                                       transform=transforms
                                       )
vgg11 = VGG('VGG11').cuda()
vgg13 = VGG('VGG13').cuda()
vgg16 = VGG('VGG16').cuda()

netD = EnsembleModel(models=[vgg11, vgg13, vgg16]).cuda()

original_net = VGG('VGG').cuda()
original_net.load_state_dict(torch.load(
    'pretrained/vgg16cifar10.pth')['model'])
original_net.eval()

nc = 3
data_list = [i for i in range(6000, 8000)]  # fast validation
testloader = torch.utils.data.DataLoader(testset, batch_size=500,
                                         sampler=sp.SubsetRandomSampler(data_list), num_workers=2)

device = torch.device("cuda:0" if opt.cuda else "cpu")


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Loss_max(nn.Module):
    def __init__(self):
        super(Loss_max, self).__init__()
        return

    def forward(self, pred, truth, proba):
        criterion_1 = nn.MSELoss()
        criterion = nn.CrossEntropyLoss()
        pred_prob = F.softmax(pred, dim=1)
        loss = criterion(pred, truth) + criterion_1(pred_prob, proba) * opt.beta
        final_loss = torch.exp(loss * -1)
        return final_loss


def get_att_results(model, pro):
    correct = 0.0
    total = 0.0
    total_L2_distance = 0.0
    att_num = 0.
    acc_num = 0.
    for data in testloader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        ones = torch.ones_like(predicted)
        zeros = torch.zeros_like(predicted)
        acc_sign = torch.where(predicted == labels, ones, zeros)
        acc_num += acc_sign.sum().float()
        adversary_ghost = LinfBasicIterativeAttack_en(netD, pro, nn.CrossEntropyLoss(reduction="sum"), eps=8 / 255,
                                                      eps_iter=2 / 255,
                                                      clip_min=0.0,
                                                      clip_max=1.0, targeted=False)
        adv_inputs_ori = adversary_ghost.perturb(inputs, labels)
        L2_distance = (adv_inputs_ori - inputs).squeeze()
        L2_distance = (torch.linalg.norm(L2_distance.flatten(start_dim=1), dim=1)).data
        L2_distance = L2_distance * acc_sign
        total_L2_distance += L2_distance.sum()
        with torch.no_grad():
            outputs = model(adv_inputs_ori)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum()
            att_sign = torch.where(predicted == labels, zeros, ones)
            att_sign = att_sign + acc_sign
            att_sign = torch.where(att_sign == 2, ones, zeros)
            att_num += att_sign.sum().float()

    att_result = (att_num / acc_num * 100.0)
    print(f'Attack success rate: %.2f %%' %
          (att_num / acc_num * 100.0))
    print('l2 distance:  %.4f ' % (total_L2_distance / acc_num))
    return att_result


class pre_conv(nn.Module):
    def __init__(self, num_class):
        super(pre_conv, self).__init__()
        self.nf = 64
        self.pre_conv = nn.Sequential(
            nn.Conv2d(nz, self.nf * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.nf * 2),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.nf * 2, self.nf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.nf * 2),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.nf * 2, self.nf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.nf * 2),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.nf * 2, self.nf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.nf * 2),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.nf * 2, self.nf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.nf * 2),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.nf * 2, self.nf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.nf * 2),
            # nn.LeakyReLU(0.2, inplace=True)
            nn.ReLU(True),
        )

    def forward(self, input):
        output = self.pre_conv(input)
        return output


pre_conv_block = []
for i in range(10):
    pre_conv_block.append(pre_conv(10).cuda())


class Generator_cifar10(nn.Module):
    def __init__(self, num_class):
        super(Generator_cifar10, self).__init__()
        self.nf = 64
        self.num_class = num_class

        self.main = nn.Sequential(  # 128, 32, 32
            nn.Conv2d(128, 256, 3, 1, 1, bias=False),  # 64 32 32
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.Conv2d(256, 512, 3, 1, 1, bias=False),  # 32 32 32
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.Conv2d(512, 256, 3, 1, 1, bias=False),  # 16 32 32
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.Conv2d(256, 128, 3, 1, 1, bias=False),  # 8 32 32
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.Conv2d(128, 64, 3, 1, 1, bias=False),  # 4 32 32
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, 3, 3, 1, 1, bias=False),  # 2 32 32
            nn.BatchNorm2d(3),
            nn.ReLU(True),

            nn.Conv2d(3, 3, 3, 1, 1, bias=False),  # 1 28 28--->3 32 32
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output


def chunks(arr, m):
    n = int(math.ceil(arr.size(0) / float(m)))
    return [arr[i:i + n] for i in range(0, arr.size(0), n)]


netG = Generator_cifar10(10).cuda()

criterion = nn.CrossEntropyLoss()
criterion_max = Loss_max()

# setup optimizer
optimizer0 = optim.Adam(netD.models[0].parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizer1 = optim.Adam(netD.models[1].parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizer2 = optim.Adam(netD.models[2].parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizers = [optimizer0, optimizer1, optimizer2]
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr * 2.0, betas=(opt.beta1, 0.999))
optimizer_block = []
for i in range(10):
    optimizer_block.append(optim.Adam(pre_conv_block[i].parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)))

with torch.no_grad():
    correct_netD = 0.0
    total = 0.0
    for i in range(3):
        netD.models[i].eval()
    for data in testloader:
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs = original_net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct_netD += (predicted == labels).sum()
    print('original net accuracy: %.2f %%' %
          (100. * correct_netD.float() / total))

batch_num = 500
best_acc_num = 0.0
best_att = 0.0
cnt = 0

for epoch in range(opt.niter):
    flag = 0
    print('-------------------train D-----------------')
    for i in range(3):
        netD.models[i].train()

    for ii in range(batch_num):
        for i in range(3):
            netD.models[i].zero_grad()

        ############################
        # (1) Update D network:
        ###########################
        noise = torch.rand(opt.batchSize, nz, 1, 1, device=device).cuda()
        noise_chunk = chunks(noise, 10)
        # print(len(noise_chunk))
        for i in range(len(noise_chunk)):
            tmp_data = pre_conv_block[i](noise_chunk[i])
            gene_data = netG(tmp_data)
            label = torch.full((noise_chunk[i].size(0),), i).cuda()
            if i == 0:
                data = gene_data
                set_label = label
            else:
                data = torch.cat((data, gene_data), 0)
                set_label = torch.cat((set_label, label), 0)

        index = torch.randperm(set_label.size()[0])
        data = data[index]
        set_label = set_label[index]
        # obtain the output label of T
        with torch.no_grad():
            outputs = original_net(data)
            cnt += 1
            _, label = torch.max(outputs.data, 1)
            outputs = F.softmax(outputs, dim=1)

        # optimize three nets
        for n in range(3):
            output = netD.models[n](data.detach())
            prob = F.softmax(output, dim=1)
            # print(torch.sum(outputs) / 500.)
            errD_prob = mse_loss(prob, outputs, reduction='mean')
            errD_fake = criterion(output, label) + errD_prob * opt.beta
            errD_fake.backward()
            errD = errD_fake
            optimizers[n].step()

            if (ii % 20) == 0:
                if n == 0:
                    print(f'[%d/%d][%d/%d]net{n} errD: %.4f D_prob: %.4f'
                          % (epoch, opt.niter, ii, batch_num,
                             errD.item(), errD_prob.item()))
                else:
                    print(f'               net{n} errD: %.4f D_prob: %.4f'
                          % (errD.item(), errD_prob.item()))
        del output, errD_fake

        ############################
        # (2) Update G network:
        ###########################
        netG.zero_grad()
        for i in range(10):
            pre_conv_block[i].zero_grad()
        output0 = netD.models[0](data)
        output1 = netD.models[1](data)
        output2 = netD.models[2](data)
        output = F.softmax(output0 + output1 + output2, dim=0)
        loss_imitate = criterion_max(pred=output0, truth=label, proba=outputs) + criterion_max(pred=output1,
                                                                                               truth=label,
                                                                                               proba=outputs) + criterion_max(
            pred=output2, truth=label, proba=outputs)
        loss_diversity = criterion(output, label.squeeze().long())
        errG = opt.alpha * loss_diversity + loss_imitate
        if loss_diversity.item() <= 0.6:
            opt.alpha = loss_diversity.item() / 3
        errG.backward()
        if (ii % 20) == 0:
            print('               loss_imitate: %.4f loss_diversity: %.4f'
                  % (loss_imitate.item(), loss_diversity.item()))
            print('current opt.alpha: ', opt.alpha)

    ################################################
    # evaluate the accuracy of trained D:
    ################################################
    with torch.no_grad():
        correct_netD = 0.0
        correct = [0.0, 0.0, 0.0]
        total = 0.0
        for i in range(3):
            netD.models[i].eval()
        for data in testloader:
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            total += labels.size(0)

            outputs = netD(inputs)
            _, predicted = torch.max(outputs.data, 1)
            correct_netD += (predicted == labels).sum()
            for i in range(3):
                output = netD.models[i](inputs)
                _, predicted = torch.max(output.data, 1)
                correct[i] += (predicted == labels).sum()
        print('Accuracy of netD(evenly): %.2f %%' %
              (100. * correct_netD.float() / total))
        acc = []
        for i in range(3):
            acc.append(correct[i].float() / total)
        print('Accuracies of nets: %.2f %%      %.2f %%      %.2f %% ' % (100. * acc[0], 100. * acc[1], 100. * acc[2]))
        if best_acc_num < correct_netD:
            flag = 1
            torch.save(netD.models[0].state_dict(),
                       opt.save_folder + '/netD_vgg11_epoch_%d.pth' % (epoch))
            torch.save(netD.models[1].state_dict(),
                       opt.save_folder + '/netD_vgg13_epoch_%d.pth' % (epoch))
            torch.save(netD.models[2].state_dict(),
                       opt.save_folder + '/netD_vgg16_epoch_%d.pth' % (epoch))
            torch.save(netG.state_dict(),
                       opt.save_folder + '/netG_epoch_%d.pth' % (epoch))
            best_acc_num = correct_netD
            print('model saved')
        else:
            print('Best ACC: %.2f %%' % (100. * best_acc_num.float() / total))
        pro = acc[:]
        sum = 0
        for i in range(3):
            sum += acc[i]
        pro = [x / sum for x in pro]
        pro = torch.tensor(pro)

    ####################################################
    # evaluate the asr based on trained D
    ####################################################
    for i in range(3):
        netD.models[i].eval()
    att_result = get_att_results(original_net, pro)
    print('Attack success rate(weighted): %.2f %%' % (att_result))
    # att_result_evenly = get_att_results(original_net, [0.3333, 0.3333, 0.3333])
    # print('Attack success rate(evenly): %.2f %%' % (att_result_evenly))
    if best_att < att_result:
        best_att = att_result
        if flag == 0:
            torch.save(netD.models[0].state_dict(),
                       opt.save_folder + '/netD_vgg11_epoch_%d.pth' % (epoch))
            torch.save(netD.models[1].state_dict(),
                       opt.save_folder + '/netD_vgg13_epoch_%d.pth' % (epoch))
            torch.save(netD.models[2].state_dict(),
                       opt.save_folder + '/netD_vgg16_epoch_%d.pth' % (epoch))
            torch.save(netG.state_dict(),
                       opt.save_folder + '/netG_epoch_%d.pth' % (epoch))
            print('model saved')
    else:
        print('Best ASR: %.2f %%' % (best_att))
