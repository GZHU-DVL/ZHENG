from __future__ import print_function
import os
import argparse
import gc
import sys
import xlwt
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn.functional import mse_loss
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torch.utils.data.sampler as sp
from nets import Net_s, Net_m, Net_l, Generator, pre_conv, EnsembleModel
from utils import chunks, weights_init, Loss_exp
from attack import LinfBasicIterativeAttack

cudnn.benchmark = True
workbook = xlwt.Workbook(encoding='utf-8')
worksheet = workbook.add_sheet('imitation_network_sig')
nz = 128  # noise
nc = 1  # channel

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=500, help='input batch size')
parser.add_argument('--dataset', type=str, default='mnist', help='')
parser.add_argument('--niter', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam.')
parser.add_argument('--cuda', default=True, action='store_true', help='enables cuda')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--alpha', type=float, default=0.2, help='')
parser.add_argument('--beta', type=float, default=0.0, help='set to 0.0 in the label-only scenario and to 0.1 in the '
                                                            'probability scenario')
parser.add_argument('--save_folder', type=str, default='saved_model_L', help='')

opt = parser.parse_args()
print(opt)

net_s = Net_s().cuda()
net_m = Net_m().cuda()
net_l = Net_l().cuda()

ensemble_model = EnsembleModel(models=[net_s, net_m, net_l])
netD = ensemble_model.cuda()
device = torch.device("cuda:0" if opt.cuda else "cpu")


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


sys.stdout = Logger('imitation_network_mnist_L.log', sys.stdout)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

testset = torchvision.datasets.MNIST(root='datasets/', train=False,
                                     download=True,
                                     transform=transforms.Compose([
                                         # transforms.Pad(2, padding_mode="symmetric"),
                                         transforms.ToTensor(),
                                         # transforms.RandomCrop(32, 4),
                                         # normalize,
                                     ]))

# target model
original_net = Net_m().cuda()
state_dict = torch.load(
    'pretrained/net_m.pth')
original_net.load_state_dict(state_dict)
original_net = nn.DataParallel(original_net)
original_net.eval()

data_list = [i for i in range(6000, 8000)]  # fast validation
testloader = torch.utils.data.DataLoader(testset, batch_size=500,
                                         sampler=sp.SubsetRandomSampler(data_list),
                                         num_workers=2)
pre_conv_block = []
for i in range(10):
    pre_conv_block.append(nn.DataParallel(pre_conv(10).cuda()))

netG = Generator(10).cuda()
netG.apply(weights_init)
netG = nn.DataParallel(netG)

criterion = nn.CrossEntropyLoss()
criterion_exp = Loss_exp()

# setup optimizer
optimizerS = optim.Adam(netD.models[0].parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerM = optim.Adam(netD.models[1].parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerL = optim.Adam(netD.models[2].parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizers = [optimizerS, optimizerM, optimizerL]

optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizer_block = []
for i in range(10):
    optimizer_block.append(optim.Adam(pre_conv_block[i].parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)))

################################################
# estimate the accuracy of initial substitute model:
################################################
with torch.no_grad():
    correct = [0.0, 0.0, 0.0]
    total = 0.0

    for i in range(3):
        netD.models[i].eval()

    for data in testloader:
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        total += labels.size(0)
        for ii in range(3):
            outputs = netD.models[ii](inputs)
            _, predicted = torch.max(outputs.data, 1)
            correct[ii] += (predicted == labels).sum()
    print(
        'Accuracy of the initial net_s: %.2f %%' %
        (100. * correct[0] / total))
    print(
        'Accuracy of the initial net_m: %.2f %%' %
        (100. * correct[1] / total))
    print(
        'Accuracy of the initial net_l: %.2f %%' %
        (100. * correct[2] / total))

pro_sml = []
for i in range(3):
    pro_sml.append(correct[i] / total)
sum = 0
for i in range(3):
    sum += pro_sml[i]
pro_sml = [x / sum for x in pro_sml]
pro_sml = torch.tensor(pro_sml)
adversary_ghost = LinfBasicIterativeAttack(netD, pro_sml, nn.CrossEntropyLoss(reduction="sum"), eps=0.3, nb_iter=100,
                                           eps_iter=0.01, clip_min=0.0,
                                           clip_max=1.0, targeted=False)
################################################
# estimate the attack success rate of initial D:
################################################
correct_en = 0.0
total = 0.0
netD.eval()
for data in testloader:
    inputs, labels = data
    inputs = inputs.cuda()
    labels = labels.cuda()
    total += labels.size(0)
    adv_inputs_ghost = adversary_ghost.perturb(inputs, labels)
    with torch.no_grad():
        outputs1 = original_net(adv_inputs_ghost)
        _, predicted1 = torch.max(outputs1.data, 1)
    correct_en += (predicted1 == labels).sum()

print('initial attack success rate:  %.2f %%' %
      (100 - 100. * correct_en.float() / total))
del inputs, labels, adv_inputs_ghost
torch.cuda.empty_cache()
gc.collect()

batch_num = 200
best_accuracy = 0.0
best_att = 0.0

for epoch in range(opt.niter):
    flag = 0  # best flag
    for i in range(3):
        netD.models[i].train()
    for ii in range(batch_num):
        noise = torch.randn(opt.batchSize, nz, 1, 1).cuda()
        noise_chunk = chunks(noise, 10)

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
            _, label = torch.max(outputs.data, 1)
            outputs = F.softmax(outputs, dim=1)
        for n in range(3):
            netD.models[n].zero_grad()
            ############################
            # (1) Update D networks:
            ###########################
            output = netD.models[n](data.detach())
            prob = F.softmax(output, dim=1)
            errD_prob = mse_loss(prob, outputs, reduction='mean')
            errD_ce = criterion(output, label)
            errD_fake = errD_ce + errD_prob * opt.beta
            D_G_z1 = errD_fake.mean().item()
            errD_fake.backward()

            errD = errD_fake
            optimizers[n].step()
            if ii % 20 == 0:
                if n == 0:
                    print(
                        '[%d/%d][%d/%d] net%d errD: %.4f errD_ce: %.4f errD_prob: %.4f'
                        % (epoch, opt.niter, ii, batch_num, n,
                           errD.item(), errD_ce.item(), errD_prob.item()))
                else:
                    print('               ' + 'net%d errD: %.4f errD_ce: %.4f errD_prob: %.4f'
                          % (n, errD.item(), errD_ce.item(), errD_prob.item()))
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
        output = netD(data)
        loss_imitate = criterion_exp(pred=output0, truth=label, proba=outputs, beta=opt.beta) + criterion_exp(
            pred=output1,
            truth=label,
            proba=outputs, beta=opt.beta) + criterion_exp(
            pred=output2, truth=label, proba=outputs, beta=opt.beta)
        loss_diversity = criterion(output, set_label.squeeze().long())
        errG = opt.alpha * loss_diversity + loss_imitate
        if loss_diversity.item() <= 0.1:
            opt.alpha = loss_diversity.item()
        errG.backward()
        D_G_z2 = errG.mean().item()
        optimizerG.step()
        for i in range(10):
            optimizer_block[i].step()
        if (ii % 20) == 0:
            print('errG: %.4f loss_imitate: %.4f loss_diversity: %.4f' % (
                errG.item(), loss_imitate, loss_diversity))

    ################################################
    # evaluate the accuracy of trained D:
    ################################################
    with torch.no_grad():
        correct = [0.0, 0.0, 0.0]
        correct_en = 0.0
        total = 0.0
        for i in range(3):
            netD.models[i].eval()
        for data in testloader:
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            total += labels.size(0)
            for i in range(3):
                outputs = netD.models[i](inputs)
                _, predicted = torch.max(outputs.data, 1)
                correct[i] += (predicted == labels).sum()
            outputs = netD(inputs)
            _, predicted = torch.max(outputs.data, 1)
            correct_en += (predicted == labels).sum()
        acc = []
        acc_en = 100. * correct_en.float() / total
        for i in range(3):
            acc.append(correct[i].float() / total)
        print('Accuracy of the network on trained nets: %.2f %%     %.2f %%     %.2f %%' % (
            100. * acc[0], 100. * acc[1], 100. * acc[2]))
        for i in range(3):
            worksheet.write(epoch, i, acc[i].item())
        print('Accuracy of the ensemble network(evenly): %.2f%%' % acc_en)
        worksheet.write(epoch, 3, acc_en.item())
        pro_sml = []
        for i in range(3):
            pro_sml.append(acc[i])
        sum = 0
        for i in range(3):
            sum += pro_sml[i]
        pro_sml = [x / sum for x in pro_sml]
        pro_sml = torch.tensor(pro_sml)

        if acc_en > best_accuracy:
            flag = 1
            best_accuracy = acc_en
            torch.save(netD.models[0].state_dict(),
                       opt.save_folder + '/net_s_epoch_%d.pth' % (epoch))
            torch.save(netD.models[1].state_dict(),
                       opt.save_folder + '/net_m_epoch_%d.pth' % (epoch))
            torch.save(netD.models[2].state_dict(),
                       opt.save_folder + '/net_l_epoch_%d.pth' % (epoch))
            torch.save(netG.state_dict(),
                       opt.save_folder + '/netG_epoch_%d.pth' % (epoch))
            print('In terms of accuracy, this is the best model')

    ################################################
    # estimate the attack success rate of trained D:
    ################################################
    adversary_ghost = LinfBasicIterativeAttack(netD, pro_sml, nn.CrossEntropyLoss(reduction="sum"), eps=0.3,
                                               nb_iter=100, eps_iter=0.01, clip_min=0.0,
                                               clip_max=1.0, targeted=False)
    correct_en = 0.0
    total = 0.0
    netD.eval()
    for data in testloader:
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        adv_inputs_ghost = adversary_ghost.perturb(inputs, labels)
        with torch.no_grad():
            outputs = original_net(adv_inputs_ghost)
            _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct_en += (predicted == labels).sum()

        att_en = 100 - 100. * correct_en.float() / total

    print(' attack success rate: %.2f %%' %
          att_en)

    if att_en > best_att:
        best_att = att_en
        print('In terms of attack success rate, this is the best model')
        if flag == 0:
            torch.save(netD.models[0].state_dict(),
                       opt.save_folder + '/net_s_epoch_%d.pth' % (epoch))
            torch.save(netD.models[1].state_dict(),
                       opt.save_folder + '/net_m_epoch_%d.pth' % (epoch))
            torch.save(netD.models[2].state_dict(),
                       opt.save_folder + '/net_l_epoch_%d.pth' % (epoch))
            torch.save(netG.state_dict(),
                       opt.save_folder + '/netG_epoch_%d.pth' % (epoch))

    worksheet.write(epoch, 4, (1 - correct_en.float() / total).item())

    del inputs, labels, adv_inputs_ghost
    torch.cuda.empty_cache()
    gc.collect()
workbook.save('imitation_network_saved_mnist_L.xls')
