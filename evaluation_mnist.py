from __future__ import print_function
import os
import argparse
import gc
import sys
import xlwt
import random
import numpy as np
# import foolbox
import torch
import torchvision
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.utils.data.sampler as sp
from nets import Net_s, Net_m, Net_l
from nets import EnsembleModel
from attack import GradientSignAttack as GradientSignAttack_en
from attack import L2PGDAttack as L2PGDAttack_en
from attack import L2BasicIterativeAttack as L2BasicIterativeAttack_en
from advertorch.attacks import GradientSignAttack, L2BasicIterativeAttack, L2PGDAttack

# from vgg import VGG

SEED = 10000

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(10000)

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading\
    workers', default=2)
parser.add_argument('--cuda', default=True, help='enables cuda')
parser.add_argument('--batch_size', default=500, help='batch size')
parser.add_argument('--adv', type=str, default='PGD', help='attack method')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--target', default=False, help='')

opt = parser.parse_args()
print(opt)

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
# print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with \
         --cuda")

testset = torchvision.datasets.MNIST(root='datasets/', train=False,
                                     download=False,
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                     ]))
data_list = [i for i in range(0, 10000)]
testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size,
                                         sampler=sp.SubsetRandomSampler(data_list), num_workers=2)


def get_adv_in(net, inputs, labels):
    # single model attack
    # BIM
    if opt.adv == 'BIM':
        adversary = L2BasicIterativeAttack(
            net,
            loss_fn=nn.CrossEntropyLoss(reduction="sum"),
            eps=4.8,
            nb_iter=120, eps_iter=0.2, clip_min=0.0, clip_max=1.0,
            targeted=opt.target)
    # PGD
    elif opt.adv == 'PGD':
        adversary = L2PGDAttack(
            net,
            loss_fn=nn.CrossEntropyLoss(reduction="sum"),
            eps=4.8,
            nb_iter=120, eps_iter=0.2, clip_min=0.0, clip_max=1.0,
            targeted=opt.target)
        # FGSM
    elif opt.adv == 'FGSM':
        adversary = GradientSignAttack(
            net,
            loss_fn=nn.CrossEntropyLoss(reduction="sum"),
            eps=0.3,
            targeted=opt.target)

    return adversary.perturb(inputs, labels)


def test_adver(net, tar_net, attack, target):
    for i in range(3):
        net.models[i].eval()
    tar_net.eval()
    # ----------------------------------
    # Obtain the accuracy of the models
    # ----------------------------------

    with torch.no_grad():
        correct = [0.0, 0.0, 0.0]
        correct_en = 0.0
        total = 0.0
        tmp = 0
        for i in range(3):
            net.models[i].eval()
        tar_net.eval()
        for data in testloader:
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            total += labels.size(0)

            outputs_tar = tar_net(inputs)
            _, label_tar = torch.max(outputs_tar.data, 1)

            for i in range(3):
                outputs = net.models[i](inputs)
                _, predicted = torch.max(outputs.data, 1)
                correct[i] += (predicted == labels).sum()

            outputs = net(inputs)
            _, predicted_en = torch.max(outputs.data, 1)
            correct_en += (predicted_en == labels).sum()

    accuracies = []
    for i in range(3):
        accuracies.append(correct[i].float() / total)
    print('Accuracy of the networks: %.2f %%      %.2f %%     %.2f %%' %
          (100. * accuracies[0], 100. * accuracies[1], 100. * accuracies[2]))
    print('Ensemble Accuracy: %.2f%%' % (100. * correct_en / total))
    # Obtain the weights based on accuracies
    pro_sml = accuracies[:]
    sum = 0
    for i in range(3):
        sum += pro_sml[i]
    pro_sml = [x / sum for x in pro_sml]
    pro_sml = torch.tensor(pro_sml)
    # ensemble attack
    if attack == 'BIM':
        adversary = L2BasicIterativeAttack_en(
            net, pro_sml,
            loss_fn=nn.CrossEntropyLoss(reduction="sum"),
            eps=4.8,
            nb_iter=120, eps_iter=0.2, clip_min=0.0, clip_max=1.0,
            targeted=opt.target)

    elif attack == 'PGD':
        adversary = L2PGDAttack_en(
            net, pro_sml,
            loss_fn=nn.CrossEntropyLoss(reduction="sum"),
            eps=4.8,
            nb_iter=120, eps_iter=0.2, clip_min=0.0, clip_max=1.0,
            targeted=opt.target)

    elif attack == 'FGSM':
        adversary = GradientSignAttack_en(
            net, pro_sml,
            loss_fn=nn.CrossEntropyLoss(reduction="sum"),
            eps=0.3,
            targeted=opt.target)

    # ----------------------------------
    # Obtain the attack success rate of the model
    # ----------------------------------
    att_num = 0
    success_num = 0
    success_nums = [0.0, 0.0, 0.0]
    total_L2_distance = 0.0
    total_L2_distances = [0.0, 0.0, 0.0]
    tar_net.eval()
    for data in testloader:
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs = tar_net(inputs)
        _, predicted = torch.max(outputs.data, 1)

        zeros = torch.zeros(predicted.size(0)).cuda()
        ones = torch.ones(predicted.size(0)).cuda()

        if target:
            # randomly choose the specific label of targeted attack
            labels = torch.randint(0, 9, (inputs.size(0),)).cuda()

            att_index = torch.where(predicted == labels, zeros, ones)
            att_num += att_index.sum()

            adv_inputs_ori = adversary.perturb(inputs, labels)
            outputs = tar_net(adv_inputs_ori)
            _, predicted = torch.max(outputs.data, 1)
            tmp_index = torch.where(predicted == labels, ones, zeros)
            tmp_index = tmp_index + att_index
            success_index = torch.where(tmp_index == 2, ones, zeros)
            success_num += success_index.sum()

            L2_distance = (adv_inputs_ori - inputs).squeeze()
            L2_distance = torch.linalg.norm(torch.flatten(L2_distance, start_dim=1), dim=1).data
            L2_distance = L2_distance * att_index
            total_L2_distance += L2_distance.sum().float()
            for i in range(3):
                adv_in = get_adv_in(net.models[i], inputs, labels)

                L2_distance = (adv_in - inputs).squeeze()
                L2_distance = torch.linalg.norm(torch.flatten(L2_distance, start_dim=1), dim=1).data
                L2_distance = L2_distance * att_index
                total_L2_distances[i] += L2_distance.sum().float()

                outputs = tar_net(adv_in)
                _, predicted = torch.max(outputs.data, 1)
                tmp_index = torch.where(predicted == labels, ones, zeros)
                tmp_index = tmp_index + att_index
                success_index = torch.where(tmp_index == 2, ones, zeros)
                success_nums[i] += success_index.sum()


        else:
            # test the images which are classified correctly
            att_index = torch.where(predicted == labels, ones, zeros)
            att_num += att_index.sum()

            adv_inputs_ori = adversary.perturb(inputs, labels)

            L2_distance = (adv_inputs_ori - inputs).squeeze()
            L2_distance = torch.linalg.norm(torch.flatten(L2_distance, start_dim=1), dim=1).data
            L2_distance = L2_distance * att_index
            total_L2_distance += L2_distance.sum().float()

            outputs = tar_net(adv_inputs_ori)
            _, predicted = torch.max(outputs, dim=1)
            tmp_index = torch.where(predicted == labels, zeros, ones)
            tmp_index = tmp_index + att_index
            success_index = torch.where(tmp_index == 2, ones, zeros)
            success_num += success_index.sum()

    print('Attack success rate: %.2f %%' %
          (100. * success_num / att_num))
    print('l2 distance:  %.4f ' % (total_L2_distance / att_num))


target_net = Net_m().cuda()
state_dict = torch.load('pretrained/net_m.pth')
target_net.load_state_dict(state_dict)
target_net.eval()

net_s = Net_s().cuda()
state_dict = torch.load(
    'saved_model_L/net_s_epoch_99.pth')
net_s.load_state_dict(state_dict)

net_m = Net_m().cuda()
state_dict = torch.load(
    'saved_model_L/net_m_epoch_99.pth')
net_m.load_state_dict(state_dict)
net_l = Net_l().cuda()
state_dict = torch.load(
    'saved_model_L/net_l_epoch_99.pth')
net_l.load_state_dict(state_dict)

attack_net = EnsembleModel([net_s, net_m, net_l])

test_adver(attack_net, target_net, opt.adv, opt.target)
