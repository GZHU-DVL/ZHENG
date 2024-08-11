from __future__ import print_function
import os
import argparse
import random
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
import torch.utils.data.sampler as sp
from nets import EnsembleModel
from attack import GradientSignAttack as GradientSignAttack_en
from attack import L2PGDAttack as L2PGDAttack_en
from attack import L2BasicIterativeAttack as L2BasicIterativeAttack_en
from advertorch.attacks import GradientSignAttack, L2PGDAttack, L2BasicIterativeAttack
from vgg import VGG

SEED = 10000

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(10000)

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading\
    workers', default=2)
parser.add_argument('--cuda', default=True, help='enables cuda')
parser.add_argument('--adv', type=str, default='PGD', help='attack method')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--target', default=False)

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

testset = torchvision.datasets.CIFAR10(root='datasets/', train=False,
                                       download=False,
                                       transform=transforms.Compose([
                                           transforms.ToTensor(),
                                       ]))

data_list = [i for i in range(0, 10000)]
testloader = torch.utils.data.DataLoader(testset, batch_size=500,
                                         sampler=sp.SubsetRandomSampler(data_list), num_workers=4)


def get_adv_in(net, inputs, labels):
    # BIM
    if opt.adv == 'BIM':
        adversary = L2BasicIterativeAttack(
            net,
            loss_fn=nn.CrossEntropyLoss(reduction="sum"),
            eps=1.3,
            eps_iter=0.2, clip_min=0.0, clip_max=1.0, nb_iter=120,
            targeted=opt.target)

    # PGD
    elif opt.adv == 'PGD':
        adversary = L2PGDAttack(
            net,
            loss_fn=nn.CrossEntropyLoss(reduction="sum"),
            eps=1.3,
            nb_iter=20, eps_iter=0.2, clip_min=0.0, clip_max=1.0,
            targeted=opt.target)
    # FGSM
    elif opt.adv == 'FGSM':
        adversary = GradientSignAttack(
            net,
            loss_fn=nn.CrossEntropyLoss(reduction="sum"),
            eps=8 / 255,
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
        total = 0.0
        for i in range(3):
            net.models[i].eval()
        for data in testloader:
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            total += labels.size(0)

            for i in range(3):
                outputs = net.models[i](inputs)
                _, predicted = torch.max(outputs.data, 1)
                correct[i] += (predicted == labels).sum()

    accuracies = []
    for i in range(3):
        accuracies.append(correct[i].float() / total)
    print('Accuracy of the networks: %.2f %%      %.2f %%     %.2f %%' %
          (100. * accuracies[0], 100. * accuracies[1], 100. * accuracies[2]))
    # Obtain the weights based on accuracies
    pro = accuracies[:]
    sum = 0
    for i in range(3):
        sum += pro[i]
    pro = [x / sum for x in pro]
    pro = torch.tensor(pro)

    # BIM
    if attack == 'BIM':
        adversary = L2BasicIterativeAttack_en(
            net, pro,
            loss_fn=nn.CrossEntropyLoss(reduction="sum"),
            eps=1.3,
            nb_iter=120, eps_iter=0.2, clip_min=0.0, clip_max=1.0,
            targeted=opt.target)

    # PGD
    elif attack == 'PGD':
        adversary = L2PGDAttack_en(
            net, pro,
            loss_fn=nn.CrossEntropyLoss(reduction="sum"),
            eps=1.3,
            nb_iter=20, eps_iter=0.2, clip_min=0.0, clip_max=1.0,
            targeted=opt.target)
    # FGSM
    elif attack == 'FGSM':
        adversary = GradientSignAttack_en(
            net, pro,
            loss_fn=nn.CrossEntropyLoss(reduction="sum"),
            eps=8 / 255,
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
            # test the images which are not classified as the specific label

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
                if i == 2:
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

            for i in range(3):
                if i == 2:
                    adv_in = get_adv_in(net.models[i], inputs, labels)
                    L2_distance = (adv_in - inputs).squeeze()
                    L2_distance = torch.linalg.norm(torch.flatten(L2_distance, start_dim=1), dim=1).data
                    L2_distance = L2_distance * att_index
                    total_L2_distances[i] += L2_distance.sum().float()
                    outputs = tar_net(adv_in)
                    _, predicted = torch.max(outputs.data, 1)
                    tmp_index = torch.where(predicted == labels, zeros, ones)
                    tmp_index = tmp_index + att_index
                    success_index = torch.where(tmp_index == 2, ones, zeros)
                    success_nums[i] += success_index.sum()

    print('Attack success rate: %.2f %%' %
          (100. * success_num / att_num))
    print('l2 distance:  %.4f ' % (total_L2_distance / att_num))
    for i in range(3):
        print(f'Asr based on net{i}: %.2f%%' % (100. * success_nums[i] / att_num))
        print('l2 distance: %.4f' % (total_L2_distances[i] / att_num))


target_net = VGG('VGG').cuda()
state_dict = torch.load('pretrained/vgg16cifar10.pth')
target_net.load_state_dict(state_dict['model'])
target_net.eval()
vgg11 = VGG('VGG11').cuda()
state_dict = torch.load(
    'saved_model_L/netD_vgg11_epoch_108.pth')
vgg11.load_state_dict(state_dict)

vgg13 = VGG('VGG13').cuda()
state_dict = torch.load(
    'saved_model_L/netD_vgg13_epoch_108.pth')
vgg13.load_state_dict(state_dict)

vgg16 = VGG('VGG16').cuda()
state_dict = torch.load(
    'saved_model_L/netD_vgg16_epoch_108.pth')
vgg16.load_state_dict(state_dict)

attack_net = EnsembleModel([vgg11, vgg13, vgg16])

test_adver(attack_net, target_net, opt.adv, opt.target)
