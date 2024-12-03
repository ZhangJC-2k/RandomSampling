import os
import datetime
import argparse
import random
import numpy as np
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
import torchvision.utils as tvu
import wandb
import pandas as pd
from autoattack import AutoAttack
from attacks.pgd_eot import PGD
from attacks.pgd_eot_l2 import PGDL2
from load_data import load_dataset_by_name
from load_model import load_models
from purification import PurificationForward
from utils import copy_source
from path import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

parser = argparse.ArgumentParser(description=globals()['__doc__'])
parser.add_argument('--seed', type=int, default=0, help='Random seed')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--dataset_size', type=int, default=512)

# Attack
parser.add_argument("--attack_method", type=str, default='pgd', choices=['pgd', 'pgd_l2'])
parser.add_argument('--n_iter', type=int, default=200, help='The nubmer of iterations for the attack generation')
parser.add_argument('--eot', type=int, default=5, help='The number of EOT samples for the attack')

# Purification hyperparameters in defense
parser.add_argument("--def_max_timesteps", type=str, default="1000",
                    help='The number of forward steps for each purification step in defense')
parser.add_argument('--def_num_denoising_steps', type=str, default="5",
                    help='The number of denoising steps for each purification step in defense')
parser.add_argument('--def_sampling_method', type=str, default='random', choices=['ddpm', 'ddim', 'random'],
                    help='Sampling method for the purification in defense')
parser.add_argument("--def_guidance", default=True, help="Whether guidance gpu or not in defense")
parser.add_argument('--num_ensemble_runs', type=int, default=1,
                    help='The number of ensemble runs for purification in defense')

# Purification hyperparameters in attack generation
parser.add_argument("--att_max_timesteps", type=str, default="1000",
                    help='The number of forward steps for each purification step in attack')
parser.add_argument('--att_num_denoising_steps', type=str, default="5",
                    help='The number of denoising steps for each purification step in attack')
parser.add_argument('--att_sampling_method', type=str, default='random', choices=['ddpm', 'ddim', 'random'],
                    help='Sampling method for the purification in attack')
parser.add_argument("--att_guidance", default=False, help="Whether guidance gpu or not in attack")
# Others
parser.add_argument("--use_cuda", action='store_true', help="Whether use gpu or not")
parser.add_argument("--use_wandb", action='store_true', default=False, help="Whether use wandb or not")
parser.add_argument("--wandb_project_name", default='test', help="Wandb project name")
parser.add_argument('--exp', type=str, default='test', help='Experiment name')
parser.add_argument("--dataset", type=str, default='cifar10', choices=['cifar10'])

opt = parser.parse_args()


def get_diffusion_params(max_timesteps, num_denoising_steps):
    max_timestep_list = [int(i) for i in max_timesteps.split(',')]
    num_denoising_steps_list = [int(i) for i in num_denoising_steps.split(',')]
    assert len(max_timestep_list) == len(num_denoising_steps_list)

    diffusion_steps = []
    for i in range(len(max_timestep_list)):
        diffusion_steps.append([i - 1 for i in range(max_timestep_list[i] // num_denoising_steps_list[i],
                                                     max_timestep_list[i] + 1,
                                                     max_timestep_list[i] // num_denoising_steps_list[i])])
        max_timestep_list[i] = max_timestep_list[i] - 1

    return max_timestep_list, diffusion_steps


def predict(x, args, defense_forward, num_classes):
    ensemble = torch.zeros(x.shape[0], num_classes).to(x.device)
    for _ in range(args.num_ensemble_runs):
        _x = x.clone()

        logits = defense_forward(_x)
        pred = logits.max(1, keepdim=True)[1]

        for idx in range(x.shape[0]):
            ensemble[idx, pred[idx]] += 1

    pred = ensemble.max(1, keepdim=True)[1]
    return pred


if __name__ == '__main__':

    # Set seed
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(opt.seed)
    torch.backends.cudnn.benchmark = True

    if opt.use_wandb:
        wandb.init(project=opt.wandb_project_name)

    model_src = diffusion_model_path[opt.dataset]
    dataset_root = './dataset'
    num_classes = 10

    # Set test directory name
    exp_dir = './result/{}/{}-max_t_{}-denoising_step-{}-att_{}'.format(
        opt.exp,
        opt.def_sampling_method,
        opt.def_max_timesteps,
        opt.def_num_denoising_steps,
        opt.att_num_denoising_steps,
    )
    os.makedirs('./result', exist_ok=True)
    os.makedirs('./result/{}'.format(opt.exp), exist_ok=True)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs('{}/imgs'.format(exp_dir), exist_ok=True)
    copy_source(__file__, exp_dir)

    # Device
    device = torch.device('cuda:0')

    # Load dataset
    assert opt.dataset_size % opt.batch_size == 0
    testset = load_dataset_by_name(opt.dataset, dataset_root, opt.dataset_size)
    testLoader = torch.utils.data.DataLoader(testset,
                                             batch_size=opt.batch_size,
                                             num_workers=2,
                                             pin_memory=True,
                                             drop_last=False)

    # Load models
    clf, diffusion = load_models(opt, model_src, device)

    # Process diffusion hyperparameters
    def_max_timesteps, def_diffusion_steps = get_diffusion_params(
        opt.def_max_timesteps, opt.def_num_denoising_steps)
    att_max_timesteps, att_diffusion_steps = get_diffusion_params(
        opt.att_max_timesteps, opt.att_num_denoising_steps)

    print('def_max_timesteps: ', def_max_timesteps)
    print('def_diffusion_steps: ', def_diffusion_steps)
    print('def_sampling_method: ', opt.def_sampling_method)
    print('def_guidance: ', opt.def_guidance)

    print('att_max_timesteps: ', att_max_timesteps)
    print('att_diffusion_steps: ', att_diffusion_steps)
    print('att_sampling_method: ', opt.att_sampling_method)
    print('att_guidance: ', opt.att_guidance)

    # Set diffusion process for attack and defense
    attack_forward = PurificationForward(
        clf, diffusion, att_max_timesteps, att_diffusion_steps, opt.att_sampling_method, opt.att_guidance, device)
    defense_forward = PurificationForward(
        clf, diffusion, def_max_timesteps, def_diffusion_steps, opt.def_sampling_method, opt.def_guidance, device)

    # Set adversarial attack
    if opt.dataset == 'cifar10':
        print('[Dataset] CIFAR-10')
        if opt.attack_method == 'pgd':  # PGD Linf
            eps = 8. / 255.
            attack = PGD(attack_forward, attack_steps=opt.n_iter,
                         eps=eps, step_size=0.007, eot=opt.eot)
            print('[Attack] PGD Linf | attack_steps: {} | eps: {:.3f} | eot: {}'.format(
                opt.n_iter, eps, opt.eot))
        elif opt.attack_method == 'pgd_l2':  # PGD L2
            eps = 0.5
            attack = PGDL2(attack_forward, attack_steps=opt.n_iter,
                           eps=eps, step_size=0.007, eot=opt.eot)
            print('[Attack] PGD L2 | attack_steps: {} | eps: {} | eot: {}'.format(
                opt.n_iter, eps, opt.eot))

    correct_nat = torch.tensor([0]).to(device)
    correct_adv = torch.tensor([0]).to(device)
    total = torch.tensor([0]).to(device)
    std_nat_collector = []
    std_adv_collector = []
    for idx, (x, y) in enumerate(testLoader):
        print('start', 'time = %s' % (datetime.datetime.now()))
        x = x.to(device)
        y = y.to(device)

        clf.eval()
        diffusion.eval()

        x_adv = attack(x, y)
        with torch.no_grad():
            pred_nat = predict(x, opt, defense_forward, num_classes)
            correct_nat += pred_nat.eq(y.view_as(pred_nat)).sum().item()

            pred_adv = predict(x_adv, opt, defense_forward, num_classes)
            correct_adv += pred_adv.eq(y.view_as(pred_adv)).sum().item()

        total += x.shape[0]

        std_nat_collector.append((correct_nat / total *
                                  100).item())
        std_adv_collector.append((correct_adv / total * 100).item())

        print('rank {} | {} | num_samples: {} | acc_nat: {:.3f}% | acc_adv: {:.3f}%'.format(
            0, idx, total.item(), (correct_nat / total *
                                   100).item(), (correct_adv / total * 100).item()
        ))
        tvu.save_image(x, '{}/imgs/{}_x_clean.png'.format(exp_dir, idx))
        tvu.save_image(x_adv, '{}/imgs/{}_x_adv.png'.format(exp_dir, idx))

        print('end', 'time = %s' % (datetime.datetime.now()))

    print('rank {} | num_samples: {} | acc_nat: {:.3f}% | acc_adv: {:.3f}%'.format(
        0, total.item(), (correct_nat / total *
                          100).item(), (correct_adv / total * 100).item()
    ))

    std_nat_collector = np.array(std_nat_collector)
    std_adv_collector = np.array(std_adv_collector)
    std_nat = np.std(std_nat_collector, ddof=1)
    std_adv = np.std(std_adv_collector, ddof=1)
    print(f" the nat std is: {std_nat}, the adv std is: {std_adv}")


