import torch
import torch.nn as nn
from torch import optim
from utils import *
from tqdm import tqdm
import numpy as np
import os

args = dotdict({
    'lr': 0.0001,
    'dropout': 0.3,
    'epochs': 5,
    'batch_size': 64,
    'cuda': False,#torch.cuda.is_available(),
    'hidden_size': 512,
    'penalty_weight': 1e-5,
})

class SkipModule(nn.Module):
    def __init__(self, model, alpha = 1) -> None:
        super().__init__()
        self.model = model
        self.alpha = alpha
    
    def forward(self, x):
        out = self.model(x)
        out += x*self.alpha
        return out

class NN(nn.Module):

    def __init__(self, observation_dim, action_dim, hidden_size = args.hidden_size) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder = nn.Sequential(
            nn.Linear(observation_dim, hidden_size),
            nn.ReLU(),
            SkipModule(nn.Linear(self.hidden_size, self.hidden_size)),
            nn.ReLU(),
            SkipModule(nn.Linear(self.hidden_size, self.hidden_size)),
            nn.ReLU(),
        )
        self.decoder_pi = nn.Sequential(
            nn.Linear(hidden_size, action_dim),
            nn.Softmax(dim=1)
        )
        self.decoder_v = nn.Sequential(
            nn.Linear(hidden_size, 1)
        )
        self.apply(self.init_weight)
    
    def init_weight(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.encoder(x)
        pi = self.decoder_pi(x)
        v = self.decoder_v(x)
        return pi, v

class FCNet(nn.Module):

    def __init__(self, game):
        super().__init__()
        observation_dim = game.getBoardSize()
        action_dim = game.getActionSize()
        self.nnet = NN(observation_dim, action_dim)
        if args.cuda:
            self.nnet.cuda()
        self.optimizer = optim.Adam(self.nnet.parameters(),weight_decay=1e-3)


    def predict(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float)
            x = x.unsqueeze(0)
            if args.cuda:
                x = x.cuda()
        with torch.no_grad():
            pi, v = self.nnet(x)
        return pi.cpu().numpy()[0], v.cpu().numpy()[0].item()
    
    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        self.nnet.train()
        for epoch in range(args.epochs):
            print('EPOCH ::: ' + str(epoch + 1))

            pi_losses = AverageMeter()
            v_losses = AverageMeter()
            penalty_losses = AverageMeter()

            batch_count = int(len(examples) / args.batch_size)

            t = tqdm(range(batch_count), desc='Training Net')
            for _ in t:
                sample_ids = np.random.randint(len(examples), size=args.batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                # predict
                if args.cuda:
                    boards, target_pis, target_vs = boards.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()

                # compute output
                out_pi, out_v = self.nnet(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                l_penalty = self.loss_penalty(out_pi)
                total_loss = l_pi + l_v + args.penalty_weight*l_penalty

                # record loss
                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))
                penalty_losses.update(l_penalty.item(), boards.size(0))
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses, Loss_p=penalty_losses)

                # compute gradient and do SGD step
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
        self.nnet.eval()

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]
    
    def loss_penalty(self, pi):
        return -torch.sum(pi*torch.log(pi))

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        map_location = None if args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])