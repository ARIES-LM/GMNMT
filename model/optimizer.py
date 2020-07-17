import torch
import numpy as np
import math
import torch.nn as nn

class NoamOpt:
    def __init__(self, d_model, factor, warmup, optimizer, grad_clip=-1.0, delay_update=1):
        self.d_model = d_model
        self.factor = factor
        self.warmup = warmup
        self.optimizer = optimizer
        self.grad_clip = grad_clip

        # update steps
        self._step = 0
        self._rate = 0

        # for delay update
        self._step_delay = 0
        self.delay_update = delay_update

    def step(self):
        '''
         update learning rate first, then apply optimizer
        :return:
        '''

        self._step_delay += 1

        if self._step_delay % self.delay_update == 0:
            self._step += 1
            lrate = self.rate()
            for p in self.optimizer.param_groups:
                p['lr'] = lrate
            self._rate = lrate

            if self.grad_clip > 0:
                for p in self.optimizer.param_groups:
                    nn.utils.clip_grad_norm_(p['params'], self.grad_clip)

            self.optimizer.step()
            self.optimizer.zero_grad()

    def rate(self, step=None):
        '''
            lr = xxx
        '''
        if step is None:
            step = self._step

        lr = self.d_model ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5))

        return self.factor * lr

    def set_steps(self, batch_steps):
        '''
        when resume training, need this
        :param step:
        :return:
        '''
        self._step = batch_steps // self.delay_update
        self._step_delay = batch_steps

    def decay(self, r):
        self.factor *= r


class CommonOpt:
    def __init__(self, optimizer, grad_clip=-1.0, delay_update=1):
        self.optimizer = optimizer
        self.grad_clip = grad_clip

        # update steps
        self._step = 0

        # for delay update
        self._step_delay = 0
        self.delay_update = delay_update

        self._rate = 0

    def step(self):
        self._step_delay += 1

        if self._step_delay % self.delay_update == 0:
            self._step += 1

            if self.grad_clip > 0:
                for p in self.optimizer.param_groups:
                    nn.utils.clip_grad_norm_(p['params'], self.grad_clip)

            self.optimizer.step()
            self.optimizer.zero_grad()

            self._rate = self.optimizer.param_groups[0]['lr']

    def rate(self, step=None):
        '''
            lr = xxx
        '''
        if step is None:
            step = self._step

        lr = self.d_model ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5))

        return self.factor * lr

    def set_steps(self, batch_steps):
        '''
        when resume training, need this
        :param step:
        :return:
        '''
        self._step = batch_steps // self.delay_update
        self._step_delay = batch_steps

    def decay(self, r):
        for p in self.optimizer.param_groups:
            p['lr'] = p['lr'] * r
