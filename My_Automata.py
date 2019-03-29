# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 16:59:00 2019

@author: Zuricho
"""

import numpy as np
import matplotlib.pyplot as plt
import random


class CellAutomata(object):
 
    def __init__(self, cells_shape):

        # cells_shape : 一个元组，表示画布的大小。
        self.cells = np.zeros(cells_shape)

        # 矩阵的四周不参与运算
        real_width = cells_shape[0] - 2
        real_height = cells_shape[1] - 2
        
        self.cells[1:-1, 1:-1] = np.random.randint(2, size=(real_width, real_height))
        self.timer = 0
        self.mask = np.ones(9)
        self.mask[4] = 0




    def update_state(self):
        #更新一次状态
        buf = np.zeros(self.cells.shape)
        cells = self.cells
        def lifegame():
            # 计算该细胞周围的存活细胞数
            neighbor = cells[i-1:i+2, j-1:j+2].reshape((-1, ))
            neighbor_num = np.convolve(self.mask, neighbor, 'valid')[0]
            if neighbor_num == 3:
                buf[i, j] = 1
            elif neighbor_num == 2:
                buf[i, j] = cells[i, j]
            else:
                buf[i, j] = 0
        def forestfire():
            neighbor = cells[i-1:i+2, j-1:j+2].reshape((-1, ))
            neighbor_num = np.convolve(self.mask, neighbor, 'valid')[0]
            if neighbor_num >= 1 and cells[i,j]==0.1:
                buf[i, j] = 1/(9*int(random.random()>0.6)+1)
                #如果绿树格位的最近邻居中有一个树在燃烧，则它变成正在燃烧的树；
            elif neighbor_num < 1 and cells[i,j]==0.1:
                buf[i, j] = int(random.random()>0.999)
                #在最近的邻居中没有正在燃烧的树的情况下树在每一时步以概率f(闪电)变为正在燃烧的树。
            elif cells[i,j]==1:
                buf[i,j]=0.1;
                #正在燃烧的树变成空格位
            else:
                buf[i, j] = int(random.random()<0.5)/10.0;
                #在空格位，树以概率p生长；

        for i in range(1, cells.shape[0] - 1):
            for j in range(1, cells.shape[0] - 1):
                if self.type=="lifegame":
                    lifegame()
                elif self.type=="forestfire":
                    forestfire()
        self.cells = buf
        self.timer += 1
   
    def plot_state(self):
        #画出当前的状态
        plt.title('Iter :{}'.format(self.timer))
        plt.imshow(self.cells)
        plt.show()
 
    def update_and_plot(self, n_iter,type_auto):
        self.type=type_auto
        #更新状态并画图
        #n_iter : 更新的轮数
        plt.ion()
        for _ in range(n_iter):
            plt.title('Iter :{}'.format(self.timer))
            plt.imshow(self.cells)
            self.update_state()
            plt.pause(0.2)
        plt.ioff()
           
 
if __name__ == '__main__':
    game = CellAutomata(cells_shape=(100, 100))
    #初始化元胞自动机
    game.update_and_plot(20,'lifegame')
    #开始执行元胞自动机并同时作图