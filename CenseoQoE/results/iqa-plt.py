#!/usr/bin/env python

'''
动态折线图演示示例
'''
 
import numpy as np
import matplotlib.pyplot as plt
import json
from matplotlib.animation import FuncAnimation


my_dict = json.load(open('./jingang-08-23-21-41-47.json','r'))

myList = my_dict.items()
myList = sorted(myList) 
x, y = zip(*myList) 

fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [], 'ro')

def init():
    ax.set_xlim(0, len(x))
    ax.set_ylim(0, 1)
    return ln,

def update(frame):
    xdata.append(frame)
    ydata.append(y[frame])
    ln.set_data(xdata, ydata)
    return ln,

ani = FuncAnimation(fig, update, frames=len(x),
                    init_func=init, interval=1, blit=False)
# 即时展示
# plt.show()

# 保存视频
ani.save('iqa_animation_30fps.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

# 更好看的UI界面，是通过PyQt写出来的。
# https://stackoverflow.com/questions/12459811/how-to-embed-matplotlib-in-pyqt-for-dummies


