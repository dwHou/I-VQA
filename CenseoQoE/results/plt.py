#!/usr/bin/env python

'''
动态折线图演示示例
'''
 
import numpy as np
import matplotlib.pyplot as plt

'''
方法1
plt.ion()
plt.figure(1)
t_list = []
result_list = []
t = 0
 
while True:
    if t >= 10 * np.pi:
        plt.clf()
        t = 0
        t_list.clear()
        result_list.clear()
    else:
        t += np.pi / 4
        t_list.append(t)
        result_list.append(np.sin(t))
        plt.plot(t_list, result_list,c='r',ls='-', marker='o', mec='b',mfc='w')  ## 保存历史数据
        #plt.plot(t, np.sin(t), 'o')
        plt.pause(0.1)
'''

# api https://matplotlib.org/stable/api/animation_api.html
from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np
fig, ax = plt.subplots()

# 一个0~2π内的正弦曲线
x = np.arange(0, 2*np.pi, 0.01)
line, = ax.plot(x, np.sin(x))



# 动画显示
def animate(i):
    line.set_ydata(np.sin(x + i/10.0))
    return line,

def init():
    line.set_ydata(np.sin(x))
    return line,

ani = animation.FuncAnimation(fig=fig,
                              func=animate,
                              frames=100,
                              init_func=init,
                              interval=20,
                              blit=False)

# 即时显示
plt.show()


