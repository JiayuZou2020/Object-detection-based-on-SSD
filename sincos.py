# sincos.py画图教程，常见的画图操作
import numpy as np
import matplotlib.pyplot as plt
import math

# 以下两行代码解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# python中用math.pi来表示圆周率
# np.arrange(start, end, step)
x = np.arange(0, math.pi/3, 0.05)
y = np.sin(2*x)+2*np.cos(x)

# plt.title('函数图像演示')
plt.figure(1)
plt.subplot(121)
plt.title('Functions play')
plt.xlabel('x')
plt.ylabel('y')

# 'b--'表示设置图形的显示格式为蓝色虚线
plt.plot(x/math.pi,y,'b--')
plt.plot(x/math.pi,2*y,'rs')
plt.plot(x/math.pi,3*y,'g*')

# plt.grid(True)表示给图像添加背景格点
plt.grid(True)

# plt.text(position,text)表示添加文字备注
plt.text(0.03,3.0,r'$y = sin(2x)+2cos(x)$')

plt.subplot(122)
y1 = np.sin(x)

# label用于设置标签,注意还要加上plt.legend()才会显示label
plt.plot(x/math.pi,y1, 'c-*', label = 'sinx')
plt.legend()

# 设置latex显示效果
plt.text(0.05,0.6,r'$ \mu = 1,\ \sigma=2$')

# fontsize用来设置显示的大小
plt.xlabel(r'$\omega$', fontsize = 20)
plt.ylabel(r'$\alpha =  sin(\omega)$')

# 保存图片
plt.savefig('functions play')
plt.show()
