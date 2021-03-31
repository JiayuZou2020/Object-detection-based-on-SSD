# gaussian.py画出高斯函数图像，使用latex标题
import numpy as np
import matplotlib.pyplot as plt
import math

# 定义高斯函数
def gaussian(x,sigma,mu):
    y = 1/(np.sqrt((2*math.pi))*sigma)*np.exp(-(x-mu)**2/(2*(sigma**2)))
    return y

x = np.arange(-10,10,0.1)
mu = 0
sigma = 1
y = gaussian(x, sigma, mu)
plt.figure(1)

plt.plot(x, y, 'r--',label = r'$\mu = 0, \sigma^2 = 1$')
plt.legend()
plt.title('Gussian function')
plt.xlabel(x)
plt.ylabel(y)
plt.xlim(-10.0,10.0)
plt.show()