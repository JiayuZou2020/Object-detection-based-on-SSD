import torch
import torch.nn as nn
# 进行最大池化
pool = nn.MaxPool2d((2, 2), stride=1, padding=0, return_indices=True)
input = torch.arange(0, 9, dtype=torch.float32).view(1, 1, 3, 3)

print(input)
print(input.shape) # torch.Size([1, 1, 4, 4])

# 池化前后的值以及对应的下标
output, indices = pool(input)
print(output)       
print(output.shape) 
print(indices)      
# 反池化
unpool = nn.MaxUnpool2d((2, 2), stride=1, padding=0)

result1 = unpool(output, indices)


print(result1) 
