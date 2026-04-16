# lightweight model inspect

```
from accelerate import init_empty_weights
from diffusers import FluxTransformer2DModel
import torch

# 从 Hub 只下载配置文件（config.json），不下载权重
config = FluxTransformer2DModel.load_config(
    "black-forest-labs/FLUX.1-dev",
    subfolder="transformer"
)

# 用 meta device 实例化，零内存分配
with init_empty_weights():
    model = FluxTransformer2DModel.from_config(config)

# 现在可以自由探测结构
print(model)
for name, param in model.named_parameters():
    print(name, param.shape, param.dtype, param.device)
```

# flatten module

```python
flat = nn.Parameter(torch.cat([p.detach().reshape(-1) for p in module.parameters()]))

offset = 0
for p in module.parameters():
    p.data = flat[offset:offset+p.numel()].view(p.shape)
    offset += p.numel()
```
