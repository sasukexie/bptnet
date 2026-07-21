"""
HTNet 核心模型实现
Hierarchical Transformer Network for Micro-Expression Recognition

该模块实现了层次化Transformer网络，用于微表情识别。
主要特点：
1. 多层次特征提取（局部细粒度 + 全局粗粒度）
2. 面部区域划分（左眼、右眼、鼻子、左唇、右唇）
3. 自注意力机制捕捉肌肉运动关系
"""

import torch
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
from torch import nn, einsum


def cast_tuple(val, depth):
    """
    将值转换为元组
    
    Args:
        val: 输入值
        depth: 元组深度
        
    Returns:
        tuple: 转换后的元组
    """
    return val if isinstance(val, tuple) else ((val,) * depth)


class LayerNorm(nn.Module):
    """
    自定义层归一化
    
    Args:
        dim (int): 特征维度
        eps (float): 数值稳定性参数
    """

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class PreNorm(nn.Module):
    """
    预归一化包装器
    
    Args:
        dim (int): 特征维度
        fn (nn.Module): 要包装的函数/层
    """

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    """
    前馈神经网络模块
    
    Args:
        dim (int): 输入/输出维度
        mlp_mult (int): MLP扩展倍数
        dropout (float): Dropout概率
    """

    def __init__(self, dim, mlp_mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mlp_mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim * mlp_mult, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """
    多头自注意力机制
    
    Args:
        dim (int): 输入维度
        heads (int): 注意力头数
        dropout (float): Dropout概率
    """

    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()
        dim_head = dim // heads
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, c, h, w, heads = *x.shape, self.heads

        # 生成Q, K, V
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h=heads), qkv)

        # 计算注意力分数
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        # 应用注意力
        attn = self.attend(dots)
        attn = self.dropout(attn)

        # 加权求和
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)


def Aggregate(dim, dim_out):
    """
    特征聚合模块（卷积 + 归一化 + 池化）
    
    Args:
        dim (int): 输入维度
        dim_out (int): 输出维度
        
    Returns:
        nn.Sequential: 聚合模块
    """
    return nn.Sequential(
        nn.Conv2d(dim, dim_out, 3, padding=1),
        LayerNorm(dim_out),
        nn.MaxPool2d(3, stride=2, padding=1)
    )


class Transformer(nn.Module):
    """
    Transformer编码器块
    
    Args:
        dim (int): 特征维度
        seq_len (int): 序列长度
        depth (int): Transformer层数
        heads (int): 注意力头数
        mlp_mult (int): MLP扩展倍数
        dropout (float): Dropout概率
    """

    def __init__(self, dim, seq_len, depth, heads, mlp_mult, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.pos_emb = nn.Parameter(torch.randn(seq_len))

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_mult, dropout=dropout))
            ]))

    def forward(self, x):
        *_, h, w = x.shape

        # 添加位置编码
        pos_emb = self.pos_emb[:(h * w)]
        pos_emb = rearrange(pos_emb, '(h w) -> () () h w', h=h, w=w)
        x = x + pos_emb

        # 逐层处理
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class HTNet(nn.Module):
    """
    HTNet主模型：层次化Transformer网络
    
    该模型将面部分为多个区域，使用层次化Transformer捕捉
    局部和全局的面部肌肉运动特征。
    
    Args:
        image_size (int): 输入图像尺寸
        patch_size (int): Patch大小
        num_classes (int): 分类数量
        dim (int): 基础特征维度
        heads (int): 基础注意力头数
        num_hierarchies (int): 层级数量
        block_repeats (tuple): 各层Transformer块重复次数
        mlp_mult (int): MLP扩展倍数
        channels (int): 输入通道数
        dim_head (int): 每个注意力头的维度
        dropout (float): Dropout概率
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        model_config = config['model']
        data_config = config['data']
        image_size = data_config.get('image_size', 28)
        patch_size = model_config.get('patch_size', 7)
        num_classes = data_config.get('num_classes', 3)
        dim = model_config.get('dim', 256)
        heads = model_config.get('heads', 3)
        num_hierarchies = model_config.get('num_hierarchies', 3)
        block_repeats = tuple(model_config.get('block_repeats', [2, 2, 10]))
        mlp_mult = model_config.get('mlp_mult', 4)
        
        # 使用配置文件中的 frame_type_config
        ft_config = data_config.get('frame_type_config').get(data_config.get('frame_type'))
        channels = ft_config['channels']
        self.is_multimodal = ft_config.get('is_multimodal', False)
        
        # dim_head = model_config.get('dim_head', 64)
        dropout = model_config.get('dropout', 0.)

        # 参数验证
        assert (image_size % patch_size) == 0, \
            'Image dimensions must be divisible by the patch size.'

        # 计算基础参数
        patch_dim = channels * patch_size ** 2
        fmap_size = image_size // patch_size
        blocks = 2 ** (num_hierarchies - 1)
        seq_len = (fmap_size // blocks) ** 2

        # 计算各层级的维度
        hierarchies = list(reversed(range(num_hierarchies)))
        mults = [2 ** i for i in reversed(hierarchies)]
        layer_heads = list(map(lambda t: t * heads, mults))
        layer_dims = list(map(lambda t: t * dim, mults))
        last_dim = layer_dims[-1]
        layer_dims = [*layer_dims, layer_dims[-1]]
        dim_pairs = zip(layer_dims[:-1], layer_dims[1:])

        # Patch嵌入层
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (p1 p2 c) h w', p1=patch_size, p2=patch_size),
            nn.Conv2d(patch_dim, layer_dims[0], 1),
        )

        # 构建多层级Transformer
        block_repeats = cast_tuple(block_repeats, num_hierarchies)
        self.layers = nn.ModuleList([])

        for level, heads, (dim_in, dim_out), block_repeat in \
                zip(hierarchies, layer_heads, dim_pairs, block_repeats):
            is_last = level == 0
            depth = block_repeat

            self.layers.append(nn.ModuleList([
                Transformer(dim_in, seq_len, depth, heads, mlp_mult, dropout),
                Aggregate(dim_in, dim_out) if not is_last else nn.Identity()
            ]))

        # 分类头
        self.mlp_head = nn.Sequential(
            LayerNorm(last_dim),
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(last_dim, num_classes)
        )

    def forward(self, img):
        """
        前向传播
        
        Args:
            img: 输入数据
               - 单模态RGB: [B, 3, H, W]
               - 纯光流: [B, 2, H, W]
               - 多模态: tuple/list ([B, 3, H, W], [B, 2, H, W]) (RGB + Flow)
            
        Returns:
            Tensor: 分类 logits [B, num_classes]
        """
        # 处理多模态输入 (rgb_flow)
        if isinstance(img, (list, tuple)) and self.is_multimodal:
            rgb, flow = img
            # 拼接通道: [B, 5, H, W]
            img = torch.cat([rgb, flow], dim=1)
        # 纯光流模式不需要特殊处理，直接使用
        
        x = self.to_patch_embedding(img)
        # b, c, h, w = x.shape
        num_hierarchies = len(self.layers)

        # 逐层处理
        for level, (transformer, aggregate) in \
                zip(reversed(range(num_hierarchies)), self.layers):
            block_size = 2 ** level
            x = rearrange(x, 'b c (b1 h) (b2 w) -> (b b1 b2) c h w', b1=block_size, b2=block_size)
            x = transformer(x)
            x = rearrange(x, '(b b1 b2) c h w -> b c (b1 h) (b2 w)', b1=block_size, b2=block_size)
            x = aggregate(x)

        return self.mlp_head(x)

    def get_num_parameters(self):
        """获取模型参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
