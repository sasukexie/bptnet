"""
VITSRMCL (Vision Transformer with Self-supervised Representation Memory and Contrastive Learning)
完整的SRMCL模型实现，包含：
1. MAE (Masked Autoencoder) - 自监督预训练
2. Cluster Memory - 类别原型记忆
3. Contrastive Learning - 对比学习增强

参考原始实现: baseline/SRMCL/Model/VIT_SRMCL.py 和 memory.py
"""
import collections
from abc import ABC

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed, Block


class MaskedAutoencoderViT_SR(nn.Module):
    """
    带掩码自编码器的Vision Transformer backbone
    完整复现原始SRMCL的MAE架构
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., number_class=5, norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()
        
        # ===== MAE Encoder =====
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.class_num = number_class
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding
        
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.last_blocks = Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
        self.norm = norm_layer(embed_dim)
        
        # ===== MAE Decoder =====
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # 注意：mask_token是embed_dim维度
        
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim),
                                              requires_grad=False)
        
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])
        
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)
        
        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()
    
    def initialize_weights(self):
        """初始化权重（使用sin-cos位置编码）"""
        # 初始化位置编码（sin-cos）
        pos_embed = self._get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], 
            int(self.patch_embed.num_patches ** .5),
            cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        decoder_pos_embed = self._get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.patch_embed.num_patches ** .5), 
            cls_token=True
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        
        # 初始化patch_embed
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        
        # 初始化token
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        
        # 应用通用初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def _get_2d_sincos_pos_embed(self, embed_dim, grid_size, cls_token=False):
        """生成2D sin-cos位置编码"""
        grid_h = np.arange(grid_size, dtype=np.float32)
        grid_w = np.arange(grid_size, dtype=np.float32)
        grid = np.meshgrid(grid_w, grid_h)
        grid = np.stack(grid, axis=0)
        
        grid = grid.reshape([2, 1, grid_size, grid_size])
        pos_embed = self._get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
        
        if cls_token:
            pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
        
        return pos_embed
    
    def _get_2d_sincos_pos_embed_from_grid(self, embed_dim, grid):
        assert embed_dim % 2 == 0
        
        # 使用一半维度编码h，另一半编码w
        omega = np.arange(embed_dim // 2, dtype=np.float32)
        omega /= embed_dim / 2.
        omega = 1. / 10000**omega  # (D/2,)
        
        h = grid[0].reshape(-1)  # (H*W,)
        w = grid[1].reshape(-1)  # (H*W,)
        
        out_h = np.einsum('m,d->md', h, omega)  # (H*W, D/2)
        out_w = np.einsum('m,d->md', w, omega)  # (H*W, D/2)
        
        return np.concatenate([out_h, out_w], axis=1)  # (H*W, D)
    
    def patchify(self, imgs):
        """
        将图像转换为patches
        imgs: (N, C, H, W)
        x: (N, L, patch_size**2 * C)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x
    
    def unpatchify(self, x):
        """
        将patches还原为图像
        x: (N, L, patch_size**2 * C)
        imgs: (N, C, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs
    
    def random_masking(self, x, mask_ratio):
        """
        随机掩码机制
        x: [N, L, D], sequence
        mask_ratio: 掩码比例
        """
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
    
    def forward_encoder(self, x):
        """Encoder前向传播（无掩码）"""
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        
        return x
    
    def forward_decoder(self, x, ids_restore):
        """
        Decoder前向传播
        
        Args:
            x: 已经投影到decoder空间并添加了位置编码的输入 [B, L+1, decoder_embed_dim]
            ids_restore: 恢复顺序的索引（未使用，保留接口兼容性）
        
        Returns:
            pred: 重构的patches [B, L, patch_size^2 * in_chans]
        """
        # 注意：x已经在forward_class_train中处理完毕（投影+位置编码）
        # 这里直接进行decoder blocks处理
        
        # Decoder blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        
        # 移除CLS token
        x = x[:, 1:, :]
        return x
    
    def forward_loss(self, imgs, pred, mask):
        """MAE重构损失"""
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5
        
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss
    
    def forward_class_train(self, imgs, mask_ratio=0.75, depth=2):
        """
        训练模式：带掩码的多层特征提取
        
        严格遵循原始SRMCL实现逻辑：
        - 每次循环都从原始encoder输出开始
        - 独立进行masking和重构
        - decoder_embed在拼接完成后进行
        """
        pred_m = []
        x_cls_m0 = []
        loss_m = []
        
        # 获取encoder输出（不修改）
        encoder_out = self.forward_encoder(imgs)  # [B, num_patches+1, embed_dim]
        
        for i in range(depth):
            # 每次都从原始encoder输出开始
            x_mask = encoder_out[:, 1:, :]  # [B, num_patches, embed_dim]
            x_class = encoder_out[:, :1, :]  # [B, 1, embed_dim]
            
            # 随机掩码
            x_masked, mask, ids_restore = self.random_masking(x_mask, mask_ratio)
            # x_masked: [B, num_keep, embed_dim]
            
            # 添加mask tokens（在embed_dim空间）
            mask_tokens = self.mask_token.repeat(
                x_masked.shape[0], 
                ids_restore.shape[1] - x_masked.shape[1], 
                1
            )
            
            # 拼接可见patches和mask tokens
            x_ = torch.cat([x_masked, mask_tokens], dim=1)
            # 恢复原始顺序
            x_ = torch.gather(
                x_, 
                dim=1, 
                index=ids_restore.unsqueeze(-1).repeat(1, 1, x_.shape[2])
            )
            
            # 拼接CLS token
            x_full = torch.cat([x_class, x_], dim=1)
            # x_full: [B, num_patches+1, embed_dim]
            
            # 投影到decoder空间
            decoder_x = self.decoder_embed(x_full)
            # decoder_x: [B, num_patches+1, decoder_embed_dim]
            
            # 添加decoder位置编码
            decoder_x = decoder_x + self.decoder_pos_embed
            
            # LayerNorm
            decoder_x = self.decoder_norm(decoder_x)
            
            # 保存CLS token用于分类（使用encoder空间的特征）
            x_cls_m = x_full[:, 0]  # [B, embed_dim]
            
            # Decoder前向传播（重构）
            pred = self.forward_decoder(decoder_x, ids_restore)
            
            # 计算重构损失
            loss = self.forward_loss(imgs, pred, mask)
            
            # 对pred进行pooling（原始实现的奇怪操作，保留兼容性）
            m = nn.AdaptiveAvgPool2d((1, 1))
            pred_pooled = m(pred)
            pred_pooled = pred_pooled.view(pred_pooled.shape[1], -1)
            pred_pooled = pred_pooled.unsqueeze(0)
            pred_m.append(pred_pooled)
            
            x_cls_m = x_cls_m.unsqueeze(0)
            x_cls_m0.append(x_cls_m)
            loss = loss.unsqueeze(0)
            loss_m.append(loss)
        
        # 合并多次masking的损失
        loss_m = torch.cat(loss_m, dim=0)
        
        # 使用last_blocks处理原始encoder输出
        x_final = self.last_blocks(encoder_out)
        x_cls = x_final[:, 0]
        
        # 总损失 = 第一次 + 第二次
        loss = loss_m[0] + loss_m[1]
        return loss, x_cls
    
    def forward_class_test(self, imgs):
        """测试模式：无掩码的特征提取"""
        x = self.forward_encoder(imgs)
        x = self.last_blocks(x)
        x_cls = x[:, 0]
        return x_cls


# ==================== Cluster Memory 模块 ====================
class CM_Hard(torch.autograd.Function):
    """硬样本聚类记忆更新"""
    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())
        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)
        
        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, targets.tolist()):
            batch_centers[index].append(instance_feature)
        
        for index, features_list in batch_centers.items():
            distances = []
            for feature in features_list:
                distance = feature.unsqueeze(0).mm(ctx.features[index].unsqueeze(0).t())[0][0]
                distances.append(distance.cpu().numpy())
            
            median = np.argmin(np.array(distances))
            ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * features_list[median]
            ctx.features[index] /= ctx.features[index].norm()
        
        return grad_inputs, None, None, None


def cm_hard(inputs, indexes, features, momentum=0.5):
    return CM_Hard.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class CM_Mean(torch.autograd.Function):
    """均值聚类记忆更新"""
    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())
        return outputs
    
    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)
        
        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, targets.tolist()):
            batch_centers[index].append(instance_feature)
        
        for index, features_list in batch_centers.items():
            ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * torch.stack(features_list).mean(dim=0)
            ctx.features[index] /= ctx.features[index].norm()
        
        return grad_inputs, None, None, None


def cm_mean(inputs, indexes, features, momentum=0.5):
    return CM_Mean.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class ClusterMemory(nn.Module, ABC):
    """
    聚类记忆模块 - 维护类别原型特征并计算对比学习损失
    
    Args:
        num_features: 特征维度
        num_samples: 样本数量（类别数）
        temp: 温度系数
        momentum: 动量更新系数
        use_hard: 是否使用硬样本更新
    """
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.1, use_hard=False):
        super(ClusterMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples
        self.momentum = momentum
        self.temp = temp
        self.use_hard = use_hard
        self.register_buffer('features', torch.zeros(num_samples, num_features))
    
    def forward(self, inputs, targets, return_logits=False):
        inputs = F.normalize(inputs, dim=1).cuda() if inputs.is_cuda else F.normalize(inputs, dim=1)
        
        if self.use_hard:
            outputs = cm_hard(inputs, targets, self.features, self.momentum)
        else:
            outputs = cm_mean(inputs, targets, self.features, self.momentum)
        
        outputs /= self.temp
        loss = F.cross_entropy(outputs, targets)
        
        if return_logits:
            return loss, outputs
        return loss


# ==================== VITSRMCL 主模型 ====================
class VITSRMCL(nn.Module):
    """
    完整的SRMCL模型实现
    
    包含：
    1. MAE Backbone (Encoder + Decoder)
    2. Cluster Memory (对比学习)
    3. 多任务损失组合
    
    支持两种模式：
    - Mode 1: 纯监督分类（mask_ratio=0）
    - Mode 2: MAE预训练 + 监督微调 + 对比学习（mask_ratio>0）
    
    Args:
        config (dict): 配置字典
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        model_config = config['model']
        data_config = config['data']
        
        # ===== 基础参数 =====
        self.num_classes = data_config.get('num_classes', 3)
        self.img_size = data_config.get('image_size', 28)
        self.patch_size = model_config.get('patch_size', 4)
        self.embed_dim = model_config.get('embed_dim', 256)
        self.depth = model_config.get('depth', 6)
        self.num_heads = model_config.get('num_heads', 8)
        self.mlp_ratio = model_config.get('mlp_ratio', 4.0)
        
        # 使用配置文件中的 frame_type_config
        ft_config = data_config.get('frame_type_config').get(data_config.get('frame_type'))
        in_chans = ft_config['channels']
        self.is_multimodal = ft_config.get('is_multimodal', False)
        
        # ===== MAE相关参数 =====
        self.decoder_embed_dim = model_config.get('decoder_embed_dim', 128)
        self.decoder_depth = model_config.get('decoder_depth', 4)
        self.decoder_num_heads = model_config.get('decoder_num_heads', 8)
        self.mask_ratio = model_config.get('mask_ratio', 0.0)
        self.norm_pix_loss = model_config.get('norm_pix_loss', False)
        
        # ===== 记忆模块参数 =====
        self.use_memory = model_config.get('use_memory', True)
        self.memory_momentum = model_config.get('memory_momentum', 0.1)
        self.memory_temp = model_config.get('memory_temp', 0.05)
        self.use_hard = model_config.get('use_hard', False)
        
        # ===== 损失权重 =====
        self.alpha = model_config.get('alpha', 0.5)  # 重构损失权重
        self.beta = model_config.get('beta', 0.5)    # 记忆损失权重
        
        # ===== 构建MAE Backbone =====
        self.backbone = MaskedAutoencoderViT_SR(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_chans=in_chans,
            embed_dim=self.embed_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            decoder_embed_dim=self.decoder_embed_dim,
            decoder_depth=self.decoder_depth,
            decoder_num_heads=self.decoder_num_heads,
            mlp_ratio=self.mlp_ratio,
            number_class=self.num_classes,
            norm_pix_loss=self.norm_pix_loss
        )
        
        # ===== 分类头 =====
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.num_classes)
        )
        
        # ===== Cluster Memory =====
        if self.use_memory:
            num_samples = self.num_classes
            self.memory = ClusterMemory(
                num_features=self.embed_dim,
                num_samples=num_samples,
                temp=self.memory_temp,
                momentum=self.memory_momentum,
                use_hard=self.use_hard
            )
        else:
            self.memory = None
        
        # ===== 损失函数 =====
        self.id_loss = nn.CrossEntropyLoss()
        self.classification = True
        self.mem = self.use_memory
    
    def forward_encoder_only(self, x):
        """仅使用encoder（测试模式或纯分类模式）"""
        x = self.backbone.patch_embed(x)
        x = x + self.backbone.pos_embed[:, 1:, :]
        
        cls_token = self.backbone.cls_token + self.backbone.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        for blk in self.backbone.blocks:
            x = blk(x)
        
        x = self.backbone.last_blocks(x)
        x = self.backbone.norm(x)
        return x[:, 0]  # CLS token
    
    def train_forward(self, feats, labels, recon_loss=None):
        """
        训练模式前向传播
        
        Args:
            feats: 特征向量 [B, embed_dim]
            labels: 标签 [B]
            recon_loss: 重构损失（可选）
        
        Returns:
            logits: 分类logits
            total_loss: 总损失
        """
        total_loss = 0
        
        # 分类损失
        if self.classification:
            logits = self.mlp_head(feats)
            cls_loss = self.id_loss(logits.float(), labels)
            total_loss += cls_loss
            
            # 添加重构损失
            if recon_loss is not None:
                total_loss += self.alpha * recon_loss
        else:
            logits = None
        
        # 记忆损失（对比学习）
        if self.mem and self.memory is not None:
            feat_normalized = F.normalize(feats, dim=1)
            mem_loss, mem_logits = self.memory(feat_normalized, labels, return_logits=True)
            total_loss += self.beta * mem_loss
            
            # 如果未启用分类，使用记忆logits
            if not self.classification:
                logits = mem_logits
        
        return logits, total_loss
    
    def test_forward(self, feats, labels=None):
        """
        测试模式前向传播
        
        Args:
            feats: 特征向量 [B, embed_dim]
            labels: 标签（可选，用于计算loss）
        
        Returns:
            feats: 归一化特征
            logits: 分类logits
            loss: 损失（如果提供labels）
        """
        feats_normalized = F.normalize(feats, dim=1)
        logits = self.mlp_head(feats)
        
        if labels is not None:
            loss = self.id_loss(logits.float(), labels)
            return feats_normalized, logits, loss
        else:
            return feats_normalized, logits, None
    
    def forward(self, x, labels=None, mode='auto'):
        """
        统一的前向传播接口
        
        Args:
            x: 输入数据
               - 单模态: [B, 3, H, W] (RGB图像)
               - 多模态: tuple/list ([B, 3, H, W], [B, 2, H, W]) (RGB + Flow)
               - 元组: (data, labels)
            labels: 标签（训练时必需，可选）
            mode: 'train', 'test', 或 'auto'
        
        Returns:
            训练模式: (logits, total_loss)
            测试模式: logits
        """
        # 处理多模态输入
        if isinstance(x, (list, tuple)) and self.is_multimodal:
            rgb, flow = x
            # 拼接通道: [B, 5, H, W]
            x = torch.cat([rgb, flow], dim=1)
        
        # 支持元组输入 (data, labels)
        if isinstance(x, (tuple, list)) and len(x) == 2:
            x, labels = x
        
        # 自动判断模式
        if mode == 'auto':
            mode = 'train' if self.training else 'test'
        
        if self.mask_ratio > 0 and mode == 'train':
            # MAE训练模式：带掩码的多层特征提取
            recon_loss, feats = self.backbone.forward_class_train(
                x, mask_ratio=self.mask_ratio, depth=2
            )
            
            if not self.training:
                # 测试时返回归一化特征
                return F.normalize(feats, dim=1)
            else:
                # 训练时计算多任务损失
                return self.train_forward(feats, labels, recon_loss)
        else:
            # 纯分类模式：无掩码
            feats = self.backbone.forward_class_test(x)
            
            if self.training:
                return self.train_forward(feats, labels, None)
            else:
                # 测试模式只返回logits（兼容trainer）
                _, logits, _ = self.test_forward(feats, labels)
                return logits
    
    def get_num_parameters(self):
        """获取模型参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
