import torch
import torch.nn as nn
import torch.utils.data
from bert.configuration_bert import BertConfig
from bert.modeling_bert import  BertModel, BertStage


from lib.seg_decoder import SimpleDecoding
from lib.vision_encoder import  VisionStage

import torch.nn.functional as F

import torch.utils.checkpoint
from timm.models.layers import  trunc_normal_


class SegModel(nn.Module):
    def __init__(self, 
                args,
                pretrain_img_size=512,
                patch_size=4,
                embed_dim=96,
                depths=[2, 2, 18, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
                mlp_ratio=4.,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.2,
                norm_layer=nn.LayerNorm,
                patch_norm=True,
                use_checkpoint=False,
                training=True
                ):

        super(SegModel, self).__init__()
        self.args = args
        self.training = training
        self.backbone = nn.ModuleList()
        
        # vision stages
        for i in range(4):
            layer = VisionStage(pretrain_img_size=pretrain_img_size,
                 patch_size=patch_size,
                 embed_dim=embed_dim,
                 depths=depths,
                 num_heads=num_heads,
                 window_size=window_size,
                 mlp_ratio=mlp_ratio,
                 qkv_bias=qkv_bias,
                 qk_scale=qk_scale,
                 drop_rate=drop_rate,
                 attn_drop_rate=attn_drop_rate,
                 drop_path_rate=drop_path_rate,
                 norm_layer=norm_layer,
                 patch_norm=patch_norm,
                 use_checkpoint=use_checkpoint,
                 i_layer=i)
            self.backbone.append(layer)

        # language stages
        config = BertConfig.from_json_file('config/config1.json')
        self.lang_stage1 = BertModel(config)

        config1 = BertConfig.from_json_file('config/config2.json')
        self.lang_stage2 = BertStage(config1)

        config2 = BertConfig.from_json_file('config/config3.json')
        self.lang_stage3 = BertStage(config2)

        config3 = BertConfig.from_json_file('config/config4.json')
        self.lang_stage4 = BertStage(config3)

        # segmentation decoder
        self.classifier = SimpleDecoding(8*embed_dim)
        
        # temperature (Only used in train mode)
        if self.training:
            self.temp = nn.Parameter(torch.ones([4]))
            
        self._init_weights()
        
    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def _init_weights(self):
        device = "cuda" if torch.cuda.is_available() else 'cpu'
        for name , m in self.named_modules():
            if 'backbone' in name:
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)
                elif isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)

        model_dict = self.state_dict()
        pretrained_dict_new = {}
        swin_pre = torch.load(self.args.pretrained_swin_weights,map_location=device)['model']
        
        for k, v in swin_pre.items():
            k = 'backbone.' + k

            if 'patch_embed' in k:
                k = k.replace('backbone','backbone.0')
                
            if 'layers.0' in k:
                k = k.replace('layers.0','0.layers')
                
            elif 'layers.1' in k:
                k = k.replace('layers.1','1.layers')
                
            elif 'layers.2' in k:
                k = k.replace('layers.2','2.layers')
               
            elif 'layers.3' in k:
                k = k.replace('layers.3','3.layers')
            
            if ('attn_mask' not in k) and ('head' not in k) and ('backbone.norm' not in k):
                pretrained_dict_new[k] = v 
    
        model_dict.update(pretrained_dict_new)
        self.load_state_dict(model_dict)
        del swin_pre
    
    def sim_map(self, vis, lang, Wh, Ww):
        vis = F.normalize(vis, dim=-1)
        lang = F.normalize(lang, dim=1)
        m = torch.matmul(vis, lang).view(vis.size(0), Wh, Ww, 1).permute(0,3,1,2).contiguous()
        return m

    def forward(self, x, l, l_mask):
        input_shape = x.shape[-2:]

        # Stage 1
        last_hidden_states, att_mask_, head_mask,cls_token1 = self.lang_stage1(l, l_mask)
        l_mask_ = l_mask.unsqueeze(dim=1)
        cls1 = cls_token1.unsqueeze(-1)
        
        x_proj1, x1, x, Wh, Ww, x_h = self.backbone[0](x, 120, 120, last_hidden_states, l_mask_)
        image_atts = torch.ones(x_h.size()[:-1],dtype=torch.long).to(x.device)

        # Stage 2
        last_hidden_states,cls_token2  = self.lang_stage2(last_hidden_states, attention_mask=att_mask_, head_mask=head_mask,
                                                encoder_hidden_states=x_h, encoder_attention_mask=image_atts)
        cls2 = cls_token2.unsqueeze(-1)

        x_proj2, x2, x, Wh, Ww, x_h = self.backbone[1](x, Wh, Ww, last_hidden_states, l_mask_)
        image_atts = torch.ones(x_h.size()[:-1],dtype=torch.long).to(x.device)

        # Stage 3
        last_hidden_states,cls_token3 = self.lang_stage3(last_hidden_states, attention_mask=att_mask_, head_mask=head_mask,
                                                encoder_hidden_states=x_h, encoder_attention_mask=image_atts)
        cls3 = cls_token3.unsqueeze(-1)

        x_proj3, x3, x, Wh, Ww, x_h = self.backbone[2](x, Wh, Ww, last_hidden_states, l_mask_)
        image_atts = torch.ones(x_h.size()[:-1],dtype=torch.long).to(x.device)

        # Stage 4
        last_hidden_states, cls_token4= self.lang_stage4(last_hidden_states, attention_mask=att_mask_, head_mask=head_mask,
                                                encoder_hidden_states=x_h, encoder_attention_mask=image_atts)
        cls4 = cls_token4.unsqueeze(-1)

        x_proj4, x4, x, Wh, Ww = self.backbone[3](x, Wh, Ww, last_hidden_states, l_mask_)

        # Segmentation Decoder
        x = self.classifier(x4,x3,x2,x1)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)
        
        # Train mode
        if self.training:
            # Compute similarity map
            t1,t2,t3,t4 = self.temp[0],self.temp[1], self.temp[2], self.temp[3]
            sim1 = self.sim_map(x_proj1, cls1, 8*Wh, 8*Ww)
            sim2 = self.sim_map(x_proj2, cls2, 4*Wh, 4*Ww)
            sim3 = self.sim_map(x_proj3, cls3, 2*Wh, 2*Ww)
            sim4 = self.sim_map(x_proj4, cls4, Wh, Ww)

            return x, sim1/t1, sim2/t2, sim3/t3, sim4/t4
        # Evaluation mode
        return x
    
