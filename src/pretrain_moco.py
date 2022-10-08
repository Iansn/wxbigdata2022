from pretrain_dataset import MutiModalDataset
from model import MutiModalModel

import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import random
from warmup_scheduler import WarmUpLR
from transforms import collate_fn_pretrain1
from torch.cuda.amp import autocast
from torch.cuda.amp.grad_scaler import GradScaler
unlabel_train_path=r'./data/annotations/unlabeled.json'

unlabel_video_feat_path=r'./data/zip_feats/unlabeled.zip'

label_train_path=r'./data/annotations/labeled.json'
label_video_feat_path=r'./data/zip_feats/labeled.zip'
bert_path=r'./src/chinese-roberta-wwm-ext/pytorch_model.bin'
bert_config_path=r'./src/chinese-roberta-wwm-ext/config.json'
bert_token_path=r'./src/chinese-roberta-wwm-ext'
save_model_path=r'./src/pretrain_model_moco'
cata_num1=23
cata_num2=200
batch_size=128
def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False






def build_optimizer_and_scheduler(model):
    module = (
        model.module if hasattr(model, "module") else model
    )

    # 差分学习率
    #no_decay = ["bias", "LayerNorm.weight"]
    model_param = list(module.named_parameters())

    bert_param_optimizer = []
    other_param_optimizer = []
    no_decay=[]
    for name, para in model_param:

        space = name.split('.')
        #print(name)
        if space[0] == 'bert':
            bert_param_optimizer.append((name, para))
            if 'bias' in name or 'LayerNorm.weight' in name:
                no_decay.append(name)

        else:
            other_param_optimizer.append((name, para))



    optimizer_grouped_parameters = [
        # bert other module
        {"params": [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": 0.05, 'lr': 1e-5},
        {"params": [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': 1e-5},
        # 其他模块，差分学习率
        {"params": [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": 0.05, 'lr': 1e-4},
        {"params": [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': 1e-4},

    ]


    #optimizer = AdamW(optimizer_grouped_parameters, lr=3e-4)


    return optimizer_grouped_parameters



def train():
    set_random_seed(1000, deterministic=True)
    scaler = GradScaler()
    max_epoch=10
    data_loader_train=DataLoader(dataset=MutiModalDataset(unlabel_data_path=unlabel_train_path,label_data_path=label_train_path,unlabel_video_feat_path=unlabel_video_feat_path,label_video_feat_path=label_video_feat_path,bert_path=bert_token_path),
                                batch_size=batch_size,shuffle=True,num_workers=8,collate_fn=collate_fn_pretrain1,drop_last=True)

    model=MutiModalModel(bert_config_path=bert_config_path,model_path=bert_path,cata1_num=cata_num1,cata2_num=cata_num2,pretrain=True,pretrain_moco=True)
    model = model.cuda()

    opt_param = build_optimizer_and_scheduler(model)
    opt = torch.optim.AdamW(opt_param, lr=1e-4, weight_decay=0.05)
    sch = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[9])
    #sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=max_epoch*len(data_loader_train),eta_min=1e-6)
    warm_iter = len(data_loader_train)//3
    warm_sch = WarmUpLR(opt, warm_iter)
    total_iter = 0
    last_score=-1
    best_score=-1
    best_epoch=0
    for epoch in range(max_epoch):
        model.train()
        for i, (text_ids, text_mask,video_feats,video_mask) in enumerate(data_loader_train):
            
            opt.zero_grad()

            text_ids=text_ids.cuda()
            text_mask=text_mask.cuda()
            video_feats=video_feats.cuda()
            video_mask=video_mask.cuda()
            with autocast():

                pred_text,pred_video=model.forward_pretrain_moco(text_ids,text_mask,video_feats,video_mask)
                text_video_pred=pred_text.softmax(dim=-1)
                video_text_pred=pred_video.softmax(dim=-1)
                label_text_video=torch.eye(text_video_pred.size(0),text_video_pred.size(1),device=text_video_pred.device)
                label_video_text = torch.eye(video_text_pred.size(0), video_text_pred.size(1), device=video_text_pred.device)
                loss1=torch.mean(torch.sum(-label_text_video*torch.log(text_video_pred+1e-8),dim=-1))
                loss2=torch.mean(torch.sum(-label_video_text*torch.log(video_text_pred+1e-8),dim=-1))
                loss=loss1+loss2
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            
            if total_iter < warm_iter:
                warm_sch.step()
            total_iter += 1
            print(
                'epoch:[%d/%d] iter:[%d/%d] lr:%f loss:%f loss1:%f loss2:%f tau1:%f tau2:%f last_score:%f best_score:%f best_epoch:%d' % (
                    epoch, max_epoch, i, len(data_loader_train), opt.param_groups[0]['lr'], float(loss),float(loss1),float(loss2),float(model.tau[0]),float(model.tau[1]),last_score,best_score,best_epoch))
        torch.save(model.state_dict(), os.path.join(save_model_path, 'pretrain.pth'))
        sch.step()



if __name__=='__main__':
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
    train()