from dataset import MutiModalDataset
from model import MutiModalModel

import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import random

from transforms import collate_fn_train,collate_fn_test
from sklearn.metrics import f1_score, accuracy_score
import category_id_map
from torch.cuda.amp import autocast
from torch.cuda.amp.grad_scaler import GradScaler

cross_dir='./src/cross'
video_feat_path=r'./data/zip_feats/labeled.zip'
bert_path=r'./src/chinese-roberta-wwm-ext/pytorch_model.bin'
bert_config_path=r'./src/chinese-roberta-wwm-ext/config.json'
bert_token_path=r'./src/chinese-roberta-wwm-ext'
save_model_path=r'./src/model_cross'
pretrain_path=r'./src/pretrain_model_moco/pretrain.pth'
cata_num1=23
cata_num2=200
batch_size=32
cross_num=5

def evaluate(predictions, labels):
    # prediction and labels are all level-2 class ids

    lv1_preds = [category_id_map.lv2id_to_lv1id(lv2id) for lv2id in predictions]
    lv1_labels = [category_id_map.lv2id_to_lv1id(lv2id) for lv2id in labels]

    lv2_f1_micro = f1_score(labels, predictions, average='micro')
    lv2_f1_macro = f1_score(labels, predictions, average='macro')
    lv1_f1_micro = f1_score(lv1_labels, lv1_preds, average='micro')
    lv1_f1_macro = f1_score(lv1_labels, lv1_preds, average='macro')
    mean_f1 = (lv2_f1_macro + lv1_f1_macro + lv1_f1_micro + lv2_f1_micro) / 4.0

    eval_results = {'lv1_acc': accuracy_score(lv1_labels, lv1_preds),
                    'lv2_acc': accuracy_score(labels, predictions),
                    'lv1_f1_micro': lv1_f1_micro,
                    'lv1_f1_macro': lv1_f1_macro,
                    'lv2_f1_micro': lv2_f1_micro,
                    'lv2_f1_macro': lv2_f1_macro,
                    'mean_f1': mean_f1}

    return eval_results
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

def test(model,data_loader_test):
    model.eval()
    labels=[]
    preds=[]
    for i, (title_ids, title_mask, asr_ids, asr_mask, ocr_ids, ocr_mask, video_feats, video_mask, label1, label2) in enumerate(data_loader_test):

        title_ids = title_ids.cuda()
        title_mask = title_mask.cuda()
        asr_ids = asr_ids.cuda()
        asr_mask = asr_mask.cuda()
        ocr_ids = ocr_ids.cuda()
        ocr_mask = ocr_mask.cuda()
        video_feats = video_feats.cuda()
        video_mask = video_mask.cuda()

        label2 = label2.numpy()
        label2=np.argmax(label2,axis=-1)
        with torch.no_grad():
            pred2 = model(title_ids, title_mask, asr_ids, asr_mask, ocr_ids, ocr_mask, video_feats, video_mask)
            pred2=torch.argmax(pred2,dim=-1)
        pred2=pred2.to('cpu').numpy()

        for j in range(len(pred2)):
            labels.append(label2[j])
            preds.append(pred2[j])
        print(i,len(data_loader_test))
    labels=np.array(labels)
    preds=np.array(preds)
    eval_results=evaluate(preds,labels)

    print(eval_results)

    return eval_results['mean_f1']






def build_optimizer_and_scheduler(model):
    module = (
        model.module if hasattr(model, "module") else model
    )

    # 差分学习率
    #no_decay = ["bias", "LayerNorm.weight"]
    model_param = list(module.named_parameters())

    bert_param_optimizer = []
    img_param_optimizer=[]
    other_param_optimizer = []

    no_decay=[]
    for name, para in model_param:

        space = name.split('.')
        #print(name)
        if space[0] == 'bert':
            bert_param_optimizer.append((name, para))
            if 'bias' in name or 'LayerNorm.weight' in name:
                no_decay.append(name)

        elif space[0]=='transformer2' or space[0]=='transformer_layer2':
            #print(name)
            img_param_optimizer.append((name, para))
        else:
            other_param_optimizer.append((name, para))



    optimizer_grouped_parameters = [
        # bert other module
        {"params": [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": 0.05, 'lr': 5e-6},
        {"params": [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': 5e-6},
        {"params": [p for n, p in img_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": 0.05, 'lr': 5e-5},
        {"params": [p for n, p in img_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': 5e-5},
        # 其他模块，差分学习率
        {"params": [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": 0.05, 'lr': 1e-4},
        {"params": [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': 1e-4},

    ]


    #optimizer = AdamW(optimizer_grouped_parameters, lr=3e-4)


    return optimizer_grouped_parameters



def train_one_cross(data_loader_train,data_loader_test,num):
    scaler = GradScaler()
    max_epoch = 4


    model = MutiModalModel(bert_config_path=bert_config_path, model_path=bert_path, cata1_num=cata_num1,
                           cata2_num=cata_num2)
    state_dict = torch.load(pretrain_path, map_location='cpu')
    new_dict = {}
    for name in state_dict:
        if 'out_projection' in name or 'key_text_encoder' in name or 'key_video_encoder' in name:
            continue
        # print(name)
        new_dict[name] = state_dict[name]
    model.load_state_dict(new_dict, strict=False)
    model = model.cuda()

    opt_param = build_optimizer_and_scheduler(model)
    opt = torch.optim.AdamW(opt_param, lr=1e-4, weight_decay=0.05)
    sch = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[2, 3])

    best_score = -1
    last_score = -1
    best_epoch = -1
    total_iter = 0
    for epoch in range(max_epoch):
        model.train()
        avg_loss = 0
        for i, (title_ids, title_mask, asr_ids, asr_mask, ocr_ids, ocr_mask, video_feats, video_mask, label1,
                label2) in enumerate(data_loader_train):
            opt.zero_grad()
            title_ids = title_ids.cuda()
            title_mask = title_mask.cuda()
            asr_ids = asr_ids.cuda()
            asr_mask = asr_mask.cuda()
            ocr_ids = ocr_ids.cuda()
            ocr_mask = ocr_mask.cuda()
            video_feats = video_feats.cuda()
            video_mask = video_mask.cuda()
            label1 = label1.cuda()
            label2 = label2.cuda()
            with autocast():
                pred2 = model(title_ids, title_mask, asr_ids, asr_mask, ocr_ids, ocr_mask, video_feats,
                                     video_mask)
                loss2 = torch.mean(torch.sum(-label2 * torch.log(pred2 + 1e-8), dim=-1))
                loss = loss2
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            avg_loss += float(loss)
            total_iter += 1
            print(
                'cross_num:[%d/%d] epoch:[%d/%d] iter:[%d/%d] lr1:%f lr2:%f loss:%f loss2:%f last_score:%f best_score:%f best_epoch:%d' % (
                    num,cross_num,epoch, max_epoch, i, len(data_loader_train), opt.param_groups[0]['lr'], opt.param_groups[2]['lr'],
                    float(loss), float(loss2), last_score,
                    best_score, best_epoch))
        score = test(model, data_loader_test)
        last_score = score
        if score >= best_score:
            best_score = score
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(save_model_path, 'best_'+str(num)+'.pth'))
        sch.step()
def train():
    set_random_seed(1000, deterministic=True)
    for num in range(cross_num):


        label_weight2,label_weight1=category_id_map.get_cata_weight(os.path.join(cross_dir,'train_'+str(num)+'.json'))
        data_loader_train=DataLoader(dataset=MutiModalDataset(data_path=os.path.join(cross_dir,'train_'+str(num)+'.json'),video_feat_path=video_feat_path,
                                                          id_weight2=label_weight2,id_weight1=label_weight1,
                                                          cata1_num=cata_num1,cata2_num=cata_num2,
                                                          bert_path=bert_token_path),
                                batch_size=batch_size,num_workers=8,collate_fn=collate_fn_train,shuffle=True,drop_last=True)
        data_loader_test = DataLoader(dataset=MutiModalDataset(data_path=os.path.join(cross_dir,'test_'+str(num)+'.json'), video_feat_path=video_feat_path,
                                                           cata1_num=cata_num1, cata2_num=cata_num2,
                                                           bert_path=bert_token_path,test=True),
                                  batch_size=batch_size*2, num_workers=8, collate_fn=collate_fn_test)

        train_one_cross(data_loader_train,data_loader_test,num)



if __name__=='__main__':
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
    train()