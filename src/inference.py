from predict_dataset import MutiModalDataset
from model import MutiModalModel
import torch
from torch.utils.data import DataLoader
from predict_transforms import collate_fn
import category_id_map
import copy
import os

test_path=r'./data/annotations/test_b.json'
video_feat_path=r'./data/zip_feats/test_b.zip'
model_path=r'./src/model_cross'

result_path=r'./data/result.csv'
cata_num1=23
cata_num2=200
batch_size=128
bert_config_path=r'./src/chinese-roberta-wwm-ext/config.json'
bert_token_path=r'./src/chinese-roberta-wwm-ext'
def inference():
    data_loader_test = DataLoader(dataset=MutiModalDataset(data_path=test_path, video_feat_path=video_feat_path,bert_path=bert_token_path),
                                  batch_size=batch_size, num_workers=8, collate_fn=collate_fn)

    model = MutiModalModel(bert_config_path=bert_config_path,cata1_num=cata_num1,cata2_num=cata_num2)
    state_dict=torch.load(model_path,map_location='cpu')
    model.load_state_dict(state_dict,strict=False)
    model=model.cuda()
    model.eval()
    writer=open(result_path,'w')
    for i, (ids,title_ids, title_mask, asr_ids, asr_mask, ocr_ids, ocr_mask, video_feats, video_mask) in enumerate(data_loader_test):

        title_ids = title_ids.cuda()
        title_mask = title_mask.cuda()
        asr_ids = asr_ids.cuda()
        asr_mask = asr_mask.cuda()
        ocr_ids = ocr_ids.cuda()
        ocr_mask = ocr_mask.cuda()
        video_feats = video_feats.cuda()
        video_mask = video_mask.cuda()
        with torch.no_grad():
            pred2 = model(title_ids, title_mask, asr_ids, asr_mask, ocr_ids, ocr_mask, video_feats, video_mask)
            pred2=torch.argmax(pred2,dim=-1)
        pred2=pred2.to('cpu').numpy()
        for j in range(len(pred2)):
            writer.write(ids[j]+','+category_id_map.lv2id_to_category_id(int(pred2[j]))+'\n')
        print(i,len(data_loader_test))
    writer.close()
def inference_merge():
    data_loader_test = DataLoader(
        dataset=MutiModalDataset(data_path=test_path, video_feat_path=video_feat_path, bert_path=bert_token_path),
        batch_size=batch_size, num_workers=8, collate_fn=collate_fn)
    models=[]
    for i in range(5):
        model = MutiModalModel(bert_config_path=bert_config_path, cata1_num=cata_num1, cata2_num=cata_num2)
        state_dict = torch.load(os.path.join(model_path,'best_'+str(i)+'.pth'), map_location='cpu')
        model.load_state_dict(state_dict,strict=False)
        model = model.cuda()
        model.eval()
        models.append(model)


    writer = open(result_path, 'w')
    for i, (ids, title_ids, title_mask, asr_ids, asr_mask, ocr_ids, ocr_mask, video_feats, video_mask) in enumerate(
            data_loader_test):

        title_ids = title_ids.cuda()
        title_mask = title_mask.cuda()
        asr_ids = asr_ids.cuda()
        asr_mask = asr_mask.cuda()
        ocr_ids = ocr_ids.cuda()
        ocr_mask = ocr_mask.cuda()
        video_feats = video_feats.cuda()
        video_mask = video_mask.cuda()

        with torch.no_grad():
            preds = torch.zeros(title_ids.size(0), cata_num2).cuda()
            for model in models:
                pred2 = model(copy.deepcopy(title_ids), copy.deepcopy(title_mask), copy.deepcopy(asr_ids), copy.deepcopy(asr_mask), copy.deepcopy(ocr_ids), copy.deepcopy(ocr_mask), copy.deepcopy(video_feats), copy.deepcopy(video_mask))
                preds+=pred2/5.0
            pred2 = torch.argmax(preds, dim=-1)
        pred2 = pred2.to('cpu').numpy()
        for j in range(len(pred2)):
            writer.write(ids[j] + ',' + category_id_map.lv2id_to_category_id(int(pred2[j])) + '\n')
        print(i, len(data_loader_test))
    writer.close()
if __name__=='__main__':
    inference_merge()

