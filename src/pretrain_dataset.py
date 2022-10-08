import torch
from torch.utils.data.dataset import Dataset
import json
import random as rnd
import numpy as np
from transformers import BertTokenizer
import re
import zipfile
from io import BytesIO
import category_id_map
class MutiModalDataset(Dataset):
    def __init__(self,label_data_path,unlabel_data_path,label_video_feat_path,unlabel_video_feat_path,bert_path,num_worker=8,max_token_len=128,max_frame=32,test=False,seed=1000):
        self.datas=[]
        self.test=test
        self.label_video_feat_path=label_video_feat_path
        rnd.seed(seed)
        if not test:
            self.unlabel_video_feat_path=unlabel_video_feat_path
        json_data=json.load(open(label_data_path,'r',encoding='utf-8'))
        for item in json_data:
            if len(item['title'])==0 and len(item['asr'])==0 and len(item['ocr'])==0:
                continue
            elif len(item['ocr'])>0:
                flag=0
                for ocr_s in item['ocr']:
                    s=self.filter_str(ocr_s['text'])
                    if len(s)>0:
                        flag=1
                        break
                if flag==0:
                    continue

            r=rnd.random()
            if r<0.8:
                self.datas.append(item)
        if not test:
            json_data = json.load(open(unlabel_data_path, 'r', encoding='utf-8'))
            for item in json_data:
                if len(item['title']) == 0 and len(item['asr']) == 0 and len(item['ocr']) == 0:
                    continue
                elif len(item['ocr']) > 0:
                    flag = 0
                    for ocr_s in item['ocr']:
                        s=self.filter_str(ocr_s['text'])
                        if len(s) > 0:
                            flag = 1
                            break
                    if flag == 0:
                        continue
                self.datas.append(item)

        self.tokenizer=BertTokenizer.from_pretrained(bert_path)
        self.max_token_len=max_token_len
        #self.zip_file=zipfile.ZipFile(video_feat_path, 'r')
        self.max_frame=max_frame
        self.num_worker=num_worker
        if self.num_worker > 0:
            # lazy initialization for zip_handler to avoid multiprocessing-reading error
            self.handles_label = [None for _ in range(num_worker)]
            if not test:
                self.handles_unlabel=[None for _ in range(num_worker)]
        else:
            self.handles_label = zipfile.ZipFile(label_video_feat_path, 'r')
            if not test:
                self.handles_unlabel = zipfile.ZipFile(unlabel_video_feat_path, 'r')

        if not test:
            rnd.shuffle(self.datas)
    def __len__(self):
        return len(self.datas)
    def token_text(self,text,max_length):
        if len(text)>0:
            title_data = self.tokenizer(text, return_tensors='pt', max_length=max_length, truncation=True,
                                        padding='max_length')

            title_ids = title_data['input_ids']

            title_mask = title_data['attention_mask']
        else:
            title_ids=torch.zeros(1,max_length,dtype=torch.long)
            title_mask=torch.zeros(1,max_length,dtype=torch.long)
            title_mask[0][0] = 1
            title_mask[0][-1] = 1
            title_ids[0][0] = 101
            title_ids[0][-1] = 102
        return title_ids,title_mask

    def get_visual_feats(self, id,is_labeled):
        # read data from zipfile
        if self.num_worker > 0:

            worker_id = torch.utils.data.get_worker_info().id
            if is_labeled:
                if self.handles_label[worker_id] is None:
                    self.handles_label[worker_id] = zipfile.ZipFile(self.label_video_feat_path, 'r')
                handle = self.handles_label[worker_id]
            else:
                if self.handles_unlabel[worker_id] is None:
                    self.handles_unlabel[worker_id] = zipfile.ZipFile(self.unlabel_video_feat_path, 'r')
                handle = self.handles_unlabel[worker_id]
        else:
            if is_labeled:
                handle = self.handles_label
            else:
                handle=self.handles_unlabel
        raw_feats = np.load(BytesIO(handle.read(name=f'{id}.npy')), allow_pickle=True)
        #print(raw_feats.shape)
        raw_feats = raw_feats.astype(np.float32)  # float16 to float32
        num_frames, feat_dim = raw_feats.shape

        feat = np.zeros((self.max_frame, feat_dim), dtype=np.float32)
        mask = np.ones((self.max_frame,), dtype=np.int32)
        if num_frames <= self.max_frame:
            feat[:num_frames] = raw_feats
            mask[num_frames:] = 0
        else:
            # if the number of frames exceeds the limitation, we need to sample
            # the frames.
            if self.test:
                # uniformly sample when test mode is True
                step = num_frames // self.max_frame
                select_inds = list(range(0, num_frames, step))
                select_inds = select_inds[:self.max_frame]
            else:
                # randomly sample when test mode is False
                select_inds = list(range(num_frames))
                rnd.shuffle(select_inds)
                select_inds = select_inds[:self.max_frame]
                select_inds = sorted(select_inds)
            for i, j in enumerate(select_inds):
                feat[i] = raw_feats[j]
        feat = torch.FloatTensor(feat)
        mask = torch.LongTensor(mask)
        return feat.unsqueeze(0), mask.unsqueeze(0)

    def filter_str(self, s):
        s = s.strip()
        s = re.sub(' +', ' ', s)
        s = s.replace('\t', '')
        s = s.replace('\n', '')
        s = s.replace('“', '"')
        s = s.replace('‘', '\'')
        return s

    def __getitem__(self, idx):

        data=self.datas[idx]
        id=data['id']
        selects = []

        title=data['title']
        title=self.filter_str(title)
        if len(title)>0:
            selects.append(title)
        ocr = data['ocr']
        ocr_s = ''
        ocr_map={}
        for i in range(len(ocr)):
            cur_ocr = ocr[i]['text']
            cur_ocr=self.filter_str(cur_ocr)
            if cur_ocr not in ocr_map:
                ocr_map[cur_ocr]=''

                ocr_s+=cur_ocr
        if len(ocr_s)>0:
            selects.append(ocr_s)
        asr = data['asr']
        asr=self.filter_str(asr)
        if len(asr) > 0:
            selects.append(asr)
        if not self.test:
            text=selects[rnd.randint(0,len(selects)-1)]
        else:
            text=selects[(idx%len(selects))]
        text_ids,text_mask=self.token_text(text,self.max_token_len)
        if 'category_id' in data:
            video_feat,video_mask=self.get_visual_feats(id,True)
        else:
            video_feat, video_mask = self.get_visual_feats(id, False)

        return text_ids,text_mask,video_feat,video_mask