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
    def __init__(self,data_path,video_feat_path,bert_path,num_worker=8,max_token_lens=[128,128,128],max_frame=32):
        self.datas=[]

        self.video_feat_path=video_feat_path

        json_data=json.load(open(data_path,'r',encoding='utf-8'))
        for item in json_data:
            self.datas.append(item)
        self.tokenizer=BertTokenizer.from_pretrained(bert_path)
        self.max_token_lens=max_token_lens
        #self.zip_file=zipfile.ZipFile(video_feat_path, 'r')
        self.max_frame=max_frame
        self.num_worker=num_worker
        if self.num_worker > 0:
            # lazy initialization for zip_handler to avoid multiprocessing-reading error
            self.handles = [None for _ in range(num_worker)]
        else:
            self.handles = zipfile.ZipFile(video_feat_path, 'r')

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
            title_mask[0][0]=1
            title_mask[0][-1]=1
            title_ids[0][0]=101
            title_ids[0][-1]=102
        return title_ids,title_mask

    def get_visual_feats(self, id):
        # read data from zipfile
        if self.num_worker > 0:
            worker_id = torch.utils.data.get_worker_info().id
            if self.handles[worker_id] is None:
                self.handles[worker_id] = zipfile.ZipFile(self.video_feat_path, 'r')
            handle = self.handles[worker_id]
        else:
            handle = self.handles
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
            step = num_frames // self.max_frame
            select_inds = list(range(0, num_frames, step))
            select_inds = select_inds[:self.max_frame]

            for i, j in enumerate(select_inds):
                feat[i] = raw_feats[j]
        feat = torch.FloatTensor(feat)
        mask = torch.LongTensor(mask)
        return feat.unsqueeze(0), mask.unsqueeze(0)
    def filter_str(self,s):
        s=s.strip()
        s=re.sub(' +',' ',s)
        s=s.replace('\t','')
        s=s.replace('\n','')
        s=s.replace('“','"')
        s=s.replace('‘','\'')
        return s
    def __getitem__(self, idx):

        data=self.datas[idx]
        id=data['id']
        asr=data['asr']
        asr=self.filter_str(asr)
        title=data['title']
        title=self.filter_str(title)
        ocr = data['ocr']
        ocr_s = ''

        ocr_map = {}
        for i in range(len(ocr)):
            cur_ocr = ocr[i]['text']
            cur_ocr=self.filter_str(cur_ocr)
            if cur_ocr not in ocr_map:
                ocr_map[cur_ocr] = ''

                ocr_s += cur_ocr

        asr_ids,asr_mask=self.token_text(asr,self.max_token_lens[0])
        title_ids,title_mask=self.token_text(title,self.max_token_lens[1])
        ocr_ids,ocr_mask=self.token_text(ocr_s,self.max_token_lens[2])

        video_feat,video_mask=self.get_visual_feats(id)

        return id,title_ids,title_mask,asr_ids,asr_mask,ocr_ids,ocr_mask,video_feat,video_mask