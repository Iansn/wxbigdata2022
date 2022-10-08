import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
import torch.nn.functional as F



class MutiModalModel(nn.Module):
    def __init__(self, bert_config_path, cata1_num,cata2_num,model_path=None,pretrain=False,pretrain_moco=False):
        super(MutiModalModel,self).__init__()
        self.bert_config = BertConfig.from_pretrained(bert_config_path)
        # self.bert_config.output_hidden_states = True
        self.bert = BertModel(self.bert_config)
        if model_path is not None:
            state_dict = torch.load(model_path, map_location='cpu')
            new_state_dict = {}
            for name in state_dict:
                new_name = name.replace('bert.', '')

                new_state_dict[new_name] = state_dict[name]
            self.bert.load_state_dict(new_state_dict, strict=False)

        self.transformer_layer2 = nn.TransformerEncoderLayer(768, nhead=8, dim_feedforward=3072, activation='gelu')
        self.transformer2 = nn.TransformerEncoder(self.transformer_layer2, 1)
        self.pretrain=pretrain
        self.pretrain_moco=pretrain_moco
        if not self.pretrain:

            self.transformer_layer1 = nn.TransformerEncoderLayer(768, nhead=8, dim_feedforward=3072, activation='gelu')
            self.transformer1 = nn.TransformerEncoder(self.transformer_layer1, 3)


            self.cata2_output = nn.Sequential(

                nn.Linear(768,cata2_num)
            )
        else:
            if self.pretrain_moco:
                self.tau = nn.Parameter(torch.ones(2)*0.05, requires_grad=True)
            else:
                self.tau=nn.Parameter(torch.ones(1)*0.05,requires_grad=True)
            self.out_projection=nn.Sequential(
                nn.Linear(768,768),
                nn.GELU(),
                nn.Linear(768,768)
            )

            if self.pretrain_moco:

                self.out_projection_k = nn.Sequential(
                    nn.Linear(768, 768),
                    nn.GELU(),
                    nn.Linear(768, 768)
                )

                self.key_text_encoder = BertModel(self.bert_config)
                transformer_layer = nn.TransformerEncoderLayer(768, nhead=8, dim_feedforward=3072, activation='gelu')

                self.key_video_encoder = nn.TransformerEncoder(transformer_layer, 1)
                for param1, param2 in zip(self.bert.parameters(), self.key_text_encoder.parameters()):
                    param2.data.copy_(param1.data)
                    param2.requires_grad = False

                for param1, param2 in zip(self.transformer2.parameters(), self.key_video_encoder.parameters()):
                    param2.data.copy_(param1.data)
                    param2.requires_grad = False
                for param1, param2 in zip(self.out_projection.parameters(),
                                          self.out_projection_k.parameters()):
                    param2.data.copy_(param1.data)
                    param2.requires_grad = False


                self.K = 1024
                self.register_buffer("queue_text", torch.randn(self.K, 768))
                self.register_buffer("queue_video", torch.randn(self.K, 768))
                self.queue_text = F.normalize(self.queue_text, dim=-1)
                self.queue_video = F.normalize(self.queue_video, dim=-1)

                self.register_buffer("queue_ptr_text", torch.zeros(1, dtype=torch.long))
                self.register_buffer("queue_ptr_video", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        m=0.999
        for param_q, param_k in zip(self.bert.parameters(), self.key_text_encoder.parameters()):
            param_k.data = param_k.data *m  + param_q.data * (1. - m)
        for param_q, param_k in zip(self.transformer2.parameters(), self.key_video_encoder.parameters()):
            param_k.data = param_k.data *m  + param_q.data * (1. - m)
        for param_q, param_k in zip(self.out_projection.parameters(), self.out_projection_k.parameters()):
            param_k.data = param_k.data *m  + param_q.data * (1. - m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, video_keys,text_keys):
        # gather keys before updating queue

        batch_size = video_keys.shape[0]

        ptr_text = int(self.queue_ptr_text)
        ptr_video=int(self.queue_ptr_video)

        # replace the keys at ptr (dequeue and enqueue)
        if ptr_text+batch_size<=self.K:
            self.queue_text[ptr_text:ptr_text + batch_size] = text_keys
        else:
            self.queue_text[ptr_text:self.K]=text_keys[:self.K-ptr_text]
            self.queue_text[0:ptr_text+batch_size-self.K]=text_keys[self.K-ptr_text:]

        ptr_text = (ptr_text + batch_size) % self.K  # move pointer

        self.queue_ptr_text[0] = ptr_text

        if ptr_video+batch_size<=self.K:
            self.queue_video[ptr_video:ptr_video + batch_size] = video_keys
        else:
            self.queue_video[ptr_video:self.K]=video_keys[:self.K-ptr_video]
            self.queue_video[0:ptr_video+batch_size-self.K]=video_keys[self.K-ptr_video:]
        ptr_video = (ptr_video + batch_size) % self.K  # move pointer

        self.queue_ptr_video[0] = ptr_video

    def forward_pretrain_moco(self,text_ids,text_mask,video_feats,video_mask):
        text_feat=self.bert(text_ids,attention_mask=text_mask)
        text_feat=text_feat[0]
        video_feats = video_feats.permute(1, 0, 2)



        with torch.no_grad():
            self._momentum_update_key_encoder()
            text_feat_k=self.key_text_encoder(text_ids,attention_mask=text_mask)
            text_feat_k=text_feat_k[0]
            video_feat_k=self.key_video_encoder(video_feats,src_key_padding_mask=(video_mask == 0))
            video_feat_k = video_feat_k.permute(1, 0, 2)
            text_feat_k = text_feat_k * text_mask.view(text_mask.size(0), text_mask.size(1), 1)
            text_feat_k = torch.sum(text_feat_k, dim=1) / torch.sum(text_mask, dim=1, keepdim=True)

            video_feat_k = video_feat_k * video_mask.view(video_mask.size(0), video_mask.size(1), 1)
            video_feat_k = torch.sum(video_feat_k, dim=1) / torch.sum(video_mask, dim=1, keepdim=True)
            text_feat_k = self.out_projection_k(text_feat_k)
            video_feat_k = self.out_projection_k(video_feat_k)
            text_feat_k=F.normalize(text_feat_k)
            video_feat_k=F.normalize(video_feat_k)
        video_feats = self.transformer2(video_feats, src_key_padding_mask=(video_mask == 0))
        video_feats = video_feats.permute(1, 0, 2)




        text_feat=text_feat*text_mask.view(text_mask.size(0),text_mask.size(1),1)
        text_feat = torch.sum(text_feat, dim=1) / torch.sum(text_mask, dim=1, keepdim=True)

        video_feats = video_feats * video_mask.view(video_mask.size(0), video_mask.size(1), 1)
        video_feats = torch.sum(video_feats, dim=1) / torch.sum(video_mask, dim=1, keepdim=True)
        text_feat=self.out_projection(text_feat)
        video_feats=self.out_projection(video_feats)
        text_feat = F.normalize(text_feat, dim=-1)
        video_feats = F.normalize(video_feats, dim=-1)

        l_pos_text=torch.matmul(text_feat,video_feat_k.permute(1,0))
        l_neg_text=torch.matmul(text_feat,self.queue_video.clone().detach().permute(1,0))
        l_pos_text=torch.cat([l_pos_text,l_neg_text],dim=1)/torch.clamp(self.tau[0],0.005,2.0)

        l_pos_video=torch.matmul(video_feats,text_feat_k.permute(1,0))
        l_neg_video=torch.matmul(video_feats,self.queue_text.clone().detach().permute(1,0))
        l_pos_video=torch.cat([l_pos_video,l_neg_video],dim=1)/torch.clamp(self.tau[1],0.005,2.0)

        self._dequeue_and_enqueue(video_feat_k,text_feat_k)



        return l_pos_text,l_pos_video


    def forward(self,title_ids,title_mask,asr_ids,asr_mask,ocr_ids,ocr_mask,video_feats,video_mask):

        title_feat = self.bert(title_ids, attention_mask=title_mask)
        title_feat = title_feat[0]


        asr_feat = self.bert(asr_ids, attention_mask=asr_mask)
        asr_feat = asr_feat[0]



        ocr_feat = self.bert(ocr_ids, attention_mask=ocr_mask)
        ocr_feat = ocr_feat[0]







        all_feats_text = torch.cat([title_feat, asr_feat, ocr_feat], dim=1)
        all_mask_text = torch.cat([title_mask, asr_mask, ocr_mask], dim=-1)

        all_feats_text = all_feats_text.permute(1, 0, 2)

        video_feats = video_feats.permute(1, 0, 2)
        # print(all_feats_text,torch.sum(all_mask_text,dim=-1))
        video_feats = self.transformer2(video_feats, src_key_padding_mask=(video_mask == 0))

        
        
        
        all_feats = torch.cat([all_feats_text, video_feats], dim=0)
        all_mask = torch.cat([all_mask_text, video_mask], dim=-1)
        
        all_feats = self.transformer1(all_feats, src_key_padding_mask=(all_mask == 0))
        cata2_feat = all_feats.permute(1, 0, 2)
        cata2_feat = cata2_feat * all_mask.view(all_mask.size(0), all_mask.size(1), 1)

        video_feats=cata2_feat[:,all_mask_text.size(1):,:]


        cata2_feat = torch.sum(video_feats, dim=1) / torch.sum(video_mask, keepdim=True, dim=-1)
        cata2_out = self.cata2_output(cata2_feat)
        return cata2_out.softmax(dim=-1)