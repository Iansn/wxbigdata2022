import torch
def collate_fn(batch):
    id=[item[0] for item in batch]
    title_ids=torch.cat([item[1] for item in batch],dim=0)
    title_mask=torch.cat([item[2] for item in batch],dim=0)
    asr_ids = torch.cat([item[3] for item in batch], dim=0)
    asr_mask = torch.cat([item[4] for item in batch], dim=0)
    ocr_ids = torch.cat([item[5] for item in batch], dim=0)
    ocr_mask = torch.cat([item[6] for item in batch], dim=0)
    video_feat = torch.cat([item[7] for item in batch], dim=0)
    video_mask = torch.cat([item[8] for item in batch], dim=0)


    return id,title_ids,title_mask,asr_ids,asr_mask,ocr_ids,ocr_mask,video_feat,video_mask
def collate_fn1(batch):
    id=[item[0] for item in batch]
    text_ids=torch.cat([item[1] for item in batch],dim=0)
    text_mask=torch.cat([item[2] for item in batch],dim=0)

    video_feat = torch.cat([item[3] for item in batch], dim=0)
    video_mask = torch.cat([item[4] for item in batch], dim=0)


    return id,text_ids,text_mask,video_feat,video_mask