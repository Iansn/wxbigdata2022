import json
import random as rnd
import os
rnd.seed(1000)
path=r'./data/annotations/labeled.json'
cross_dir=r'./src/cross'
cross_num=10
def make_cross():
    cata_data = {}
    if not os.path.exists(cross_dir):
        os.makedirs(cross_dir)
    json_file = json.load(open(path, 'r', encoding='utf-8'))

    for item in json_file:

        cata_id = item['category_id']
        if cata_id not in cata_data:
            cata_data[cata_id] = [item]
        else:
            cata_data[cata_id].append(item)
    for num in range(cross_num//2):
        train=[]
        test=[]

        for cata_id in cata_data:
            data=cata_data[cata_id]
            l=len(data)
            step=l//cross_num+1
            start=num*step
            end=(num+1)*step
            if end>l:
                end=l
            for i in range(l):
                if i>=start and i<end:
                    test.append(data[i])
                else:
                    train.append(data[i])
        with open(os.path.join(cross_dir,'train_'+str(num)+'.json'), 'w', encoding='utf-8') as f:
            json.dump(train, f)
        with open(os.path.join(cross_dir,'test_'+str(num)+'.json'), 'w', encoding='utf-8') as f:
            json.dump(test, f)



if __name__=='__main__':
    make_cross()