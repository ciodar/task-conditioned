import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import pandas as pd

def count_people():
    json_path = 'K:/dataset/flir dataset/train/thermal_annotations.json'
    output_path = 'K:/dataset/flir output dataset/'
    category_dict_path = 'K:/dataset/flir dataset/train/thermal_annotations.json'
    with open(json_path) as json_file:
        data = json.load(json_file)
        categories = ['person','car','bicycle','dog']
        cat = pd.DataFrame(data['categories']).rename(columns={'id':'category_id','name':'category'})
        #for c in categories:
        df = pd.merge(pd.DataFrame(data['annotations']),cat,on=['category_id'],how='inner')
        df.to_excel('annotations.xlsx')
        #gby_category = df.value_counts(['category'])
        #plt.bar(range(len(gby_category)), gby_category.values, align='center')
        #plt.xticks(range(len(gby_category)), gby_category.index.values, size='small')
        #plt.show()
        #people_group = df[['image_id','category']].groupby(['image_id','category']).count().reset_index(name='count')
        #people_group.to_excel("people_group.xlsx")
        #gby_ann_num = people_group.groupby(['category','count']).size().reset_index(name='count_q')
        #plt.bar(gby_ann_num['count_q'], gby_ann_num.index.values, align='center')
        #plt.show()
        #gby_people.hist(by)
        #print(people_group)

def square_mean_loss(annotation_json,detection_json):
    with open(annotation_json) as ann_file:
        ann_data = json.load(ann_file)
    with open(detection_json) as det_file:
        det_data = json.load(det_file)
    cat = pd.DataFrame(ann_data['categories']).rename(columns={'id':'category_id','name':'category'})
    images = pd.DataFrame(ann_data['images']).rename(columns={'id':'image_id'})
    ann_df = pd.merge(pd.DataFrame(ann_data['annotations']),cat,on=['category_id'],how='inner')
    ann_df = pd.merge(images,ann_df,on=['image_id'],how='left')
    ann_df['image_id'] = ann_df['image_id'].astype('category')
    ann_df = ann_df[ann_df['category'] == 'person'].groupby(['image_id']).size().reset_index(name='count')
    det_df = pd.DataFrame(det_data)
    conf = np.linspace(0,1,40)
    mse = []
    positive = []
    negative = []
    for i in conf:
        det_df_c = det_df[det_df['score']>=i].groupby(['image_id']).size().reset_index(name='count')
        df = ann_df.merge(det_df_c,how='left',on='image_id').fillna(0)
        i_mse = mean_squared_error(df['count_x'], df['count_y'], squared=False)
        print('MSE for i= {:.4f}: {:.4f}'.format(i,i_mse))
        mse.append(i_mse)
        negative.append((df[df['count_x'] - df['count_y']>0].count_x - df[df['count_x'] - df['count_y']>0].count_y).sum())
        positive.append((df[df['count_x'] - df['count_y']<0].count_y - df[df['count_x'] - df['count_y']<0].count_x).sum())
    plt.plot(conf,negative,label='negative',marker='o')
    plt.plot(conf,positive,label='positive',marker='o')
    plt.yticks(np.arange(0, 80000, 5000))
    plt.xticks(np.arange(0, 1, 0.05),rotation=90)
    plt.grid()
    plt.legend()
    plt.xlabel('confidence')
    plt.ylabel('errors')
    plt.show()
    plt.plot(conf,mse,marker='o')
    plt.ylim(0,30)
    plt.xticks(np.arange(0, 1, 0.05),rotation=90)
    plt.yticks(np.arange(0, 30, 2))
    plt.grid()
    plt.xlabel('confidence')
    plt.ylabel('RMSE')
    plt.show()
    return()

def mse_by_det_num(annotation_json,detection_json,conf):
    with open(annotation_json) as ann_file:
        ann_data = json.load(ann_file)
    with open(detection_json) as det_file:
        det_data = json.load(det_file)
    cat = pd.DataFrame(ann_data['categories']).rename(columns={'id':'category_id','name':'category'})
    images = pd.DataFrame(ann_data['images']).rename(columns={'id':'image_id'})
    ann_df = pd.merge(pd.DataFrame(ann_data['annotations']),cat,on=['category_id'],how='inner')
    ann_df = pd.merge(images,ann_df,on=['image_id'],how='left')
    ann_df['image_id'] = ann_df['image_id'].astype('category')
    ann_df['file_name'] = ann_df['file_name'].astype('category')
    det_df = pd.DataFrame(det_data)
    det_df_c = det_df[det_df['score']>=conf].groupby(['image_id']).size().reset_index(name='count')
    df = ann_df.merge(det_df_c,how='left',on='image_id').fillna(0)
    mse = {}
    for i in sorted(df['count_x'].unique()):
        mse[i] = mean_squared_error(df[df['count_x']==i].count_x, df[df['count_x']==i].count_y, squared=False)
    plt.plot(*zip(*mse.items()))
    #plt.ylim(0,30)
    #plt.xticks(np.arange(0, 1, 0.05),rotation=90)
    #plt.yticks(np.arange(0, 30, 2))
    plt.grid()
    plt.xlabel('# of detection per image')
    plt.ylabel('RMSE')
    plt.show()
    return()




def count_all():
    json_path = 'K:/dataset/flir dataset/video/thermal_annotations.json'


if __name__ == '__main__':
    #count_people()
    #square_mean_loss('K:/dataset/flir dataset/train/thermal_annotations.json','K:\output\detection_results.json')
    mse_by_det_num('K:/dataset/flir dataset/train/thermal_annotations.json','K:\output\detection_results.json',0.1795)