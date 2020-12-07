import json
import matplotlib.pyplot as plt
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


def count_all():
    json_path = 'K:/dataset/flir dataset/video/thermal_annotations.json'


if __name__ == '__main__':
    count_people()
    #count_all()