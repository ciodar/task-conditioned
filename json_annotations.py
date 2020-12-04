import json
import matplotlib.pyplot as plt

def count():
    json_path = 'K:/dataset/flir dataset/video/thermal_annotations.json'
    with open(json_path) as json_file:
        data = json.load(json_file)
        annotations_dict = dict()
        for i in data['images']:
            if not i['id'] in annotations_dict:
                annotations_dict[i['id']] = 0
        for a in data ['annotations']:
            if len(a['bbox'])>0 and a['category_id']==1:
                annotations_dict[a['image_id']] +=1
    distribution_dict = dict()
    for i in annotations_dict.values():
        if not i in distribution_dict:
            distribution_dict[i] = 0
        distribution_dict[i]+=1
    plt.bar(distribution_dict.keys(), distribution_dict.values())
    plt.title('FLIR dataset person count statistics')
    plt.ylabel('images')
    plt.xlabel('annotated people')

    plt.show()
if __name__ == '__main__':
    count()