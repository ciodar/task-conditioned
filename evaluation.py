import torch
from darknet import Darknet
import dataset
from torchvision import datasets, transforms
from utils import get_all_boxes, bbox_iou, nms, read_data_cfg, load_class_names, get_image_list
from image import correct_yolo_boxes
import os
import tqdm
#from my_eval import _do_python_eval
#from lamr_ap import meanAP_LogAverageMissRate
from to_JSON import convert_predict_to_JSON
#from cfg import parse_cfg

def valid(datacfg, cfgfile, modelfile, outfile, condition=False, use_cuda=False,gpu = 0):
    options = read_data_cfg(datacfg)
    valid_images = options['valid']
    print('Validate with the list file: ',valid_images)
    name_list = options['names']
    names = load_class_names(name_list)
    #trainset_path = options['trainset_path']
    valset_path = options['valset_path']
    outpath = '/'.join(outfile.split('/')[:-1])
    outfilename = outfile.split('/')[-1]

    valid_files = get_image_list(valset_path,valid_images)
    
    m = Darknet(cfgfile)

    check_model = modelfile.split('.')[-1]
    if check_model == 'model':
        checkpoint = torch.load(modelfile)
        print('Load model from ', modelfile)
        m.load_state_dict(checkpoint['state_dict'])
        condition = True
    else:
        m.load_weights(modelfile)
        print('Load weight from ', modelfile)
    # m.print_network()


    # m.savemodel()
    if use_cuda:
        m.cuda()
    m.eval()
    #TODO Windows compatibiity
    valid_dataset = dataset.listDataset(valid_images, shape=(m.width, m.height),
                       shuffle=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                       ]))
    valid_batchsize = 2
    assert(valid_batchsize > 1)

    kwargs = {'num_workers': 4, 'pin_memory': True}
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=valid_batchsize, shuffle=False, **kwargs) 

    fps = [0]*m.num_classes
    if not os.path.exists(outpath):
        os.mkdir(outpath)
    for i in range(m.num_classes):
        buf = '%s/%s_%s.txt' % (outpath,outfilename, names[i])
        fps[i] = open(buf, 'w')
   
    lineId = -1
    
    conf_thresh = 0.005
    nms_thresh = 0.45
    if m.net_name() == 'region': # region_layer
        shape=(0,0)
    else:
        shape=(m.width, m.height)

    for count_loop, (data, target, org_w, org_h) in enumerate(tqdm.tqdm(valid_loader)):
        if use_cuda:
            data = data.cuda(gpu)

        if condition:
            output, _cls = m(data)
        else:
            output = m(data)

        batch_boxes = get_all_boxes(output, shape, conf_thresh, m.num_classes, only_objectness=0, validation=True,use_cuda=use_cuda)
        
        for i in range(len(batch_boxes)):
            lineId += 1
            fileId = os.path.basename(valid_files[lineId]).split('.')[0]
            #width, height = get_image_size(valid_files[lineId])
            width, height = float(org_w[i]), float(org_h[i])
            # print(valid_files[lineId])
            boxes = batch_boxes[i]
            correct_yolo_boxes(boxes, width, height, m.width, m.height)
            boxes = nms(boxes, nms_thresh)
            for box in boxes:
                x1 = int((box[0] - box[2]/2.0) * width)
                y1 = int((box[1] - box[3]/2.0) * height)
                x2 = int((box[0] + box[2]/2.0) * width)
                y2 = int((box[1] + box[3]/2.0) * height)

                det_conf = box[4]
                for j in range((len(box)-5)//2):
                    cls_conf = box[5+2*j]
                    cls_id = int(box[6+2*j])
                    prob = det_conf * cls_conf
                    fps[cls_id].write('%s %f %f %f %f %f\n' % (fileId, prob, x1, y1, x2, y2))

    for i in range(m.num_classes):
        fps[i].close()

def evaluation_models(*args):
    #default options
    options = {
        'datacfg' : 'data/flir.data',
        'outfile' : 'K:\\output\\det_test',
        'cfgfile' : 'cfg/yolov3_flir.cfg',
        'modelfile' : 'K:/weights/yolov3_kaist_mix_80_20.weights',
        'use_cuda' : False,
        'gpu' : 0
    }
    options.update(args[0])

    datacfg = options['datacfg']
    outfile = options['outfile']
    cfgfile = os.path.realpath(options['cfgfile'])
    modelfile = os.path.realpath(options['modelfile'])
    use_cuda = options['use_cuda'] == 'True'
    gpu = options['gpu']

    data_options = read_data_cfg(datacfg)
    testlist = data_options['valid']
    class_names = data_options['names']

    res_prefix = 'results/' + outfile

    valid(datacfg, cfgfile, modelfile, outfile,use_cuda=use_cuda)
    #cur_mAP = _do_python_eval(res_prefix, testlist, class_names, output_dir='output')
    convert_predict_to_JSON('/'.join(outfile.split('/')[:-1]))
    #all_ap, day_ap, night_ap, all_mr, day_mr, night_mr = meanAP_LogAverageMissRate()
    #print('mAP: %.4f \nap: %.4f ap_d: %.4f ap_n: %.4f lamr: %.4f mr_d: %.4f mr_n: %.4f \n' % (
    #    cur_mAP, all_ap / 100.0, day_ap / 100.0, night_ap / 100.0, all_mr / 100.0, day_mr / 100.0, night_mr / 100.0))
if __name__ == '__main__':
    import sys
    argv = sys.argv[1:]
    kwargs = {kw[0][1:]: kw[1] for kw in [ar.split('=') for ar in argv if ar.find('=') > 0]}
    args = [arg for arg in argv if arg.find('=') < 0]
    if len(sys.argv) >=1:
        evaluation_models(kwargs)
    else:
        print('Usage:')
        print(' python Evaluation_model.py')
