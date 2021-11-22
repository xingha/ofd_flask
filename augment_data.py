"""
@Filename : augment_data.py
@Time : 2021/09/19 10:49:06
@Author : zhoubishu
@Email : zhoubs11@chinaunicom.cn
@Descript : augment bill data
"""

import cv2
import json
import os
import numpy as np

class ResizeAugmentor:
    def __init__(self,gt_dir,scale=None) -> None:
        self.gt_dir = gt_dir
        self.imagepath = []
        self.labels = []
        self.scale = scale
        self.lines = None

    def readlabel(self):
        if not os.path.exists(self.gt_dir):
            print('label.txt not found.')
            exit(0)
        with open(self.gt_dir,encoding='utf-8') as f:
            self.lines = f.readlines()

    def run_proc(self,show=False):
        self.readlabel()
        if not self.scale or not self.lines:
            print('dont proc')
            exit(0)
        for idx,line in enumerate(self.lines):
            imagep = line.strip().split('\t')[0].strip()
            labels = line.strip().split('\t')[1].strip()
            
            imagep = self.resize_im(imagep)
            labels = self.paser_label(labels)
            print('post-position:',labels[0]['points'])
            newlabel = '\t'.join([imagep,str(labels)+'\n'])
            self.labels.append(newlabel)
            print('[{}] {}'.format(idx,newlabel[:60]+'...'))
            if show:
                self.show_re(imagep,labels)
        self.save()

    def show_re(self,imp,labels):
        im = cv2.imread(imp)
        for line in labels:
            positions = line['points']
            im = cv2.line(im,tuple(positions[0]),tuple(positions[1]),color=(0,255,0),thickness=1)
            im = cv2.line(im,tuple(positions[1]),tuple(positions[2]),thickness=1,color=(0,255,0))
            im = cv2.line(im,tuple(positions[2]),tuple(positions[3]),thickness=1,color=(0,255,0))
            im = cv2.line(im,tuple(positions[3]),tuple(positions[0]),thickness=1,color=(0,255,0))
        cv2.imshow('re',im)
        cv2.waitKey(0)

    def save(self):
        save_name = '{}_{}.txt'.format(self.gt_dir[:-4], '%.2f'%self.scale)
        with open(save_name,'w',encoding='utf-8') as f:
            f.writelines(self.labels)
        print('save fineshed!')

    def paser_label(self,l:str):
        l_list = json.loads(l)
        print("pre-position:",l_list[0]['points'])
        for item_dic in l_list:
            item_dic['points'] = self.resize_bbox(item_dic['points'])
        return l_list

    def resize_bbox(self, bbox):
        bbox_arr = np.array(bbox)*self.scale
        return np.floor(bbox_arr).astype(int).tolist()

    def resize_im(self, imp):
        im = cv2.imread(imp)
        im_se = cv2.resize(im,(0,0),fx=self.scale,fy=self.scale)
        save_name = '{}_{}.jpg'.format(imp[:-4], '%.2f'%self.scale)
        cv2.imwrite(save_name, im_se)
        return save_name

def merge_label(dir):
    file_list = [os.path.join(dir,i) for i in os.listdir(dir) if i.startswith('Label_')]
    total = []
    for filep in file_list:
        with open(filep,encoding='utf-8') as f:
            lines = f.readlines()
        total.extend(lines)
    with open('Label_re.txt', 'w', encoding='utf-8') as f:
        f.writelines(total)
    print('merge label finished!')

def filestatus(dir):
    tot = []
    for i in os.scandir(dir):
        if i.name.endswith('.jpg'):
            line = i.path + '\t' + '1\n'
            tot.append(line)
    with open('fileState.txt','w',encoding='utf-8') as f:
        f.writelines(tot)
    print('filestatus finished!')

def get_rotate_crop_image(img, points):
    # Use Green's theory to judge clockwise or counterclockwise
    # author: biyanhua
    points = np.array(points,dtype=np.float32)
    d = 0.0
    for index in range(-1, 3):
        d += -0.5 * (points[index + 1][1] + points[index][1]) * (
                    points[index + 1][0] - points[index][0])
    if d < 0: # counterclockwise
        tmp = np.array(points)
        points[1], points[3] = tmp[3], tmp[1]

    try:
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3])))
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2])))
        pts_std = np.float32([[0, 0], [img_crop_width, 0],
                              [img_crop_width, img_crop_height],
                              [0, img_crop_height]])
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M, (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC)
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img
    except Exception as e:
        print(e)

def crop_rec_gt(path):
    with open(path, encoding='utf-8') as f:
        rfs = f.readlines()

    save_dir = 'augment-data/crop_img'
    with open(r'augment-data\rec_gt.txt', 'w',encoding='utf-8') as f:
        for rf in rfs:
            img_path = rf.strip().split('\t')[0].strip()
            st = rf.strip().split('\t')[1].strip()
            try:
                content = eval(st)
            except:
                content = json.loads(st)
            for idx,po in enumerate(content):
                point = po['points']
                im = cv2.imread(img_path)
                dst_im = get_rotate_crop_image(im,point)
                names = img_path.split('/')[-1][:-4]
                cropname = names + '_crop_%d.jpg'%idx
                imname = '/'.join([save_dir, cropname])
                cv2.imwrite(imname,dst_im)
                label = po['transcription']+'\n'
                save_line = '{}\t{}'.format(imname, label)
                f.write(save_line)
                print('log:', imname)
                # cv2.imshow('dst',dst_im)
                # cv2.waitKey(0)

# 填写文件状态
# filestatus('augment-data')
# 合并label
# merge_label('augment-data')
# exit(0)
def main():
    # 总标签
    # aug_gt = 'Label_re.txt'
    # 标准label
    gt = 'augment-data\Label.txt'
    # 1.检测数据增强
    # for i in range(20):
    #     scale = 2.5 - i*0.1
    #     augor = ResizeAugmentor(gt,scale)
    #     augor.run_proc()
    # 2.合并增强的子Label文件
    # merge_label('augment-data')
    # 3.填写文件状态(丢弃)
    # filestatus('augment-data')
    # 4.recget增强
    crop_rec_gt(gt)
    print('finished!')


if __name__ == "__main__":
    main()
