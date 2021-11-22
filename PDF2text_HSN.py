#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
#
# Copyright (C) 2021 #
# @Time    : 2021/6/1 9:15
# @Author  : huangshenneng
# @Email   : 15915816235@163.com
# @File    : PDF2text.py
# @Software: PyCharm
'''
安装环境： pip install PyMuPDF
'''
import copy

import fitz
# import pikepdf
import os,sys
import cv2
import numpy as np
import logging
import re

TITLE=['票据代码','金额合计','开票日期','票据号码','收款单位','复核人','收款人','小写']

def pdf2dict(path , size =1,dilate_kesize = 55):
    '''

    :param path:                输入pdf文件
    :param size:                pdf生成图片的倍数
    :param dilate_kesize:       膨胀核大小
    :return:                    pdf中每张图片和其对应生成的发票结构化数据
    '''

    #  打开PDF文件，生成一个对象
    doc = fitz.open(path)

    pg_num=0
    result_list=[]
    # 一张pdf有多张发票
    for pg in range(doc.pageCount):
        pg_num+=1
        if pg_num>1:
            break
        page = doc[pg]
        img_cv,mask_img,index_img,info,result_dict,origin_img=get_info_img(page, size, drawSize=5)


        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_kesize,1))
        mask_img = cv2.dilate(mask_img, kernel)

        #进行信息匹配
        all_match=match_info_pre(info, index_img, mask_img,origin_img)
        result_dict=match_info_filter(all_match,result_dict)
        result_list.append([img_cv,result_dict])

    return result_list


def get_info_img(page,size, drawSize = 1):

    rotate = int(0)
    zoom_x = size
    zoom_y = size
    trans = fitz.Matrix(zoom_x, zoom_y).preRotate(rotate)
    trans_old = fitz.Matrix(1, 1).preRotate(rotate)
    pm = page.getPixmap(matrix=trans, alpha=False)  # alpha=0，白色背景，不透明
    pm_old = page.getPixmap(matrix=trans_old, alpha=False)  # alpha=0，白色背景，不透明
    h_ratio = pm.h / pm_old.h
    w_ratio = pm.w / pm_old.w
    pm3 = page.get_text("words")  # alpha=0，白色背景，不透明

    info = []
    result_dict={}
    for index, t in enumerate(pm3):
        txt=t[4]
        # 去掉空格
        txt = re.sub(r"\s+", "", txt)
        key,val=split_info(txt)
        if key is not None:
            result_dict[key]=val
        info.append([int(t[0] * w_ratio), int(t[1] * h_ratio), int(t[2] * w_ratio)
                        , int(t[3] * h_ratio), index + 1, t[4], t[5], t[6]])

    # 二、这里将pdf转成numpy数据，再转成opencv能够识别的数据
    pngdata = pm.getImageData(output='png')
    image_array = np.frombuffer(pngdata, dtype=np.uint8)
    img_cv = cv2.imdecode(image_array, cv2.IMREAD_ANYCOLOR)
    origin_img=copy.deepcopy(img_cv)

    mask_img = np.zeros((img_cv.shape[0], img_cv.shape[1])).astype(np.uint8)
    index_img = np.zeros((img_cv.shape[0], img_cv.shape[1])).astype(np.int8)
    for info_one in info:
        color = tuple([int(255 * np.random.rand()) for i in range(3)])
        # cv2.putText(img_cv, str(info_one[-1]), (info_one[0], info_one[1]), 1, 2, color, drawSize)
        cv2.rectangle(img_cv, (info_one[0], info_one[1]), (info_one[2], info_one[3]), color, drawSize)
        cv2.rectangle(mask_img, (info_one[0], info_one[1]), (info_one[2], info_one[3]), 255, -1)
        cv2.rectangle(index_img, (info_one[0], info_one[1]), (info_one[2], info_one[3]), info_one[4], -1)

    return img_cv,mask_img,index_img,info,result_dict,origin_img

# TODO 第二种方法，利用矩阵，求iou值
def match_info_pre(info,index_img,mask_img,img_cv):
    show=copy.deepcopy(img_cv)
    info=np.array(info)
    temp_index_img=np.zeros_like(index_img)
    show_img=np.zeros((index_img.shape[0],index_img.shape[1],3))
    contours,_ =cv2.findContours(mask_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for id,contour in enumerate(contours):
        cv2.drawContours(temp_index_img,[contour],-1,id+1,3)
        color = tuple([int(255 * np.random.rand()) for i in range(3)])
        cv2.drawContours(show_img,[contour],-1,color,3)
        cv2.drawContours(show,[contour],-1,color,3)
    cv2.imencode('.png', show_img)[1].tofile('show_img.png')
    cv2.imencode('.png', show)[1].tofile('show_img2.png')

    all_match=[]
    for id in range(len(contours)):
        temp_id=np.where(temp_index_img==(id+1))
        vals=index_img[temp_id]
        index=np.array(list(set(vals)))
        index=np.setdiff1d(index,[0])
        index-=1
        info_match=info[index]
        all_match.append(info_match)
    return all_match

def split_info(input):

    for name_tile in TITLE:
        if input.find(name_tile) != -1:
            try:
                key,val=input.split('：')
                return key,val
            except:
                return None,None
    return None, None

def has_info(names):

    for indx,name in enumerate(names):
        for name_tile in TITLE:
            if name.find(name_tile)!=-1:
                return indx,True
    return -1,False

def print_dict(input):

    for key,item in input.items():
        print(key,'  :  ',item)

def match_info_filter(all_match,result_dict):

    for match in all_match:
        length=len(match)
        if length==2:
            name1=match[0][5]
            name2=match[1][5]
            fir_id,re=has_info([name1,name2])
            if fir_id<0:
                continue
            key=match[fir_id][5]
            key=key.strip('：').strip(':')
            result_dict[key]=match[1-fir_id][5]
        # TODO: 如果有多个则进行位置和逻辑判断

    return result_dict


if __name__ == '__main__':

    # path=r'D:\工作\财政电子票据'
    # path=r'D:\E\项目\联通\代码\code\PDF处理发票\新的数据'
    path=r'E:\项目\联通\代码\code\PDF处理发票\处理pdf'
    for index,name in enumerate(os.listdir(path)):
        # if name!='非税收入统一票据100.pdf':
        #     continue
        if name.find('.jpg')!=-1 or name.find('.py')!=-1:
            continue

        file=os.path.join(path,name)
        if not os.path.isfile(file):
            continue
        print('---------------------\n', name)
        result_list=pdf2dict(file,size =3)
        np_pdf=result_list[0][0]
        result_dict=result_list[0][1]
        print_dict(result_dict)
        out_name=name.replace('.PDF','.png').replace('.pdf','.png')
        cv2.imencode('.png',np_pdf)[1].tofile(out_name)
