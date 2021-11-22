"""
@Filename : mytest.py
@Time : 2021/08/26 14:33:59
@Author : zhoubishu
@Email : zhoubs11@chinaunicom.cn
@Descript : 
"""

from paddleocr import PaddleOCR
import cv2
import re
import os

# trace_field = { "发票类型": "*.发票", "开票日期": "*年*月*日",
#                "大号": "[0-9]10", "小号": "[0-9]8",
#                "发票号码": "[0-9]8", "发票代码": "[0-9]12", "第几联": "第*联"}

# pattern_小号 = re.compile(r'[0-9]{8,8}')
# pattern_大号 = re.compile(r'[0-9]{10,10}')
# pattern_发票类型 = re.compile(r'[\u4e00-\u9fa5]{,10}发票')   # 边界
# pattern_开票日期 = re.compile(r'[0-9]{4,4}年\d{1,2}月\d{1,2}日')
# pattern_第几联 = re.compile(r'第[一二三四五六七八九]联')

# result_1 = re.findall(pattern_小号, 'string')


class MyOCR:
    def __init__(self, rec_model_dir=None, det_model_dir=None, cls_model_dir=None) -> None:
        self.ocr = PaddleOCR(use_pdserving=False, use_angle_cls=True, det=True, cls=True,
                             use_gpu=False, use_tensorrt=False,
                             lang='ch', drop_score=0.8, show_log=False, 
                             det_model_dir=det_model_dir, rec_model_dir=rec_model_dir,
                             cls_model_dir=cls_model_dir,det_db_unclip_ratio=2.5,det_db_box_thresh=0.6)
        self.dict = []
        # pattern_bill = re.compile('[\u4e00-\u9fa5]{,2}发票$')
        pattern_opener = re.compile('^交款人：[\u4e00-\u9fa5]+')
        pattern_checker = re.compile('^复核人：?[\u4e00-\u9fa5]+')
        pattern_receiver = re.compile('^收款人：?[\u4e00-\u9fa5]+')
        pattern_date = re.compile('[0-9]{4,4}-\d{1,2}-\d{1,2}$')
        # pattern_big = re.compile(r'[0-9]{10,10}')
        # pattern_big = re.compile('^\d{10}$')
        # pattern_small = re.compile(r'[0-9]{8,8}')
        # pattern_small = re.compile('^\d{8}$')
        # pattern_number = re.compile(r'[0-9]{8,8}')
        pattern_number = re.compile('^[票栗]据号码：?\d{10}$')
        # pattern_code = re.compile(r'[0-9]{12,12}')
        pattern_code = re.compile('^[票栗]据代码：?\d{8}$')
        # pattern_lian = re.compile(r'第[一二三四五六七八九]联')
        # pattern_mony = re.compile('（小写）￥.*')
        pattern_mony = re.compile('[零壹贰叁肆伍陆柒捌玖拾佰仟亿角分元圆整万]{3,}')
        self.patterns = {"开票日期": pattern_date, "收款人": pattern_receiver,
                         "交款人": pattern_opener, "复核人": pattern_checker,
                         "金额合计（大写）": pattern_mony, "票据号码": pattern_number,
                         "票据代码": pattern_code}

    def dtocr(self, img):
        result = self.ocr.ocr(img)
        # print(result)
        return result

    def paser_data(self, results, im=None):
        if not (isinstance(results, list) and results and len(results[0]) == 2 and len(results[0][0]) == 4):
            return 0
        trace_field = {"开票日期": "", "收款人": "",
                       "交款人": "", "复核人": "",
                       "金额合计（大写）": "", "票据号码": "",
                       "票据代码": ""}
        temp_dict = self.patterns.copy()
        flag = isinstance(im, type(None))
        for res in results:
            positions, txts = res[0], res[1]
            if not flag:
                im = cv2.line(im, tuple(map(int, positions[0])), tuple(
                    map(int, positions[1])), (10, 255, 10))
                im = cv2.line(im, tuple(map(int, positions[1])), tuple(
                    map(int, positions[2])), (10, 255, 10))
                im = cv2.line(im, tuple(map(int, positions[2])), tuple(
                    map(int, positions[3])), (10, 255, 10))
                im = cv2.line(im, tuple(map(int, positions[3])), tuple(
                    map(int, positions[0])), (10, 255, 10))
            for key, pattern in temp_dict.items():
                result = re.findall(pattern, txts[0])
                if result:
                    trace_field[key] = result[0].split('：')[-1].split(key)[-1]
                    del temp_dict[key]
                    break
        if not flag:
            return trace_field, im
        else:
            return trace_field


# recp = r'myocrinference\inference\rec_crnn'
# detp = r'myocrinference\inference\det_db'
recp = 'model/ch_ppocr_server_v2.0_rec_infer'
# detp = 'model\ch_ppocr_server_v2.0_det_infer'
detp = 'ch_PP-OCRv2_det_infer'
clsp = 'ch_ppocr_mobile_v2.0_cls_infer'
myocr = MyOCR(rec_model_dir=recp, det_model_dir=detp, cls_model_dir=clsp)
imroot = '.'
imlist = [os.path.join(imroot,i) for i in os.listdir(imroot) if (i.endswith('.png') or i.endswith('.jpg'))]
for impath in imlist:
    im = cv2.imread(impath)
    h,w,_ = im.shape
    print(im.shape)
    # ratio = max(h,w)/800
    # if ratio>1.:
    #     im = cv2.resize(im,(0,0),fx=1/ratio,fy=1/ratio)
    data = myocr.dtocr(im)
    print(data)
    result = myocr.paser_data(data, im)
    if not result:
        continue
    rst, im = result
    print(rst)
    cv2.imshow('piao', im)
    cv2.waitKey(0)
    # break
