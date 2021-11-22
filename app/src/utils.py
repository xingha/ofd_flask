"""
@Filename : utils.py
@Time : 2021/08/31 17:25:06
@Author : zhoubishu
@Email : zhoubs11@chinaunicom.cn
@Descript : 
"""

from paddleocr import PaddleOCR
import re
import cv2
import zipfile
import xmltodict


class OCRPaser:
    def __init__(self, rec_model_dir=None, det_model_dir=None, cls_model_dir=None) -> None:
        self.ocr = PaddleOCR(use_pdserving=False, use_angle_cls=True, det=True, cls=True,
                             use_gpu=False, use_tensorrt=False,
                             lang='ch', drop_score=0.8, show_log=False,
                             det_model_dir=det_model_dir, rec_model_dir=rec_model_dir,
                             cls_model_dir=cls_model_dir, det_db_unclip_ratio=2.6, det_db_box_thresh=0.6)
        self.dict = []
        pattern_opener = re.compile('^交款人：[\u4e00-\u9fa5]+')
        pattern_checker = re.compile('^复核人：?[\u4e00-\u9fa5]+')
        pattern_receiver = re.compile('^收款人：?[\u4e00-\u9fa5]+')
        pattern_date = re.compile('^开[票栗]日期：?[0-9]{4,4}-?\d{1,2}-?\d{1,2}$')
        pattern_number = re.compile('^[票栗]据号码：?\d{10}$')
        pattern_code = re.compile('^[票栗]据代码：?\d{8}$')
        pattern_mony = re.compile('[零壹贰叁肆伍陆柒捌玖拾佰仟亿角分元圆整万]{3,}')
        self.patterns = {"开票日期": pattern_date, "收款人": pattern_receiver,
                         "交款人": pattern_opener, "复核人": pattern_checker,
                         "金额合计（大写）": pattern_mony, "票据号码": pattern_number,
                         "票据代码": pattern_code}

    def dtocr(self, img):
        result = self.ocr.ocr(img)
        # print(result)
        return result

    def paser_data(self, results):
        if not (isinstance(results, list) and results and len(results[0]) == 2 and len(results[0][0]) == 4):
            return 0
        trace_field = {"开票日期": "", "收款人": "",
                       "交款人": "", "复核人": "",
                       "金额合计（大写）": "", "票据号码": "",
                       "票据代码": ""}
        temp_dict = self.patterns.copy()
        for res in results:
            positions, txts = res[0], res[1]
            for key, pattern in temp_dict.items():
                result = re.findall(pattern, txts[0])
                if result:
                    trace_field[key] = result[0].split('：')[-1].split(key)[-1]
                    del temp_dict[key]
                    break
        return trace_field

    def paser_data2(self, results):
        if not (isinstance(results, list) and results and len(results[0]) == 2 and len(results[0][0]) == 4):
            return 0
        trace_field = {"开票日期": "", "收款人": "",
                       "交款人": "", "复核人": "",
                       "金额合计（大写）": "", "票据号码": "",
                       "票据代码": ""}
        temp_dict = self.patterns.copy()
        for res in results:
            _, txts = res[0], res[1]
            for key, pattern in temp_dict.items():
                result = re.findall(pattern, txts[0])
                if result:
                    trace_field[key] = result[0].split('：')[-1]
                    del temp_dict[key]
                    break
        return trace_field


def ofd_zip_to_json(zip_file):
    try:
        # 创建zip文件对象
        zip_file = zipfile.ZipFile(zip_file, 'r')
        # 获取zip文件中的文件名
        file_name = zip_file.namelist()
        json_list = []
        for name in ['Doc_0/Pages/Page_0/Content.xml',
                    'Doc_0/Tpls/Tpl_0/Content.xml',
                    'OFD.xml']:
            if name in file_name:
                xml_data = zip_file.read(name)
                xml_data = xmltodict.parse(xml_data)
                json_list.append(xml_data)
        result, pmessage = extract_ofd_data(json_list)
        resulted = sorted(result, key=lambda x: x[1][0])
        resulted = sorted(resulted, key=lambda x: x[1][1])
        f_list = []
        txt_list = []
        out = []
        for item in resulted:
            if '：' in item[0]:
                f_list.append(item)
            else:
                txt_list.append(item)
        for f_item in f_list:
            for txt_item in txt_list:
                if distance(f_item[1], txt_item[1]):
                    out.append((f_item[0].split('：')[0], txt_item[0]))
                    txt_list.remove(txt_item)
                    break
        pattern_mony = re.compile('^[零壹贰叁肆伍陆柒捌玖拾佰仟亿角分元圆整万]{3,}')
        for item in txt_list:
            if pattern_mony.match(item[0]):
                out.append(('金额合计（大写）', item[0]))
    except Exception as e:
        print(e)
        out = {}
        pmessage = '解析失败'
    return dict(out), pmessage


def distance(point1, point2):
    return (point1[0]+point1[2]) + 3 >= point2[0] and abs(point1[1]-point2[1]) <= 3


def extract_ofd_data(json_list):
    result = []
    # 'ofd:Page' 'ofd:Content' 'ofd:Layer' 'ofd:TextObject' '@Boundary'
    for json_data in json_list:
        try:
            orederdict = json_data['ofd:Page']
            for content in orederdict['ofd:Content']['ofd:Layer']['ofd:TextObject']:
                box_data = content['@Boundary'].split()
                bbox = []
                for point in box_data:
                    pointdata = int(float(point))
                    bbox.append(pointdata)
                text = content['ofd:TextCode']['#text']
                result.append((text, bbox))
        except:
            pass
    return result, 'ok'
