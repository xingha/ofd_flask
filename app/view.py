"""
@Filename : view.py
@Time : 2021/08/31 17:19:44
@Author : zhoubishu
@Email : zhoubs11@chinaunicom.cn
@Descript : 
"""

from flask import Flask, request, jsonify
from src.utils import OCRPaser
import numpy as np
import cv2 
import tornado
from tornado.wsgi import WSGIContainer
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
import base64
import fitz


__ALLOWED__ = ["jpg","JPG","JPEG","png","PNG","jpeg","pdf","PDF"]

class MyApp:
    __slots__ = ["__app", "model","sild_size","_fields"]
    def __init__(self) -> None:
        self.sild_size = 800
        self.__app = Flask(__name__)
        self._fields = {"upload_field":"upload_src","port":5000,"restfulapi":"/ocrpaser"}      # 配置端口号，接口地址，上传文件字段名
        # det_model_path = 'ch_PP-OCRv2_det_infer/ch_PP-OCRv2_det_infer'
        # rec_model_path = 'ch_PP-OCRv2_rec_infer/ch_PP-OCRv2_rec_infer'
        cls_model_path = 'ch_ppocr_mobile_v2.0_cls_infer'
        det_model_path = 'model/ch_ppocr_server_v2.0_det_infer'
        rec_model_path = 'model/ch_ppocr_server_v2.0_rec_infer'
        self.model = OCRPaser(rec_model_dir=rec_model_path,det_model_dir=det_model_path,cls_model_dir=cls_model_path)

    def _view(self):
        print('biging')
        pmessage = ''
        try:
            upload_src = request.files[self._fields["upload_field"]]
        except:
            return jsonify({'msg':'request parameter error', 'code':'0002', 'data': ''})
        if upload_src.filename.split('.')[-1] not in __ALLOWED__:
            return jsonify({'msg':'File not support or not found', 'code':'0006', 'data': ''})
        try:
            src,pmessage = self._requested_file2_numpy(upload_src)            
        except:
            return jsonify({'msg':'not found file', 'code':'0005', 'data': ''})
        try:
            results = self.model.dtocr(src)
            datas = self.model.paser_data(results)
        except Exception as e:
            print(e)
            return jsonify({'msg':'other error;', 'code':'9999', 'data': ''})
        return jsonify({'msg':pmessage, 'code':'0000', 'data':datas})

    def _requested_file2_numpy(self,filename):
        if filename.filename.split('.')[-1] in ["pdf","PDF"]:
            im, pagemessage = self._pymupdf_fitz(filename)
            return im, pagemessage
        else:
            img = cv2.imdecode(np.frombuffer(filename.read(), np.uint8), -1)
            return img, 'ok'

    def _pymupdf_fitz(self,filename):
        pdf_byte = filename.read()
        pdfdoc = fitz.open(stream=pdf_byte, filetype='pdf')
        pagemessage = '解析完成'
        if pdfdoc.pageCount > 1:
            pagemessage = "文件大于1页，可能解析不好"
        page = pdfdoc.load_page(0)
        rota = int(0)
        zoom_x = 1.3333   ## 分辨率为1056X816
        zoom_y = 1.3333
        mat = fitz.Matrix(zoom_x, zoom_y).prerotate(rota)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        im = pix.tobytes()
        im_arr = np.frombuffer(im, dtype=np.uint8)
        im_or = cv2.imdecode(im_arr,-1)
        return im_or, pagemessage

    def _img2base64(self,img):
        _, im_arr = cv2.imencode('.jpg', img)
        im_str = base64.b64encode(im_arr.tostring())
        return im_str.decode()

    def init_app(self):
        self.__app.add_url_rule(rule=self._fields["restfulapi"],endpoint='index',view_func=self._view,methods=["POST"])
        http_server = HTTPServer(WSGIContainer(self.__app))
        http_server.listen(self._fields["port"],address='0.0.0.0')
        print('web start...')
        IOLoop.current().start()


if __name__ == "__main__":
    a = MyApp()
    a.init_app()
