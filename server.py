# -*- coding: UTF-8 -*-
import tornado.ioloop
import tornado.web
import tornado.options
import tornado.httpserver 

import argparse
import torch
import main
import json as js
import copy
from tornado.options import define, options


model_dir = "/app/data/CommonNER/common.0.model"
dset_dir = "/app/data/CommonNER/common.dset"
data = ""


tornado.options.define("port", default=5006, help="变量保存端口，默认8000",type = int)

class MainHandler(tornado.web.RequestHandler):
    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin","*")
        self.set_header("Access-Control-Allow-Headers","*")
        self.set_header('Access-Control-Allow-Methods','POST,GET,OPTIONS')
        self.set_header("Content-Type","application/json;charset=utf-8")
       

    def get(self):
        self.write("Hello, world")

class ParseHandler(tornado.web.RequestHandler):
    def initialize(self, data):
        self.data = data
    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin","*")
        self.set_header("Access-Control-Allow-Headers","*")
        self.set_header('Access-Control-Allow-Methods','POST,GET,OPTIONS')
        self.set_header("Content-Type","application/json;charset=utf-8")
        

    def get(self):
        self.write("parse data")

    def post(self):
        getData = js.loads(self.request.body.decode('utf-8'))  
        sentence = getData["q"]
        
        global dset_dir
        global gpu
        global model_dir


        seg = False
        data = main.load_data_setting(dset_dir)
        data.generate_instance_with_gaz(sentence, 'sentence')
        decode_results = main.load_model_decode(model_dir, data, 'raw', gpu, seg)
        result = data.write_decoded_results_back(decode_results, 'raw')
        result_output=js.dumps(result)
        self.set_status(200)
        self.finish(result_output)

class trainHandler(tornado.web.RequestHandler):
    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin","*")
        self.set_header("Access-Control-Allow-Headers","*")
        self.set_header('Access-Control-Allow-Methods','POST,GET,OPTIONS')
        self.set_header("Content-Type","application/json;charset=utf-8")

    def get(self):
        self.write("train data")

    def post(self):
        self.set_status(200)
        self.write("train data")

def make_app():
    global data 
    return tornado.web.Application([
        (r"/", MainHandler),
        (r"/parse", ParseHandler,dict(data=data)),
        (r"/train", trainHandler)
    ])

def initialize():
    
    global dset_dir
    global data 
    global gpu
   # data = main.load_data_setting(dset_dir)
    gpu = torch.cuda.is_available()
    return

if __name__ == "__main__":
    print("model initialize... please wait")
    initialize()
    print("Success model initialize!")
    app = make_app()
    tornado.options.parse_command_line()
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(tornado.options.options.port)
    tornado.ioloop.IOLoop.current().start()