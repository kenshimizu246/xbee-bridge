import os
import time
import signal
import logging
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor
from logging.handlers import RotatingFileHandler
import requests
from optparse import OptionParser

from PIL import Image
import cv2
import torch
from torch import nn
from torchvision import models, transforms
from time import sleep
import numpy as np

from digi.xbee.devices import XBeeDevice, ZigBeeDevice
from digi.xbee.devices import RemoteXBeeDevice, RemoteZigBeeDevice
from digi.xbee.devices import XBee64BitAddress
from digi.xbee.models.message import XBeeMessage
from digi.xbee.models.address import XBee16BitAddress, XBee64BitAddress
from digi.xbee.models.atcomm import ATCommandResponse, ATCommand, ATStringCommand
from digi.xbee.models.status import ATCommandStatus, DiscoveryStatus, TransmitStatus, ModemStatus, NetworkDiscoveryStatus
from digi.xbee.models.mode import OperatingMode
from digi.xbee.reader import DataReceived
from digi.xbee.packets.aft import ApiFrameType
from digi.xbee.packets.common import TransmitStatusPacket
from digi.xbee.packets.base import XBeeAPIPacket, DictKeys
from digi.xbee.exception import InvalidOperatingModeException, InvalidPacketException, TimeoutException, TransmitException
from digi.xbee.io import IOSample, IOLine
from digi.xbee.util import utils

"""
IN:
0:0 CMD_HELLO 0x01
1:- Text Message


0:0 CMD_WRITE_REQUEST = 0x02
1:4 Data Length

0:0 CMD_WRITE_DATA = 0x03
1:4 Sequence Number
5:- Data

0:0 CMD_WRITE_DONE = 0x04
1:4 Data Length
5:9 Sequence Number


OUT:
0:0 CMD_CONFIG = 0x11

0:0 CMD_RECV_STAT = 0x12

0:0 CMD_WRITE_ACK = 0x13

"""

CMD_HELLO = 0x01
CMD_WRITE_REQUEST = 0x02
CMD_WRITE_DATA = 0x03
CMD_WRITE_DONE = 0x04
CMD_GET_CONFIG = 0x05

CMD_CONFIG = 0x11
CMD_RECV_STAT = 0x12
CMD_WRITE_REQUEST_ACK = 0x13
CMD_WRITE_DATA_ACK = 0x14
CMD_WRITE_DONE_ACK = 0x15
CMD_WRITE_RESEND = 0x16
CMD_ARDUCAM_CMD = 0x17


config_path = "."
data_path = "."
log_path = "."
log_file = "{}/my_log.log".format(log_path)

xb = None
stop_flag = False
buffers = {}
seqs = {}
tpe = ThreadPoolExecutor(max_workers=5)

logger = logging.getLogger(__name__)

def write_log(msg):
    logger.info(msg)
    print(msg)

def post_django_file(address, fileName):
    global url
    logger.info("start post_django: {}".format(address))
    img_org = cv2.imread(fileName)
    logger.info("post_django fileName: {} {}".format(fileName, str(img_org)))
    cv2.imwrite(fileName, img_org)
    logger.info("post_django wrote fileName: {}".format(fileName))
    files = { 'file_uploaded': (fileName, open(fileName, 'rb'), 'image/jpeg') }
    logger.info("url:{}, filename:{}".format(url, fileName))
    response = requests.post(url, files=files)
    logger.info("post response:{}".format(response))

def post_django(address, data):
    global url
    try:
        now = datetime.now()
        write_log("start post_django: {} {} {}".format(address, len(data), os.getcwd()))
        jpg_as_np = np.frombuffer(bytes(data), dtype=np.uint8)
        write_log("post_django2: {}".format(jpg_as_np))
        img_org = cv2.imdecode(jpg_as_np, flags=1)
        write_log("post_django3")
        fileName = "{}_{}.jpg".format(address, now.strftime("%m%d%Y_%H%M%S"))
        if(os.path.exists("data")):
            fileName = "data/{}".format(fileName)
        write_log("post_django fileName: {} {}".format(fileName, str(img_org)))
        # imwrite returns bool
        ret = cv2.imwrite(fileName, img_org)
        write_log("post_django wrote fileName: {} {}".format(fileName, ret))
        files = { 'file_uploaded': (fileName, open(fileName, 'rb'), 'image/jpeg') }
        write_log("url:{}, filename:{}".format(url, fileName))
        response = requests.post(url, files=files)
        write_log("post response:{}".format(response))
    except Exception as e:
        write_log("Exception:{}".format(e))

def send_missing(addr, missing):
    global xb, config_path

    try:
        ss = len(missing)
        mm = CMD_RECV_STAT.to_bytes(1, 'big') + ss.to_bytes(4, 'big')
        for i in missing:
            mm = mm + i.to_bytes(4, 'big')
            print("missing: {}".format(i)) 
            logger.info("missing: {}".format(i)) 

        # st = xb.send_data(xbee_message.remote_device, mm)
        st = xb.send_data(addr, mm)
    except TransmitException as e:
        print("xb.send_missing: {}".format(e)) 
        logger.info("xb.send_missing: {}".format(e)) 

def send_write_request_ack(addr, req_id, len):
    global xb

    try:
        write_log("start send_write_request_ack:[addr:{}][req_id:{}][len:{}][type:{}]".format(addr, req_id, len, req_id)) 
        mm = CMD_WRITE_REQUEST_ACK.to_bytes(1, 'big') + req_id.to_bytes(1,byteorder='big') + len.to_bytes(4, 'big')
        st = xb.send_data(addr, mm)
        write_log("end send_write_request_ack ok:[{}][{}][{}]".format(addr, len, st.transmit_status)) 
    except TransmitException as e:
        write_log("send_write_request_ack error: {}".format(e)) 
    except Exception as e:
        write_log("send_write_request_ack error: {}".format(e)) 

def send_write_data_ack(addr, req_id, seq):
    global xb

    try:
        write_log("start send_write_data_ack: {} {} {}".format(addr, req_id, seq)) 
        mm = CMD_WRITE_DATA_ACK.to_bytes(1, 'big') + req_id.to_bytes(1,byteorder='big') + seq.to_bytes(4, 'big')
        st = xb.send_data(addr, mm) # returns digi.xbee.packets.common.TransmitStatusPacket
        write_log("end send_write_data_ack ok:[{}][{}][{}][{}]".format(addr, req_id, seq, st.transmit_status)) 
    except TransmitException as e:
        write_log("send_write_data_ack error: {}".format(e)) 
    except Exception as e:
        write_log("send_write_data_ack error: {}".format(e)) 

def send_write_done_ack(addr, req_id, ln, seq):
    global xb

    try:
        write_log("start send_write_done_ack: {} {} {} {}".format(addr, req_id, ln, seq)) 
        mm = CMD_WRITE_DONE_ACK.to_bytes(1, 'big') + req_id.to_bytes(1,byteorder='big') + ln.to_bytes(4, 'big') + seq.to_bytes(4, 'big')
        st = xb.send_data(addr, mm)
        write_log("end send_write_done_ack ok:[{}][{}]".format(addr, st.transmit_status)) 
    except TransmitException as e:
        write_log("send_write_done_ack error: {}".format(e)) 
    except Exception as e:
        write_log("send_write_done_ack error: {}".format(e)) 

def my_data_received_callback(xbee_message):
    global xb, config_path

    try:
        remote_dev = xbee_message.remote_device
        address = xbee_message.remote_device.get_64bit_addr()
        cmd = xbee_message.data[0]
        data = xbee_message.data

        write_log("Command from {} [cmd:{}][{}]".format(address, cmd, type(xbee_message)))
        write_log("data:{}".format(data))
        if(cmd == CMD_HELLO):
            logger.info("Hello from {} [{}]".format(address, data[1:]))
            mm = CMD_HELLO.to_bytes(1, 'big')
            logger.info("Hello reply:{}:{}".format(len(mm), mm))
            st = xb.send_data(xbee_message.remote_device, mm)
            logger.info("reply status:{}".format(st))
        elif(cmd == CMD_GET_CONFIG):
            logger.info("Get config from %s: %d" % (address, cmd))
            mm = CMD_CONFIG.to_bytes(1, 'big')
            l = 0
            d = None
            with open("{}/{}.cfg".format(config_path, address)) as f:
                s = f.read()
                if(len(s) > 0):
                    d = bytearray(s.encode())
                    l = len(d)
            if(l > 0):
                mm = mm + l.to_bytes(4, 'big') + d
            else:
                mm = mm + l.to_bytes(4, 'big')
            logger.info("send config:{}:{}".format(len(mm), mm))
            st = xb.send_data(xbee_message.remote_device, mm)
            logger.info("reply status:{}".format(st))
        elif(cmd == CMD_WRITE_REQUEST):
            req_id = data[1]
            data_len = int.from_bytes(data[2:6], byteorder='big')
            packet_cnt = int.from_bytes(data[6:10], byteorder='big')
            buffers[address] = None
            seqs[address] = []
            write_log("Data Req from {} [reqid:{}][len:{}][pkt:{}]".format(address, req_id, data_len, packet_cnt))
            tpe.submit(send_write_request_ack, remote_dev, req_id, data_len)
        elif(cmd == CMD_WRITE_DATA):
            if(address in buffers):
                req_id = data[1]
                seq = int.from_bytes(data[2:6], byteorder='big')
                write_log("Write from 1 {} [req_id:{}][seq:{}][len:{}]".format(address, req_id, seq, len(data)))
                if(address not in seqs or seqs[address] is None):
                    seqs[address] = []
                if(seq not in seqs[address]):
                    seqs[address].append(seq)
                    if(buffers[address] is None):
                        buffers[address] = {}
                    buffers[address][seq] = bytearray(data[6:])
                else:
                    write_log("Not Write from {} [req_id:{}][seq:{}][len:{}][{}]".format(address, req_id, seq, len(data), len(buffers[address])))
                write_log("Write from 3 {} [req_id:{}][seq:{}][len:{}][{}]".format(address, req_id, seq, len(data), len(buffers[address])))
                tpe.submit(send_write_data_ack, remote_dev, req_id, seq)
        elif(cmd == CMD_WRITE_DONE):
            req_id = data[1]
            ln  = int.from_bytes(data[2:6], byteorder='big')
            pktcnt = int.from_bytes(data[6:10], byteorder='big')
            if(address in buffers and address in seqs):
                write_log("Done from {} [ln:{}][seq:{}][buff:{}]".format(address, ln, pktcnt, len(buffers[address])))
                if(len(buffers[address]) == pktcnt):
                    write_log("submit post_django: {} {}".format(address, len(buffers[address]))) 
                    data = None
                    seqs[address].sort()
                    for s in seqs[address]:
                        if(data == None):
                            data = buffers[address][s]
                        else:
                            data = data + buffers[address][s]
                    tpe.submit(post_django, address, data)
                else:
                    missing = []
                    for i in range(0, pktcnt):
                        if(i not in seqs):
                            missing.append(i)
                tpe.submit(send_write_done_ack, remote_dev, req_id, ln, pktcnt)
            buffers[address] = None
            seqs[address] = []
            print("Done from %s: %d %d %d %d" % (address, cmd, req_id, ln, pktcnt))
        else:
            print("Received data from %s: %d" % (address, cmd))
    except TransmitException as e:
        write_log("my_data_received_callback error: {}".format(e)) 
    except Exception as e:
        write_log("my_data_received_callback error: {}".format(e)) 

def stop_handler(signum, frame):
    logger.info("signum:{}".format(signum))
    global stop_flag
    stop_flag = True

def main():
    global url,xb
    global config_path, data_path, log_path, log_file

    parser = OptionParser()
    parser.add_option("-z", "--zigbee", dest="zigbee",
                      help="Zig Bee Device path", default='/dev/ttyUSB0')
    parser.add_option("-b", "--baudrate", dest="baudrate",
                      help="Zig Bee Baud Rate", default=115200)
    parser.add_option("-a", "--path", dest="path",
                      help="Application Path", default='.')
    parser.add_option("-d", "--dest", dest="dest",
                      help="Destination URL", default='http://localhost:8000/facedetect/')
    parser.add_option("-f", "--file", dest="file",
                      help="Zig Bee Device Test File", default=None)

    (options, args) = parser.parse_args()

    config_path = "{}/config".format(options.path)
    data_path = "{}/data".format(options.path)
    log_path = "{}/logs".format(options.path)
    log_file = "{}/my_log.log".format(log_path)

    logger.setLevel(logging.DEBUG)
    handler = RotatingFileHandler(log_file, maxBytes=1000000, backupCount=10)
    formatter = logging.Formatter('%(asctime)s %(levelname) -8s :%(lineno) -3s %(funcName)s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info("config_path:{}".format(config_path))
    logger.info("data_path:{}".format(data_path))
    logger.info("log_path:{}".format(log_path))
    logger.info("log_file:{}".format(log_file))

    url = options.dest
    logger.info("url:{}".format(url))

    if(options.file):
        post_django_file(options.file, options.file)
        return

    logger.info("add SIGINT handler")
    signal.signal(signal.SIGINT, stop_handler)

    logger.info("ZigBeeDevice {} {}".format(options.zigbee, options.baudrate))
    xb = ZigBeeDevice(options.zigbee, options.baudrate)

    logger.info("ZigBeeDevice.optn()")
    xb.open()

    logger.info("add xbee handler.")
    xb.add_data_received_callback(my_data_received_callback)

    logger.info("start loop.")
    cnt = 0
    while(not stop_flag):
        cnt = cnt + 1
        if((cnt % 60) == 0):
            logger.info("sleepping.")
        time.sleep(1)

    logger.info("remove xbee handler.")
    xb.del_data_received_callback(my_data_received_callback)

if __name__ == "__main__":
    main()

