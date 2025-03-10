# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import os
import sys
import argparse
import csv

wrapper_lib_path = os.getenv('CYBER_WRAPPER_LIB_PATH')
if not wrapper_lib_path:
    wrapper_lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../python_wrapper'))
sys.path.append(wrapper_lib_path)

python_proto_path = os.getenv('PY_PROTO_PATH')
if not python_proto_path:
    python_proto_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../python_wrapper'))
sys.path.append(python_proto_path)

from python_wrapper import record
from car_state_pb2 import CarState
from control_command_pb2 import ControlCommand

channel_list = ["/canbus/car_state", "/control/control_command"]
t = 0
parser = argparse.ArgumentParser()
parser.add_argument('--input', default="/home/nio/Downloads/20210820111715.record.00000", help='file to load')
parser.add_argument('--output_path', default="/opt/nio/Control_Csv/", help='output csv')


def record_converter(reader_path, file_path, file_name):
    freader = record.RecordReader(reader_path)
    time.sleep(1)
    print('Begin to convert')
    for channel_name, msg, datatype, timestamp in freader.read_messages():
        if channel_name in channel_list:
            proto_msg = eval(datatype.split('.')[-1])()
            proto_msg.ParseFromString(msg)
            if channel_name == "/canbus/car_state":
                t = proto_msg.time_meas
            elif channel_name == "/control/control_command":
                t = proto_msg.time_meas
                data = proto_msg
                data_control = [t, data.angle, data.front_wheel_angle_rate, data.acceleration, data.turn_signal,
                                data.driving_mode, data.simple_debug.lateral_error, data.simple_debug.ref_heading,
                                data.simple_debug.heading, data.simple_debug.heading_error,
                                data.simple_debug.heading_error_rate, data.simple_debug.lateral_error_rate,
                                data.simple_debug.curvature, data.simple_debug.steer_angle,
                                data.simple_debug.steer_angle_feedforward,
                                data.simple_debug.steer_angle_lateral_contribution,
                                data.simple_debug.steer_angle_lateral_rate_contribution,
                                data.simple_debug.steer_angle_heading_contribution,
                                data.simple_debug.steer_angle_heading_rate_contribution,
                                data.simple_debug.steer_angle_feedback, data.simple_debug.steering_position,
                                data.simple_debug.steer_angle_limited, data.simple_debug.speed_reference,
                                data.simple_debug.speed_error, data.simple_debug.acceleration_reference,
                                data.simple_debug.speed, data.simple_debug.acceleration_feedback,
                                data.simple_debug.acceleration_error, data.simple_debug.lon_position_feedback,
                                data.simple_debug.cur_lon_position_reference, data.simple_debug.cur_lon_position_error,
                                data.simple_debug.pre_lon_position_error, data.simple_debug.pre_speed_error,
                                data.simple_debug.speed_cmd_limit, data.simple_debug.pre_speed_reference,
                                data.simple_debug.pre_acceleration_reference,
                                data.simple_debug.pre_lon_position_reference, data.simple_debug.lon_speed_feedforward,
                                data.simple_debug.pre_acceleration_error, data.simple_debug.ref_pos_x,
                                data.simple_debug.ref_pos_y, data.simple_debug.raw_curvature,
                                data.simple_debug.raw_lateral_error, data.simple_debug.raw_lateral_error_rate,
                                data.simple_debug.raw_heading_error, data.simple_debug.raw_heading_error_rate,
                                data.simple_debug.cur_pos_x, data.simple_debug.cur_pos_y]
                write_csv(data_control, file_path, file_name)
    print("Convert finished")
    print(file_name + " is created in " + file_path)


def write_csv(data_row, file_path, file_name):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    if not os.path.exists(file_path + file_name):
        with open(file_path + file_name, 'w') as f:
            csv_write = csv.writer(f)
            csv_head = ["time", "angle", "front_wheel_angle_rate", "acceleration", "turn_signal",
                        "driving_mode", "lateral_error", "ref_heading", "heading", "heading_error",
                        "heading_error_rate", "lateral_error_rate", "curvature", "steer_angle",
                        "steer_angle_feedforward", "steer_angle_lateral_contribution",
                        "steer_angle_lateral_rate_contribution", "steer_angle_heading_contribution",
                        "steer_angle_heading_rate_contribution", "steer_angle_feedback", "steering_position",
                        "steer_angle_limited", "speed_reference", "speed_error", "acceleration_reference", "speed",
                        "acceleration_feedback", "acceleration_error", "lon_position_feedback",
                        "cur_lon_position_reference", "cur_lon_position_error", "pre_lon_position_error",
                        "pre_speed_error", "speed_cmd_limit", "pre_speed_reference", "pre_acceleration_reference",
                        "pre_lon_position_reference", "lon_speed_feedforward", "pre_acceleration_error", "ref_pos_x",
                        "ref_pos_y", "raw_curvature", "raw_lateral_error", "raw_lateral_error_rate",
                        "raw_heading_error", "raw_heading_error_rate", "cur_pos_x", "cur_pos_y"]
            csv_write.writerow(csv_head)
        return

    with open(file_path + file_name, 'a+') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(data_row)


if __name__ == '__main__':
    curtime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    file_name = '%s.csv' % curtime
    file_path = parser.parse_args().output_path
    record_file = parser.parse_args().input
    print('Begin to read record file: {}'.format(record_file))
    record_converter(record_file, file_path, file_name)



