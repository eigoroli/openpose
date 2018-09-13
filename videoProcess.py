import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from PIL import ImageTk, Image, ImageDraw

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
from tf_pose import common

import ffmpeg
import os
import re
import glob

TEACHER_FILE_NAME = "./etcs/teacher.avi"
# WEBCAM_FILE_NAME = "./etcs/webcam.avi"
WEBCAM_FILE_NAME = "/Users/eigo/Downloads/recorded.mp4"
WEBCAM_WEBM_FILE_NAME = "/Users/eigo/Downloads/recorded.webm"
WIDTH = 432
HEIGHT = 368
MAXSIZE = (WIDTH, HEIGHT)
FRAME_RATE = 10

class VideoProcess():
	def __init__(self):
		stream = ffmpeg.input(WEBCAM_WEBM_FILE_NAME)
		stream = ffmpeg.hflip(stream)
		stream = ffmpeg.output(stream, WEBCAM_WEBM_FILE_NAME.replace('.webm', '.mp4'))
		ffmpeg.run(stream)
		os.remove(WEBCAM_WEBM_FILE_NAME)

	def makeData(self):
		w, h = model_wh('432x368')
		e = TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(w, h))

		i = 0
		cap = cv2.VideoCapture(WEBCAM_FILE_NAME)
		while cap.isOpened():
			endFlag, frame = cap.read()
			if endFlag == False:
				break
			cv2.imwrite('./results/webcam/webcamRaw/'+'img_%s.png' % str(i).zfill(6), frame)
			i += 1

		j = 0
		path_w = './results/webcam/webcamData.txt'
		with open(path_w, mode='w') as f:
			f.write('')
		while j < i:
			image = common.read_imgfile('./results/webcam/webcamRaw/'+'img_%s.png' % str(j).zfill(6))
			humans = e.inference(image, resize_to_default=True, upsample_size=4.0)

			centers = TfPoseEstimator.get_centers(image, humans, imgcopy=False)
			
			with open(path_w, mode='a') as f:
				f.write('t' + str(j) + ':' + str(centers) + '\n')
			j += 1

		k = 30
		while k < 80:
			image = common.read_imgfile('./results/webcam/webcamRaw/'+'img_%s.png' % str(k).zfill(6))
			humans = e.inference(image, resize_to_default=True, upsample_size=4.0)
			image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
			cv2.imwrite('./results/webcam2/webcamOpenpose/'+'img_%s.png' % str(k).zfill(6), image)
			k += 1

	def img2gif(self):
		files = sorted(glob.glob('./results/webcam/webcamOpenpose/*.png'))
		images = list(map(lambda file: Image.open(file), files))
		images[0].save('./web/webcamOpenpose.gif', save_all=True, append_images=images[1:], duration=400, loop=0)