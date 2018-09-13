import eel
import cv2
from videoProcess import VideoProcess
from scoringAlgorithm import ScoringAlgorithm

@eel.expose
def setScore():
	vp = VideoProcess()
	vp.makeData()
	vp.img2gif()
	sa = ScoringAlgorithm('./results/target/targetData.txt', './results/webcam/webcamData.txt')
	score = sa.score()
	print (score)
	path_w = './web/score.txt'
	with open(path_w, mode='w') as f:
		f.write(str(score))
	eel.start("sample_3score.html")

eel.init("web")
eel.start("sample_1top.html")