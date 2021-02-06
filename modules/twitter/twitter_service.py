import os
import re
import sys
import random
import glob
import json
import time
import cv2
import requests
from datetime import datetime
from time import gmtime, strftime
from elasticsearch import Elasticsearch
from requests.packages.urllib3.exceptions import InsecureRequestWarning
from requests_oauthlib import OAuth1
import dlib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import tensorflow.compat.v1 as tf
import numpy as np
from PIL import Image

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from imutils import face_utils
import imutils



requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
label_map = label_map_util.load_labelmap('v3/label_map.pbtxt')
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=14, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
face_d = dlib.get_frontal_face_detector()
face_r = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
shape_p = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_a = face_utils.facealigner.FaceAligner(shape_p, desiredFaceWidth=256, desiredLeftEye=(0.3, 0.3))

class TwitterFetchMedia:
	
	username = ""
	password = ""
	timer = 60
	mtype= "image"
	creds = {"oauth_token": "", "oauth_token_secret": "", "user_id": 0, "screen_name": "", "x_auth_expires": 0, "kdt": ""}
	es= None
	keywords = []
	tfgraph = None
	tfses = None
	APO = np.array([-0.12799129,0.0756199,0.02044699,-0.12095182,-0.04744143,-0.0898906,-0.01964812,-0.04443996,0.1404182,0.02220949,0.220759,0.01688146,-0.21987839,-0.02875833,-0.0382871,0.13304153,-0.13912846,-0.045706,-0.10777017,-0.07266846,0.05177985,0.05861961,0.08564904,-0.02859904,-0.09763953,-0.2993809,-0.11898426,-0.07692917,0.05785508,-0.08346874,0.07546067,0.00194696,-0.23782521,-0.06124029,0.01920654,-0.01593454,-0.13571286,-0.11797427,0.16713238,0.0144488,-0.12608257,0.06086428,0.08945096,0.21812554,0.20854723,0.0593489,0.03392382,-0.11947554,0.06276168,-0.23325357,0.13593933,0.1023512,0.08876117,0.07015303,0.15162647,-0.07746953,0.03981521,0.09219455,-0.16384746,0.02380281,0.09786145,0.05800536,-0.02771912,-0.12421358,0.15211636,0.07370955,-0.1467538,-0.08919812,0.08749228,-0.11598724,-0.02384263,0.05680263,-0.09418385,-0.16527587,-0.31112176,0.12066667,0.2801733,0.12164746,-0.09984476,-0.00144713,-0.0529428,-0.01681647,-0.0143257,-0.00937086,-0.12916265,-0.11799618,-0.08088546,0.00302109,0.17138991,-0.05688579,-0.04235391,0.1834878,-0.02102136,-0.04305467,-0.04905874,0.10848202,-0.1782188,-0.02229053,-0.05838548,0.01148089,0.05452568,-0.08194978,-0.04750768,0.17383575,-0.12151044,0.11273662,-0.03532398,-0.06354972,-0.00547409,0.03151039,-0.11140285,-0.09069368,0.02180602,-0.17530705,0.24034318,0.14381084,-0.00799688,0.07579614,0.07254957,0.09682716,-0.05786604,0.08366568,-0.10809348,-0.10781497,0.04088486,0.06499089,0.01706926,0.03314217])
	def __init__(self, username="",password="",keywords=[], timer=60,mtype="image",tfgraph=None,tfses=None,esip='localhost'):
		self.username = username
		self.password = password
		if len(username) > 0:
			self.creds = self.login()
		else:
			self.creds = self.getCreds()
		self.timer = timer
		self.mtype = mtype
		self.keywords = keywords
		self.tfgraph = tfgraph
		self.tfses = tfses
		self.es = Elasticsearch(host=esip,port=9200,timeout=30, max_retries=10, retry_on_timeout=True)
		if not os.path.isfile(".calisti"):
			self.ilkcalisma()
	def ilkcalisma(self):
		
		if  not os.path.exists('creds'):
			os.mkdir('creds')
		if  not os.path.exists('files'):
			os.mkdir('files')
		if  not os.path.exists('tespit'):
			os.mkdir('tespit')
		if  not os.path.exists('tweets'):
			os.mkdir('tweets')	
		with open(".calisti","a") as f:
			f.write("-")
		
	def yuzleriBul(self,face, s=1):
		return face_d(face, s)
	def resimHesapla(self,face,byuzler):
		yuzler = [shape_p(face, yuz) for yuz in byuzler]
		return [np.array(face_r.compute_face_descriptor(face, yuzpoz, 1)) for yuzpoz in yuzler]
	def benzerlik(self,face1, face2,xy=1):
		if len(face1) == 0:
			return np.empty((0))
		return np.linalg.norm(face1 - face2, axis=1)
	def xstart(self):
		try:
			say = 0
			while(True):
				for keyword in self.keywords:
					tweets = self.searchTweets(str(keyword))
					try:
						for twid in tweets["globalObjects"]["tweets"]:
							tweet = tweets["globalObjects"]["tweets"][twid]
							"""
							with open("tweets/"+str(twid)+".tweet", "w") as f:
								f.write(json.dumps(tweet))
							"""
							if self.checkfromid(tweet["id_str"]) == False:
								
							
									self.inserttweet(tweet)
									self.saveMedia(tweet)
									say += 1
									print(str(tweet["id_str"]),"id ile 1 tweet kayıt altına alındı, toplam", str(say), "tweet kayıt edildi.")
						time.sleep(int(self.timer))
					except Exception as ex2:
						print("xstart", ex2
		except Exception as e:
			print("xstart", e)
			sys.exit(0)
	def checkfromid(self,twid, real=False):
		try:
			page = self.es.get(index = 'murateren_twitter' ,id=str(twid))
			if real == True:
				return page["_source"]
		
			return page["found"]		
		except Exception as e:
			return False
			pass
			
	def inserttweet(self,tweet):
		try:
			tweet["insertdate"] = strftime("%Y-%m-%d %H:%M:%S", gmtime()) 
			for fix in range(0,len(tweet["extended_entities"]["media"])):
				del tweet["extended_entities"]["media"][fix]["ext"]
				res = self.es.index(index="murateren_twitter", id=str(tweet["id_str"]), body=tweet)
		except:
			pass
	def searchTweets(self,keyword,cursor="-1"):
		try:
			xtoken,xsecret = self.creds["oauth_token"],self.creds["oauth_token_secret"]
			time.sleep(2)
			return requests.get("https://api.twitter.com/2/search/adaptive.json?spelling_corrections=true&tweet_search_mode=live&q="+str(keyword)+"&query_source=typed_query&result_filter="+str(self.mtype)+"&earned=true&include_entities=true&include_cards=true&cards_platform=iOS-13&include_carousels=true&ext=mediaRestrictions%2CaltText%2CmediaStats%2CmediaColor%2Cinfo360%2ChighlightedLabel&include_media_features=true&include_blocking=true&include_blocked_by=true&include_quote_count=true&tweet_mode=extended&include_composer_source=true&include_ext_media_availability=true&include_reply_count=true&simple_quoted_tweet=true&include_user_entities=true&include_profile_interstitial_type=true&count=200",headers={"User-Agent": "Twitter-iPhone/9.62 iOS/13.3.3 (Apple;iPhone9,1;;;;;1)","Content-Type": "application/json","x-twitter-client-language": "tr"},auth=OAuth1('3nVuSoBZnx6U4vzUxf5w','Bcs59EFbbsdF6Sl9Ng71smgStWEGwXXKSjYvPVt7qys',xtoken,xsecret,decoding=None), verify=False).json()
		except 	Exception as e:
			print("searchTweets error")
			pass
	def getCreds(self):
		creds = glob.glob('creds/*.creds')
		if len(creds) < 1:
			return None
		with open(random.choice(creds)) as rf:
			data = json.load(rf)
			return data
	def login(self):
		try:
			t 		= str(self.getToken())
			data 	= {'x_auth_identifier': self.username,'x_auth_password': self.password,'send_error_codes':'true','x_auth_login_verification': '1','x_auth_login_challenge': '1','x_auth_country_code': 'TR'}
			headers	={'User-Agent': 'Twitter-iPhone/9.62 iOS/13.3.3 (Apple;iPhone9,1;;;;;1)','X-Guest-Token': t,'Authorization': 'Bearer AAAAAAAAAAAAAAAAAAAAAFXzAwAAAAAAMHCxpeSDG1gLNLghVe8d74hl6k4%3DRUMF4xAQLsbeBhTSRrCiQpJtxoGWeyHrDb5te2jpGskWDFW82F'}
			response = requests.post("https://api.twitter.com/auth/1/xauth_password.json", data=data, headers=headers, verify=False).json()
			self.saveCreds(response)
			return response
		except Exception as e:
			print("Login func",e)
			return "false"
	def getToken(self):
		page = ''
		while page == '':
			try:
				headers = {'User-Agent': 'Twitter-iPhone/9.62 iOS/13.3.3 (Apple;iPhone9,1;;;;;1)','Authorization': 'Bearer AAAAAAAAAAAAAAAAAAAAAFXzAwAAAAAAMHCxpeSDG1gLNLghVe8d74hl6k4%3DRUMF4xAQLsbeBhTSRrCiQpJtxoGWeyHrDb5te2jpGskWDFW82F'}
				return requests.post("https://api.twitter.com/1.1/guest/activate.json", headers=headers, verify=False).json()['guest_token']
			except Exception as e:
				time.sleep(5)
				continue
	def saveMedia(self,tw):
		try:
			twid = tw["id_str"]
			art = 0
			for media in tw["extended_entities"]["media"]:
				art += 1 
				mediaid = media["id_str"]
				if media["type"] == "photo":
					self.saveFromURL(media["media_url_https"],str(twid)+"_"+str(mediaid)+"_"+str(art)+".jpg")
					self.test_image("files/"+str(twid)+"_"+str(mediaid)+"_"+str(art)+".jpg",art,str(twid),str(mediaid))
				if media["type"] == "video":
					variants = {}
					for media in tw["extended_entities"]["media"]:
						for v in media["video_info"]["variants"]:
							if 'bitrate' in v:
								variants[str(v["bitrate"])] = v["url"]
					xvar = sorted(variants.items())
					self.saveFromURL(xvar[0][1],str(twid)+"_"+str(mediaid)+".mp4")
		except Exception as e:
			print("saveImage",e)
			pass
	def saveFromURL(self,url,mid):
		try:
			req = requests.get(url,stream=True)
			with open('files/'+str(mid), 'wb') as f:
				for chunk in req.iter_content(chunk_size=1024):
					if chunk:
						f.write(chunk)
			return True
		except Exception as e:
			print("saveFromURL func",e)
			pass
	def test_image(self,path,art,twid,mid):
		try:
			im = cv2.imread(path)
			#print(path)
			im = imutils.resize(im, width=640)
			im_dims = np.expand_dims(im, axis=0)
			image_tensor = self.tfgraph.get_tensor_by_name('image_tensor:0')
			boxes = self.tfgraph.get_tensor_by_name('detection_boxes:0')
			scores = self.tfgraph.get_tensor_by_name('detection_scores:0')
			classes = self.tfgraph.get_tensor_by_name('detection_classes:0')
			num_detections = self.tfgraph.get_tensor_by_name('num_detections:0')
			(boxes, scores, classes, num_detections) = self.tfses.run([boxes, scores, classes, num_detections],feed_dict={image_tensor: im_dims})
			# min_score_thresh degerini degistirerek tespit oranlarını degistirebilirsiniz. suan >%50 
			
			vis_util.visualize_boxes_and_labels_on_image_array(im,np.squeeze(boxes),np.squeeze(classes).astype(np.int32), np.squeeze(scores),category_index,use_normalized_coordinates=True,max_boxes_to_draw=20,min_score_thresh=.6,line_thickness=4)
			(height, width) = im.shape[:2]
			for i,b in enumerate(boxes[0]):
				for s in range(14):
					if classes[0][i] == s: 
						if scores[0][i] >= 0.60:
							#print(i,twid,"tespit")
							ymin = int((np.squeeze(boxes)[i][0]*height))# left
							xmin = int((np.squeeze(boxes)[i][1]*width)) # top
							ymax = int((np.squeeze(boxes)[i][2]*height)) # right
							xmax = int((np.squeeze(boxes)[i][3]*width)) # bottom
							cropped_img = im[ymin:ymax,xmin:xmax]
							label = (category_index[s].get('name'))
							if label == "APO":	
								testImage = imutils.resize(cropped_img, width=256) 
								faces = self.resimHesapla(testImage,self.yuzleriBul(testImage))
								for face in faces:
									sonuc = 100 - int(str(self.benzerlik(np.array(face),[self.APO]))[3:5])   
									if sonuc > 50:
										cv2.imwrite("tespit/"+str(twid)+"_"+str(mid)+"_"+str(art)+"_"+str(label)+".jpg", cropped_img)
							else:
								cv2.imwrite("tespit/"+str(twid)+"_"+str(mid)+"_"+str(art)+"_"+str(label)+"_"+str(random.randint(100,999))+".jpg", im)
		except Exception as e:
			print("test image",e)
			pass
	def saveCreds(self,j):
		try:
			with open("creds/"+str(j["user_id"])+".creds", "w") as f:
				f.write(json.dumps(j))
		except Exception as e:
			print("saveCreds func", e)
			pass
detection_graph = tf.Graph()
with detection_graph.as_default():
	od_graph_def = tf.GraphDef()
	with tf.gfile.GFile('v3/197073.pb', 'rb') as fid:
		serialized_graph = fid.read()
		od_graph_def.ParseFromString(serialized_graph)
		tf.import_graph_def(od_graph_def, name='')
with detection_graph.as_default():
	with tf.Session(graph=detection_graph) as sess:
		twservice = TwitterFetchMedia(keywords=["pkk"],tfses=sess,tfgraph=detection_graph,mtype="image")
		twservice.xstart()