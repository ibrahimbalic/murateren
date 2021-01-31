import sys,os,cv2
import imutils
import tensorflow as tf
import tensorflow.compat.v1 as tf
import numpy as np
from PIL import Image
from distutils.version import StrictVersion
from collections import defaultdict
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

#https://github.com/tensorflow/models/tree/master/research/object_detection
def resize_aspect_fit(im):
	#https://stackoverflow.com/questions/21517879/python-pil-resize-all-images-in-a-folder/50546731#50546731
	img = Image.fromarray(im)
	size = img.size
	ratio = float(640) / max(size)
	new_image_size = tuple([int(x*ratio) for x in size])
	img = img.resize(new_image_size, Image.ANTIALIAS)
	new_im = Image.new("RGB", (640, 640))
	new_im.paste(img, ((640-new_image_size[0])//2, (640-new_image_size[1])//2))
	return np.asarray(new_im)


# gpu veya cpu belirleyin. once sess.list_devices() ile uygun olanlara bakabilirsiniz.
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
title = "MURAT_EREN"

label_map = label_map_util.load_labelmap('v3/label_map.pbtxt')
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=14, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

detection_graph = tf.Graph()
with detection_graph.as_default():
	od_graph_def = tf.GraphDef()
	with tf.gfile.GFile('v3/201725.pb', 'rb') as fid:
		serialized_graph = fid.read()
		od_graph_def.ParseFromString(serialized_graph)
		tf.import_graph_def(od_graph_def, name='')

cap 	= cv2.VideoCapture(str(sys.argv[1]))
xwidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
xheight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

codec 	= cv2.VideoWriter_fourcc(*'mp4v')
out 	= cv2.VideoWriter(str(sys.argv[1])+"_tespit.mp4",  codec ,20, (int(xwidth), int(xheight)))

with detection_graph.as_default():
	with tf.Session(graph=detection_graph) as sess:
		ret = True
		while (ret):
			ret, im = cap.read()
			#im = resize_aspect_fit(im)
			im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
			im_dims = np.expand_dims(im, axis=0)
			image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
			boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
			scores = detection_graph.get_tensor_by_name('detection_scores:0')
			classes = detection_graph.get_tensor_by_name('detection_classes:0')
			num_detections = detection_graph.get_tensor_by_name('num_detections:0')
			(boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections],feed_dict={image_tensor: im_dims})
			vis_util.visualize_boxes_and_labels_on_image_array(im,np.squeeze(boxes),np.squeeze(classes).astype(np.int32), np.squeeze(scores),category_index,use_normalized_coordinates=True,max_boxes_to_draw=20,min_score_thresh=.5,line_thickness=4)
			out.write(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
			
			# izlemek icin
			cv2.imshow(title, cv2.cvtColor(cv2.resize(im,(int(xwidth), int(xheight))), cv2.COLOR_BGR2RGB))
			
			# resim olarak tespit tespit cikti almak icin kullanabilirsiniz
			"""
			for i,b in enumerate(boxes[0]):
				for s in range(14):
					if classes[0][i] == s: 
						if scores[0][i] >= 0.50:
							art2 += 1 
							label = (category_index[s].get('name'))
							cv2.imwrite(str(art)+"_"+str(art2)+"_"+str(label)+".jpg", cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
			if cv2.waitKey(1) & 0xFF == ord('p'):
				cv2.waitKey(-1)	
			"""
			if cv2.waitKey(1) & 0xFF == ord('q'):
				cv2.destroyAllWindows()
				cap.release()
				out.release()
				break
cv2.destroyAllWindows()
cap.release()
out.release()