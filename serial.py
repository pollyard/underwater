import cv2
import mediapipe as mp
import requests
import time
import threading
import time


mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils


# 웹으로 전송할 URL과 매개변수 설정
url = "http://10.150.151.30:8090/yuchan/data2.jsp"
params1 = {'people': '1'}
params2 = {'people': '0'}
people_test = params2
def start_thread():
    # 년월일 쓰레드 생성
    thread = threading.Thread(target=start_Data)
    # thead 사용할 함수를 지정한다
    thread.daemon = True
    # thread 사용을 허용 해준다
    thread.start()
# 이미지 파일의 경우 이것을 사용하세요:
def start_Data():
    while True:
        response = requests.post(url=url, data=people_test)
        time.sleep(10)

IMAGE_FILES = []
start_thread()
with mp_face_detection.FaceDetection(
    model_selection=1, min_detection_confidence=0.5) as face_detection:
  for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)
    # 작업 전에 BGR 이미지를 RGB로 변환합니다.
    results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # 이미지를 출력하고 그 위에 얼굴 박스를 그립니다.
    if not results.detections:
      print('0')
      continue
    annotated_image = image.copy()
    for detection in results.detections:
      print('Nose tip:')
      print(mp_face_detection.get_key_point(
          detection, mp_face_detection.FaceKeyPoint.NOSE_TIP))
      mp_drawing.draw_detection(annotated_image, detection)

    # 웹으로 전송
    #response = requests.post(url=url, data=params1)
    #print(response.text)

    cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)

# 웹캠, 영상 파일의 경우 이것을 사용하세요.:
cap = cv2.VideoCapture(0)
with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5) as face_detection:
  while cap.isOpened():
    success, image = cap.read()
    
    if not success:
      print("웹캠을 찾을 수 없습니다.")
      # 비디오 파일의 경우 'continue'를 사용하시고, 웹캠에 경우에는 'break'를 사용하세요.
      continue
    # 보기 편하기 위해 이미지를 좌우를 반전하고, BGR 이미지를 RGB로 변환합니다.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # 성능을 향상시키려면 이미지를 작성 여부를 False으로 설정하세요.
    image.flags.writeable = False
    results = face_detection.process(image)

    # 영상에 얼굴 감지 주석 그리기 기본값 : True.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.detections:
      for detection in results.detections:
        mp_drawing.draw_detection(image, detection)

      # 웹으로 전송
      #response = requests.post(url=url, data=params1)
      #print(response.text)
      people_test = params1
    else:
        #response = requests.post(url=url, data=params2)
        people_test = params2
        
    cv2.imshow('MediaPipe Face Detection', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()