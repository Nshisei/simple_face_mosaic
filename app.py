import streamlit as st
from streamlit_webrtc import webrtc_streamer
import mediapipe as mp
import cv2
import av
import os

from face_cover import overlay_image_alpha
import os
import sys
from face_mask.apply_filter import apply_face_mask, getLandmarks

def face_detection_using_mediapipe():
    min_detection_confidence = st.slider('Min Detection Confidence', min_value=0.0, max_value=1.0, value=0.5, step=0.1)
    def callback(frame):
        mp_face_detection = mp.solutions.face_detection
        image = frame.to_ndarray(format="bgr24")
        with mp_face_detection.FaceDetection(min_detection_confidence=min_detection_confidence) as face_detection:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_detection.process(image_rgb)
            if results.detections:
                for detection in results.detections:
                    
                    bbox = detection.location_data.relative_bounding_box
                    h, w, _ = image.shape
                    x_min = int(bbox.xmin * w)
                    y_min = int(bbox.ymin * h)
                    box_width = int(bbox.width * w)
                    box_height = int(bbox.height * h)

                    # 検出の信頼度を取得して表示
                    confidence = detection.score[0]
                    if min_detection_confidence > confidence:
                        break
                    text = f'{confidence:.2f}'
                    cv2.putText(image, text, (x_min, y_min - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)


                    # カバー画像を重ねる
                    overlay_size = (int(box_width * 1.5), int(box_height * 1.5))
                    print("detection process")
                    image = overlay_image_alpha(image, x_min + box_width // 2, y_min + box_height // 2, overlay_size)
                    print("end overlay")
        
        return av.VideoFrame.from_ndarray(image, format="bgr24")

    webrtc_streamer(key="face-overlay-mediapipe", video_frame_callback=callback, async_processing=True, media_stream_constraints={'video': True, 'audio': False}, rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# 顔検出とFaceMeshを使用した関数
def face_mesh_using_mediapipe():
    min_detection_confidence = st.slider('Min Detection Confidence', min_value=0.0, max_value=1.0, value=0.5, step=0.1)

    def callback(frame):
        frame = frame.to_ndarray(format="bgr24")
        frame = cv2.resize(frame, (1280, 960))
        points2 = getLandmarks(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), max_num_faces=10, min_detection_confidence=min_detection_confidence)
        if len(points2) == 0:
            image = frame
        else:
            image = apply_face_mask(frame, "smily", points2)

        return av.VideoFrame.from_ndarray(image, format="bgr24")

    webrtc_streamer(
        key="facemesh-overlay-mediapipe",
        video_frame_callback=callback,
        async_processing=True,
        media_stream_constraints={'video': True, 'audio': False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

def main():
    st.title('Moke up app')

    select_app = st.sidebar.radio('app', ('Face Detection using Mediapipe', 
                                          'Face Mesh using Mediapipe'))
    
    func_dict = {
    'Face Detection using Mediapipe': face_detection_using_mediapipe,
    'Face Mesh using Mediapipe': face_mesh_using_mediapipe,
    }

    func_dict.get(select_app, lambda: st.error('Invalid Selection'))()    


if __name__ == '__main__':
    main()