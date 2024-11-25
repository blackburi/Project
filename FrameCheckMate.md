# Frame Checkmate Project
* 영상 편집 협업 툴 프로젝트
* 기간 : 2024.10.14 ~ 2024.11.19
* 역할 : FE 개발자, AI(YOLOv8m) 파인 튜닝 및 튜닝, flask 서버 배포
* 내용 : 파인 튜닝한 YOLOv8m 모델을 활용하여 방송 심의에 걸릴 수 있는 요소가 포함된 영상을 편집할 수 있는 협업 툴을 개발했습니다. 이 툴에서 관리자가 전체 영상을 업로드하면 YOLOv8m 모델이 자동으로 담배와 칼과 같은 편집이 필요한 객체를 검출하고 해당 시점을 관리자에게 표시합니다. 관리자는 이 시점을 기반으로 영상을 여러 구간(card)으로 분할하고, 작업할 사람들을 초대하여 각 카드에 작업자와 작업 기간을 설정합니다. 설정된 내용에 따라 작업자에게 메일이 발송되며, 작업 페이지에서 작업자들은 자신의 영상 구간을 확인하고 작업을 수행할 수 있습니다. 모든 카드의 작업이 완료되면, 초기 영상에서 분할된 순서대로 자동으로 영상이 병합되어 최종 영상이 완성됩니다. 이를 통해 관리자와 작업자들은 효율적으로 협업하여 심의 요소가 제거된 영상을 빠르고 체계적으로 제작할 수 있습니다.


## AI

### Object Detection Model vs Image Classification Model
* Object Detection Model(객체 탐지 모델)과 Image Classification Model(이미지 분류 모델) 중 Object Detection Model을 선택하였다. 두 Model의 차이는 Image Classification Model의 경우 전체 image에서 single label 또는 boundary를 예측하는 특징이 있고, Object Detection Model의 경우 Image 내에서 여러 객체를 식별하고 위치를 파악한다는 특징이 있다.
* 이번 Project의 경우, 영상의 각 Frame에서 특정 Object(객체)를 검출(탐지)해야 하고, 여러 개일 경우도 존재하기 때문에 Multi Object Detection Model의 모델중 하나인 YOLOv8 Model을 사용하였고, 그 중 3분짜리 test영상(cigarette video)을 사용하였을때 정확성과 속도의 측면에서 중간 모델(YOLOv8m)을 사용하였다.

### code
* YOLOv8m 모델 학습
    * YOLOv8의 경우 `ultralytics` 라이브러리를 쉽고 빠르게 가져올 수 있다는 장점을 가지고 있다. 이번 Project에서 YOLOv8m을 사용하는데 `Knife` class는 dataset에 포함되어 있지만, `cigarette` class는 dataset에 포함되지 않아 Fine-Tuning(파인튜닝, 추가 학습)이 필요로 했다. 또한 YOLO 모델은 Object Detection Model이기 때문에 dataset 학습 시 이미지에 학습시킬 class에 대한 bounding box를 지정해야 하고, parameter로 class_id, 중심 x좌표(center_x), 중심 y좌표(center_y), 박스 너비(width), 박스 높이(height) 총 5개가 들어간다. class id를 제외한 parameter는 모두 이미지에 대한 비율값을 사용하기 때문에 모두 0.0~1.0 사이의 값을 사용한다. YOLOv8 모델은 80개의 object가 기본적으로 학습되어 있어 class_id가 0~79까지 사용되고 있기 때문에 `cigarette` class를 fine-tuning시에 `class+_id = 80`을 지정하였다.
    ```python
    # YOLOv8 학습 시키는 파일

    from ultralytics import YOLO

    # YOLOv8m 모델 로드
    model = YOLO('yolov8m.pt')

    # 모델 학습 -> 학습된 모델은 'runs/detect/train/weights/best.pt'에 저장됨
    model.train(
        data='cigarette.yaml',   # 데이터셋 구성 파일 경로
        epochs=50,               # 학습 반복 횟수
        imgsz=640,               # 입력 이미지 크기
        batch=16                 # 배치 크기 (필요에 따라 조정 가능)
    )
    ```
    ```yaml
    # cigarette.yaml

    train: C:/Users/SSAFY/Desktop/YOLOv8/cigarette/train/images
    val: C:/Users/SSAFY/Desktop/YOLOv8/cigarette/valid/images

    nc: 1
    names: ['cigarette']
    ```

* `best.py`에 저장된 fine-tuning된 모델을 불러와서 사용하고, 통신을 위해 `JSON`으로 변환하여 통신했다. Frontend에서 data 사용의 편의성을 위해 객체를 탐지한 시간을 이중 리스트 형태로 통신했다. 예를 들어 `[[0, 3], [5, 8]]`의 경우 0~3초, 5~8초 객체가 탐지된 것이다.
    ```python
    from flask import Flask, jsonify, request
    from flask_cors import CORS
    import cv2
    from ultralytics import YOLO
    from datetime import datetime
    from py_eureka_client import eureka_client
    import server_config as server

    app = Flask(__name__)
    CORS(app, resources={r"/*": {"origins": "*", "methods": ["GET", "POST", "OPTIONS"], "allow_headers": ["Content-Type", "Authorization", "Access-Control-Allow-Origin"]}})

    eureka_client.init(eureka_server=server.EUREKA_SERVER,
                    app_name=server.SERVICE_NAME,
                    instance_host=server.SERVICE_HOST,
                    instance_port=server.SERVICE_PORT)

    @app.route('/predict', methods=['POST', 'OPTIONS'])
    def predict():
        # OPTIONS 요청을 처리하여 프리플라이트 요청에 응답
        if request.method == 'OPTIONS':
            # OPTIONS 요청에 대해 허용 응답을 반환
            response = jsonify({'status': 'OK'})
            response.headers.add("Access-Control-Allow-Origin", "*")
            response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
            response.headers.add("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
            return response

        # JSON 요청에서 URL 가져오기
        data = request.get_json()
        if not data or 'url' not in data:
            print('error', ':', 'No URL provided in request')
            return jsonify({'error': 'No URL provided in request'}), 400

        video_url = data['url']
        cap = cv2.VideoCapture(video_url)

        # 학습된 모델 불러오기 -> 담배 탐지
        model = YOLO('./best.pt')

        # FPS 가져오기
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps)

        detection_times = []
        frame_index = 0
        last_time = None
        current_range = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % frame_interval == 0:
                frame_time = int(frame_index / fps)
                results = model.predict(source=frame, save=False, save_txt=False, verbose=False)

                for result in results:
                    if result.boxes:
                        if last_time is None:
                            current_range = [frame_time, frame_time]
                        elif frame_time == last_time + 1:
                            current_range[1] = frame_time
                        else:
                            detection_times.append(current_range)
                            current_range = [frame_time, frame_time]
                        last_time = frame_time
                        break

            frame_index += 1

        if current_range:
            detection_times.append(current_range)

        cap.release()

        # POST 요청에 대한 응답에 CORS 헤더 추가
        response = jsonify({
            'status': 'success',
            'detection_times': detection_times
        })
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response

    if __name__ == '__main__':
        app.run(host='0.0.0.0', port=8083, debug=True)
    ```