import numpy as np
import cv2
import onnxruntime
import time

cap = cv2.VideoCapture(0)
disease_model = onnxruntime.InferenceSession("C:/Users/franz/OneDrive/Documents/computer vision/assets/disease-model.onnx")

classes = ('healthy', 'infected')

prev_frame_time = 0
new_frame_time = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    new_frame_time = time.time()
  
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time

    resized_frame = cv2.resize(frame, (256, 256))
    normalized_frame = resized_frame / 255.0
    transposed_frame = np.transpose(normalized_frame, (2, 0, 1))
    input_frame = np.expand_dims(transposed_frame, axis=0).astype(np.float32)

    # Run inference
    output = disease_model.run(None, {'input.1': input_frame})

    # Get predicted class
    predicted_class = np.argmax(output[0])
    labeled = classes[predicted_class]

    fps = str(int(fps))

    width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    org_w = int(0.85 * width)
    org_h = int(0.85 * height)
    cv2.putText(frame, f'FPS: {str(fps)}', (org_w, org_h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    cv2.putText(frame, f'Prediction: {labeled}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Inference', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break        

cap.release()
cv2.destroyAllWindows()