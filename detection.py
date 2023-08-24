import torch
import cv2
import numpy as np
import time

# Load the PyTorch model
model = torch.load("leafdetector.pth")
disease_model = torch.jit.load("qmodel_script.pt")
device = torch.device('cpu')
model = model.to(device)
disease_model = disease_model.to(device)
disease_model.eval()
model.eval()

# Helper function to get boxes from tensor
def get_boxes(tensor, index, score=0.5):
    if index >= len(tensor) or index < 0:
        return []

    temp_boxes = []
    for i in range(len(tensor[index]['boxes'])):
        if tensor[index]['scores'][i] > score:
            temp_boxes.append(tensor[index]['boxes'][i].cpu().detach().numpy().astype(np.int32))

    return temp_boxes

def model_inference(input_slice):
    
    resized_frame = cv2.resize(input_slice, (224, 224))
    normalized_frame = resized_frame / 255.0
    transposed_frame = np.transpose(normalized_frame, (2, 0, 1))
    input_frame = np.expand_dims(transposed_frame, axis=0).astype(np.float32)
    input_frame = torch.from_numpy(input_frame)

    output = disease_model(input_frame)
    predicted_class = torch.argmax(output[0])
    labeled = classes[predicted_class]

    return labeled

def visualize_frame(frame, score=0.5, fps=0):
    image_tensor = torch.tensor(frame, dtype=torch.float32, device=device)
    image_tensor = image_tensor.permute(2, 0, 1) / 255.0  # Permute dimensions for RGB
    with torch.no_grad():
        outputs = model([image_tensor])

    boxes = get_boxes(outputs, 0, score)

    img = frame.copy()

    for box in boxes:
        x, y, w, h = box
        x, y, w, h = int(x), int(y), int(w), int(h)
        
        slice = img[y:y+h, x:x+w]
        prediction = model_inference(slice)

        cv2.putText(img, prediction, ((x + w) // 2, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(img,
                      (x, y),
                      (w, h),
                      (0, 255, 0),
                      2)

    # Add FPS information to the frame
    cv2.putText(img, f'FPS: {fps}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return img

classes = ('healthy', 'infected')
cap = cv2.VideoCapture(0)

prev_frame_time = 0
new_frame_time = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    fps = str(round(float(fps), 1))
    resized_frame = cv2.resize(frame, (1280, 720))

    result_frame = visualize_frame(resized_frame, fps=fps)

    cv2.imshow("Object Detection", result_frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
