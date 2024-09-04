import cv2
from google.cloud import vision
import io
import os
import time

# Correct the path to your Haar cascade file
harcascade = r"C:\Users\admin\Desktop\haldia3\model\haarcascade_russian_plate_number.xml"

# Set up Google Cloud Vision client
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"C:\Users\admin\Downloads\rich-access-426306-r9-a6744daa40f5.json"
client = vision.ImageAnnotatorClient()

# Create the directory to save the images if it doesn't exist
save_dir = "plates"
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)

cap.set(3, 640)  # width
cap.set(4, 480)  # height

min_area = 500
prev_image_path = os.path.join(save_dir, "scanned_img.jpg")
last_detection_time = 0
detection_interval = 25  # seconds

def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize the image to improve OCR accuracy
    resized = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    # Apply binary thresholding
    _, binary = cv2.threshold(resized, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def detect_and_process_plate(img_roi):
    # Save the image with a fixed name
    cv2.imwrite(prev_image_path, img_roi)
    print("Plate image saved.")

    # Preprocess the image
    preprocessed_image = preprocess_image(img_roi)
    preprocessed_image_path = os.path.join(save_dir, "preprocessed_img.jpg")
    cv2.imwrite(preprocessed_image_path, preprocessed_image)

    # Perform OCR on the preprocessed image
    if os.path.exists(preprocessed_image_path):
        with io.open(preprocessed_image_path, 'rb') as image_file:
            content = image_file.read()

        image = vision.Image(content=content)

        try:
            response = client.text_detection(image=image)
            texts = response.text_annotations
            if texts:
                plate_text = texts[0].description.strip()
                print(f"Detected Number Plate Text ({preprocessed_image_path}): {plate_text}")
            else:
                print("No text detected")
        except Exception as e:
            print(f"Error occurred during OCR: {e}")

# Load Haar cascade
plate_cascade = cv2.CascadeClassifier(harcascade)

# Check if the Haar cascade is loaded correctly
if plate_cascade.empty():
    print("Error loading Haar cascade file. Please check the path and file.")
    exit()

while True:
    success, img = cap.read()

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

    for (x, y, w, h) in plates:
        area = w * h

        if area > min_area:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, "Number Plate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)

            img_roi = img[y: y + h, x: x + w]
            cv2.imshow("ROI", img_roi)

            current_time = time.time()
            if current_time - last_detection_time > detection_interval:
                detect_and_process_plate(img_roi)
                last_detection_time = current_time

    cv2.imshow("Result", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()