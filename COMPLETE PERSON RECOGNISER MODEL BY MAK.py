import os
import cv2
import pickle
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
from scipy.fftpack import fft2

# Initialize InsightFace (MobileFaceNet)
app = FaceAnalysis(name="buffalo_l")  # Uses MobileFaceNet for embeddings
app.prepare(ctx_id=0, det_size=(160, 160))

# Load trained embeddings
def load_embeddings(model_path="models/face_data.pkl"):
    if not os.path.exists(model_path):
        print("❌ No stored face data found. Train the model first.")
        return None
    with open(model_path, "rb") as f:
        return pickle.load(f)

# Extract face embeddings
def extract_face_embedding(image):
    faces = app.get(image)
    if len(faces) == 0:
        return None
    return faces[0].normed_embedding  # Get feature vector

# Recognize face
def recognize_face(image, face_data, threshold=0.6):
    embedding = extract_face_embedding(image)
    if embedding is None:
        return "No face detected."
    
    best_match, best_score = None, -1
    for person_name, known_embedding in face_data.items():
        similarity = cosine_similarity([embedding], [known_embedding])[0][0]
        if similarity > best_score:
            best_score = similarity
            best_match = person_name

    return best_match if best_score >= threshold else "Unknown Person"

### --- IMAGE SOURCE DETECTION FUNCTIONS --- ###

# 1️⃣ Detect Screenshots or Images Taken from a Screen
def detect_screenshot(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_count = np.sum(edges == 255)
    fft_result = np.abs(fft2(gray))
    high_freq_energy = np.sum(fft_result[:, :10])
    return edge_count > 200000 or high_freq_energy > 10000

# 2️⃣ Detect Live Camera Images (Freshly Captured)
def detect_live_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = np.mean(gray)
    return laplacian_var > 100 and brightness > 80

# 3️⃣ Detect Pre-existing (Downloaded) Images
def detect_downloaded_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    artifact_score = np.mean(magnitude_spectrum)
    return artifact_score < 150

# 4️⃣ Detect Motion for Live Webcam
def detect_motion(prev_frame, curr_frame, threshold=5000):
    diff = cv2.absdiff(prev_frame, curr_frame)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
    motion_score = np.sum(thresh)
    return motion_score > threshold

### --- RECOGNITION FUNCTIONS --- ###

def recognize_from_image(image_path, model_path="models/face_data.pkl"):
    if not os.path.exists(image_path):
        print("❌ Invalid path. Try again.")
        return
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Error: Cannot read image.")
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_data = load_embeddings(model_path)
    if face_data is None:
        return
    
    if detect_downloaded_image(img):
        source = "Downloaded Image"
    elif detect_live_image(img):
        source = "Freshly Taken Image"
    else:
        source = "Screen Display or Printed Image"

    result = recognize_face(img_rgb, face_data)
    print(f"🔍 Recognition Result: {result}, Source: {source}")

def detect_brightness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

def detect_moire_pattern(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fft_result = np.abs(fft2(gray))
    high_freq_energy = np.sum(fft_result[:, :10])
    return high_freq_energy > 15000

# Recognize from Webcam
def recognize_from_webcam(model_path="models/face_data.pkl"):
    cap = cv2.VideoCapture(0)
    face_data = load_embeddings(model_path)
    
    if face_data is None:
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Camera Error!")
            break
        
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = recognize_face(img_rgb, face_data)
        brightness = detect_brightness(frame)
        moire_detected = detect_moire_pattern(frame)

        if brightness > 90:
            source = "Live Person"
        elif moire_detected:
            source = "Photo on Mobile Screen"
        else:
            source = "Pre-existing Image"

        cv2.putText(frame, f"Result: {result}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.putText(frame, f"Source: {source}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 
                    (0, 0, 255) if source != "Live Person" else (0, 255, 0), 3)
        
        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Main Menu
if __name__ == "__main__":
    while True:
        mode = input("Choose input method: (1) Image Path (2) Live Webcam (exit or e to quit): ").strip()
        if mode == "1":
            image_path = input("Enter image path: ").strip()
            recognize_from_image(image_path)
        elif mode == "2":
            recognize_from_webcam()
        elif mode.lower() == "exit" or "e":
            break
        else:
            print("❌ Invalid choice. Try again.")