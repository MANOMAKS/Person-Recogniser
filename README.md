AI-Powered Face Recognition with Spoof Detection

This project is an advanced face recognition system built using **Python**, **InsightFace (MobileFaceNet)**, and **OpenCV**. It can recognize known individuals and detect whether the input is from:
- A **live person** (webcam feed)
- A **screen photo** (e.g., from a phone or monitor)
- A **pre-existing image** (downloaded or printed)

Features
- Face recognition using cosine similarity and embeddings
- Input via **image path** or **live webcam**
- Detects input source type:
- Live person
- Mobile/computer screen
- Downloaded/pre-existing photo
- Uses FFT, Laplacian variance, brightness, edge detection for source classification
- Face data stored and loaded using Pickle

Tech Stack
- Python 3
- OpenCV
- InsightFace (`buffalo_l` with MobileFaceNet)
- NumPy, SciPy
- scikit-learn

Demo (Screenshots or GIFs)
1. ![image](https://github.com/user-attachments/assets/01729ba3-56c4-4f3a-a684-0624eeb2cfc5)
2. ![image](https://github.com/user-attachments/assets/6e09efef-b4bb-4661-b932-bfcc66f6bb50)
3. ![image](https://github.com/user-attachments/assets/1f58df00-5f67-431c-b9ea-c1476ed38b07)


How to Use

1. Clone the repository  
   ```bash
   git clone https://github.com/yourusername/face-recognition-spoof-detection.git
pip install insightface opencv-python scikit-learn numpy scipy
python your_script.py

Note:
This project leverages the powerful MobileFaceNet-based buffalo_l model from InsightFace, where custom face embeddings are generated and trained using user-provided images to enable highly accurate and secure face recognition.

📬 Contact

👨‍💻 Developed by Manoj Kumar . M
📧 Email: manomaksj@gmail.com
🔗 www.linkedin.com/in/manomak
