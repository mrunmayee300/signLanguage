Sign Language Real-Time Interpreter 🤟📷
This project is a real-time Sign Language Interpreter that uses MediaPipe, OpenCV, and TensorFlow to recognize and translate basic sign language gestures into text. It supports the following gestures:
✅ Hello 
✅ Bye 
✅ I Love You 
✅ Thank You 
✅ Yes 
✅ No 

✨ Features
 Real-time hand tracking using MediaPipe Hands
 Gesture classification with a trained neural network
 Live text output for recognized gestures
 Custom dataset collection for sign language training
 User-friendly interface with OpenCV



 📂 Project Structure
data_collection.py → Collects hand gesture data & saves it as CSV

train.py → Trains a gesture classification model

predict.py → Runs real-time sign language detection

sign_language_data.csv → Dataset for training

🚀 How to Run
1️) Install dependencies:
sh
Copy
Edit
pip install -r requirements.txt

2️) Collect gesture data (if needed):
sh
Copy
Edit
python data_collection.py

3️) Train the model:
sh
Copy
Edit
python train.py

4️) Run real-time interpreter:
sh
Copy
Edit
python predict.py


📌 Technologies Used
Python 
OpenCV 
MediaPipe 
TensorFlow/Keras 
NumPy & Pandas 
