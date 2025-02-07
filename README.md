# OpenSesame - Autonomous Door Access with Face Recognition

## Overview
OpenSesame is an intelligent door access system that utilizes machine learning for facial recognition and Java to autonomously unlock doors. The system processes live video streams, detects faces, identifies authorized individuals, and grants access by triggering a smart lock.

## Features
- **Live Face Detection**: Uses OpenCV to detect faces in real-time from an RTSP camera stream.
- **Face Recognition**: Implements a deep learning-based face embedding model for accurate user identification.
- **Secure Door Access**: Grants access only to authorized individuals by sending authenticated HTTP requests.
- **Optimized Frame Processing**: Uses multi-threading to ensure smooth video processing and face recognition.
- **Logging & Debugging**: Outputs recognition status and access attempts to the console.

## Technologies Used
- **Java**: Core programming language
- **OpenCV**: Face detection and image processing
- **TensorFlow**: Face embedding model (Facenet)
- **Apache HttpClient**: Sending authentication requests to unlock doors

## Prerequisites
### Software & Libraries
- Java 11 or higher
- OpenCV 4.x
- TensorFlow for Java (Facenet Model)
- Apache HttpClient 5

### Hardware
- IP Camera with RTSP support
- Smart lock with HTTP-based control

## Installation & Setup
1. **Clone the Repository:**
   ```sh
   git clone https://github.com/yourusername/OpenSesame.git
   cd OpenSesame
   ```

2. **Download Dependencies:**
    - Install OpenCV and ensure Java bindings are available.
    - Download the Facenet model (`facenet_model.pb`) and place it in the project directory.

3. **Configure Camera & Door Lock Credentials:**
    - Update `rtspURL` in `CameraStream.java` with your RTSP stream URL.
    - Modify `DOOR_UNLOCK_URL`, username, and password in `CameraStream.java`.

4. **Run the Application:**
   - Using IntelliJ Set VM options: -Djava.library.path=C:\opencv\build\java\x64

    Run CameraStream.java from IntelliJ.

## How It Works
1. The system continuously streams frames from the RTSP camera.
2. Each frame is processed for face detection using OpenCV.
3. If a face is found, it is passed through a deep learning model to extract embeddings.
4. The extracted embedding is compared with stored authorized users.
5. If a match is found, an HTTP request is sent to unlock the door.
6. The system ensures access is granted only once per minute for security.

## Future Enhancements
- Implement a web interface for user management.
- Improve model accuracy with additional training data.


## Contributors
- **Ali Alsadadi** (@aliaalsadadi)
- Open to contributions! Feel free to submit pull requests.

## Acknowledgments
- OpenCV team for real-time face detection
- TensorFlow community for deep learning support

