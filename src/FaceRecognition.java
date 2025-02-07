import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

public class FaceRecognition {
    private final CascadeClassifier faceDetector;
    private final FaceEmbedder faceEmbedder;
    private final Map<String, float[]> faceDatabase;
    private static final double RECOGNITION_THRESHOLD = 0.25;

    static {
        // Load OpenCV native library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public FaceRecognition(String cascadePath, String modelPath, String databasePath) {
        this.faceDetector = new CascadeClassifier(cascadePath);
        this.faceEmbedder = new FaceEmbedder(modelPath);
        this.faceDatabase = loadFaceDatabase(databasePath);
    }

    public void startVideoRecognition() {
        VideoCapture capture = new VideoCapture(0);
        if (!capture.isOpened()) {
            System.out.println("Error: Cannot open camera");
            return;
        }

        Mat frame = new Mat();
        while (true) {
            capture.read(frame);
            if (frame.empty()) break;

            processFrame(frame);

            // Display the frame
            HighGui.imshow("Face Recognition", frame);
            if (HighGui.waitKey(1) >= 0) break;
        }

        capture.release();
        HighGui.destroyAllWindows();
    }

    private void processFrame(Mat frame) {
        // Convert to grayscale for face detection
        Mat gray = new Mat();
        Imgproc.cvtColor(frame, gray, Imgproc.COLOR_BGR2GRAY);

        // Detect faces
        MatOfRect faces = new MatOfRect();
        faceDetector.detectMultiScale(gray, faces);

        for (Rect face : faces.toArray()) {
            // Extract and preprocess face
            Mat faceROI = new Mat(frame, face);
            Mat processedFace = preprocessFace(faceROI);

            // Get face embedding
            float[] embedding = faceEmbedder.getEmbedding(processedFace);

            // Recognize face
            String name = recognizeFace(embedding);

            // Draw rectangle and name
            Imgproc.rectangle(frame, face, new Scalar(0, 255, 0), 2);
            Imgproc.putText(frame, name,
                    new Point(face.x, face.y - 10),
                    Imgproc.FONT_HERSHEY_SIMPLEX, 1.0,
                    new Scalar(0, 255, 0), 2);
        }
    }

    private Mat preprocessFace(Mat face) {
        Mat resized = new Mat();
        Imgproc.resize(face, resized, new Size(160, 160));

        // Convert to float and normalize
        Mat normalized = new Mat();
        resized.convertTo(normalized, CvType.CV_32F, 1.0/255.0);

        return normalized;
    }

    private String recognizeFace(float[] embedding) {
        String bestMatch = "Unknown";
        double minDistance = Double.MAX_VALUE;

        for (Map.Entry<String, float[]> entry : faceDatabase.entrySet()) {
            double distance = calculateDistance(embedding, entry.getValue());
            if (distance < minDistance) {
                minDistance = distance;
                bestMatch = entry.getKey();
            }
        }
        System.out.println(minDistance);
        return minDistance < RECOGNITION_THRESHOLD ? bestMatch : "Unknown";
    }

    private double calculateDistance(float[] embedding1, float[] embedding2) {
        double sum = 0.0;
        for (int i = 0; i < embedding1.length; i++) {
            sum += Math.pow(embedding1[i] - embedding2[i], 2);
        }
        return Math.sqrt(sum);
    }

    public Map<String, float[]> loadFaceDatabase(String path) {
        Map<String, float[]> database = new HashMap<>();
        try (BufferedReader reader = new BufferedReader(new FileReader(path))) {
            String line;
            while ((line = reader.readLine()) != null) {
                String[] parts = line.split(",", 2);
                if (parts.length == 2) {
                    String name = parts[0];
                    String[] embeddingStr = parts[1].split(" ");
                    float[] embedding = new float[embeddingStr.length];
                    for (int i = 0; i < embeddingStr.length; i++) {
                        embedding[i] = Float.parseFloat(embeddingStr[i]);
                    }
                    database.put(name, embedding);
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return database;
    }
}
