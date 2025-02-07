import org.apache.hc.client5.http.auth.AuthScope;
import org.apache.hc.client5.http.auth.CredentialsProvider;
import org.apache.hc.client5.http.auth.UsernamePasswordCredentials;
import org.apache.hc.client5.http.classic.methods.HttpGet;
import org.apache.hc.client5.http.impl.auth.BasicCredentialsProvider;
import org.apache.hc.client5.http.impl.classic.CloseableHttpClient;
import org.apache.hc.client5.http.impl.classic.CloseableHttpResponse;
import org.apache.hc.client5.http.impl.classic.HttpClients;
import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;
import java.io.IOException;
import java.util.Map;

public class CameraStream {
    private final CascadeClassifier faceDetector;
    private final FaceEmbedder faceEmbedder;
    private final Map<String, float[]> faceDatabase;
    private static final double RECOGNITION_THRESHOLD = 0.25;
    private static final String DOOR_UNLOCK_URL = "http://192.168.100.109/cgi-bin/accessControl.cgi?action=openDoor&channel=1";

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public CameraStream(String cascadePath, String modelPath, String databasePath) {
        this.faceDetector = new CascadeClassifier(cascadePath);
        this.faceEmbedder = new FaceEmbedder(modelPath);
        this.faceDatabase = new FaceRecognition(cascadePath, modelPath, databasePath).loadFaceDatabase(databasePath);
    }

    public void startStream() {
        String rtspURL = "rtsp://admin:Admin@12345@192.168.100.109:554/cam/realmonitor?channel=1&subtype=0";
        VideoCapture camera = new VideoCapture();
        camera.open(rtspURL, Videoio.CAP_FFMPEG);

        if (!camera.isOpened()) {
            System.out.println("Error: Cannot open stream");
            return;
        }

        Mat frame = new Mat();
        while (camera.read(frame)) {
            if (frame.empty()) continue;

            processFrame(frame);
            HighGui.imshow("RTSP Stream - Face Recognition", frame);

            if (HighGui.waitKey(1) >= 0) break;
        }

        camera.release();
        HighGui.destroyAllWindows();
    }

    private void processFrame(Mat frame) {
        Mat gray = new Mat();
        Imgproc.cvtColor(frame, gray, Imgproc.COLOR_BGR2GRAY);

        MatOfRect faces = new MatOfRect();
        faceDetector.detectMultiScale(gray, faces);

        for (Rect face : faces.toArray()) {
            Mat faceROI = new Mat(frame, face);
            Mat processedFace = preprocessFace(faceROI);
            float[] embedding = faceEmbedder.getEmbedding(processedFace);
            String name = recognizeFace(embedding);

            // Draw rectangle and label
            Scalar color = name.equals("Unknown") ? new Scalar(0, 0, 255) : new Scalar(0, 255, 0);
            Imgproc.rectangle(frame, face, color, 2);
            Imgproc.putText(frame, name, new Point(face.x, face.y - 10),
                    Imgproc.FONT_HERSHEY_SIMPLEX, 0.9, color, 2);

            // If Ali is detected, open the door
            if (name.equals("Ali")) {
                openDoor();
            }
        }
    }

    private Mat preprocessFace(Mat face) {
        Mat resized = new Mat();
        Imgproc.resize(face, resized, new Size(160, 160));
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
        return minDistance < RECOGNITION_THRESHOLD ? bestMatch : "Unknown";
    }

    private double calculateDistance(float[] e1, float[] e2) {
        double sum = 0.0;
        for (int i = 0; i < e1.length; i++) {
            sum += Math.pow(e1[i] - e2[i], 2);
        }
        return Math.sqrt(sum);
    }

    private long lastUnlockTime = 0;

    private void openDoor() {
        long currentTime = System.currentTimeMillis();

        // Check if at least 1 minute has passed since the last unlock
        if (currentTime - lastUnlockTime < 60000) {
            System.out.println("⏳ Door was recently unlocked. Waiting before next unlock...");
            return;
        }

        System.out.println("🔓 Ali detected! Unlocking door...");
        try {
            // Create Digest Authentication Credentials
            CredentialsProvider credsProvider = new BasicCredentialsProvider();
            ((BasicCredentialsProvider) credsProvider).setCredentials(
                    new AuthScope("192.168.100.109", 80),
                    new UsernamePasswordCredentials("admin", "Admin@12345".toCharArray())
            );

            // Create HTTP Client with Digest Auth
            try (CloseableHttpClient httpClient = HttpClients.custom()
                    .setDefaultCredentialsProvider(credsProvider)
                    .build()) {

                HttpGet request = new HttpGet(DOOR_UNLOCK_URL);
                try (CloseableHttpResponse response = httpClient.execute(request)) {
                    System.out.println("Door unlock request sent. Response: " + response.getReasonPhrase());

                    // Update last unlock time
                    lastUnlockTime = System.currentTimeMillis();
                }
            }
        } catch (IOException e) {
            System.err.println("❌ Failed to open door: " + e.getMessage());
        }
    }


    public static void main(String[] args) {
        String cascadePath = "haarcascade_frontalface_default.xml";
        String modelPath = "facenet_model.pb";
        String databasePath = "face_database.csv";

        CameraStream stream = new CameraStream(cascadePath, modelPath, databasePath);
        stream.startStream();
    }
}
