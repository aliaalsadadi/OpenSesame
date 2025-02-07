import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.*;
import java.util.Arrays;

public class FaceDatabaseCreator {
    private final CascadeClassifier faceDetector;
    private final FaceEmbedder faceEmbedder;
    private final String databasePath;

    static {
        // Load OpenCV native library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public FaceDatabaseCreator(String cascadePath, String modelPath, String databasePath) {
        this.faceDetector = new CascadeClassifier(cascadePath);
        this.faceEmbedder = new FaceEmbedder(modelPath);
        this.databasePath = databasePath;
    }

    public void addFaceToDatabase(String imagePath, String personName) {
        // Read the image
        Mat image = Imgcodecs.imread(imagePath);
        if (image.empty()) {
            System.out.println("Error: Could not read image at " + imagePath);
            return;
        }

        // Convert to grayscale for face detection
        Mat gray = new Mat();
        Imgproc.cvtColor(image, gray, Imgproc.COLOR_BGR2GRAY);

        // Detect faces
        MatOfRect faces = new MatOfRect();
        faceDetector.detectMultiScale(gray, faces);

        if (faces.toArray().length == 0) {
            System.out.println("No faces detected in the image");
            return;
        }

        Rect[] faceArray = faces.toArray();

        // Draw rectangles around detected faces
        for (int i = 0; i < faceArray.length; i++) {
            Rect rect = faceArray[i];

            // Use a **different color** for the first detected face
            Scalar color = (i == 0) ? new Scalar(255, 0, 0) : new Scalar(0, 255, 0); // Blue for first, Green for others
            Imgproc.rectangle(image, rect.tl(), rect.br(), color, 2);
        }

        // Save the image with detected faces for debugging
        String debugImagePath = "debug_" + imagePath;
        Imgcodecs.imwrite(debugImagePath, image);
        System.out.println("Debug image saved to: " + debugImagePath);

        if (faceArray.length > 1) {
            System.out.println("Warning: Multiple faces detected, using the first one");
        }

        // Process the first detected face
        Rect faceRect = faceArray[0];
        Mat faceROI = new Mat(image, faceRect);
        Mat processedFace = preprocessFace(faceROI);

        // Get face embedding
        float[] embedding = faceEmbedder.getEmbedding(processedFace);

        // Save to database
        saveToDatabase(personName, embedding);

        System.out.println("Successfully added face embedding for " + personName);
    }


    private Mat preprocessFace(Mat face) {
        Mat resized = new Mat();
        Imgproc.resize(face, resized, new Size(160, 160));

        // Convert to float and normalize
        Mat normalized = new Mat();
        resized.convertTo(normalized, CvType.CV_32F, 1.0/255.0);

        return normalized;
    }
    private void saveToDatabase(String name, float[] embedding) {
        try (FileWriter fw = new FileWriter(databasePath, true);
             BufferedWriter bw = new BufferedWriter(fw)) {

            // Convert embedding array to space-separated string
            StringBuilder embeddingStr = new StringBuilder();
            for (int i = 0; i < embedding.length; i++) {
                if (i > 0) {
                    embeddingStr.append(" ");
                }
                embeddingStr.append(String.valueOf(embedding[i]));
            }

            // Write name and embedding to file
            bw.write(name + "," + embeddingStr);
            bw.newLine();

        } catch (IOException e) {
            System.out.println("Error writing to database: " + e.getMessage());
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        // Example usage
        System.out.println(
                "populating database....."
        );
        String cascadePath = "haarcascade_frontalface_default.xml";
        String modelPath = "facenet_model.pb";
        String databasePath = "face_database.csv";

        FaceDatabaseCreator creator = new FaceDatabaseCreator(
                cascadePath,
                modelPath,
                databasePath
        );

        // Add your face to the database
//        for (int i=1;i<=10;i++){
//            creator.addFaceToDatabase("me"+i+".jpg", "Ali"+i);
//
//        }
        creator.addFaceToDatabase("ali.jpg","Ali");
    }
}