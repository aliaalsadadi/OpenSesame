// FaceRecognition.java
import org.opencv.core.*;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.ndarray.buffer.DataBuffers;
import org.tensorflow.proto.framework.GraphDef;
import org.tensorflow.types.TBool;
import org.tensorflow.types.TFloat32;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;

// FaceEmbedder.java
class FaceEmbedder implements AutoCloseable {
    private final Graph graph;
    private final Session session;

    public FaceEmbedder(String modelPath) {
        this.graph = new Graph();
        try {
            byte[] graphDef = Files.readAllBytes(Path.of(modelPath));
            graph.importGraphDef(GraphDef.parseFrom(graphDef));
            this.session = new Session(graph);
        } catch (IOException e) {
            throw new RuntimeException("Failed to load model", e);
        }
    }
    public float[] getEmbedding(Mat face) {
        // Convert Mat to tensor
        float[] inputFlattened = new float[1 * 160 * 160 * 3];
        int index = 0;

        for (int y = 0; y < 160; y++) {
            for (int x = 0; x < 160; x++) {
                double[] pixel = face.get(y, x);
                for (int c = 0; c < 3; c++) {
                    inputFlattened[index++] = (float) pixel[c];
                }
            }
        }

        try (TFloat32 inputTensor = TFloat32.tensorOf(
                Shape.of(1, 160, 160, 3),
                DataBuffers.of(inputFlattened));
             TBool phaseTrainTensor = TBool.scalarOf(false)) {

            Tensor outputTensor = session.runner()
                    .feed("input", inputTensor)
                    .feed("phase_train", phaseTrainTensor)  // Feed the phase_train placeholder
                    .fetch("embeddings")
                    .run()
                    .get(0);

            float[] embedding = new float[128];
            outputTensor.asRawTensor().data().asFloats().read(embedding);
            return embedding;
        }

    }


    @Override
    public void close() {
        session.close();
        graph.close();
    }
}

// Main.java
public class Main {
    public static void main(String[] args) {
        String cascadePath = "haarcascade_frontalface_default.xml";
        String modelPath = "facenet_model.pb";
        String databasePath = "face_database.csv";

        FaceRecognition faceRecognition = new FaceRecognition(
                cascadePath,
                modelPath,
                databasePath
        );

        faceRecognition.startVideoRecognition();
    }
}