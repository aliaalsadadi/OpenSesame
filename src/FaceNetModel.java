//import com.google.protobuf.InvalidProtocolBufferException;
//import org.opencv.core.Core;
//import org.opencv.core.CvType;
//import org.opencv.core.Mat;
//import org.opencv.core.Scalar;
//import org.opencv.imgproc.Imgproc;
//import org.tensorflow.Graph;
//import org.tensorflow.Session;
//import org.tensorflow.Tensor;
//import org.tensorflow.TensorFlow;
//import org.tensorflow.proto.framework.GraphDef;
//
//import java.io.IOException;
//import java.nio.file.Files;
//import java.nio.file.Path;
//import java.nio.file.Paths;
//
//
//public class FaceNetModel {
//
//    private Graph graph;
//    private Session session;
//    private String modelPath;
//
//    public FaceNetModel(String modelPath) {
//        this.modelPath = modelPath;
//    }
//
//    public void loadModel() throws InvalidProtocolBufferException {
//        graph = new Graph();
//        byte[] modelBytes = readAllBytesOrExit(Paths.get(modelPath));
//        graph.importGraphDef(GraphDef.parseFrom(modelBytes));
//        session = new Session(graph);
//    }
//
//    public float[] getFaceEmbedding(Mat face) {
//        float[] embedding = null;
//        try (Tensor tensor = normalizeImage(face)) {
//            Tensor output = session.runner()
//                    .feed("input_1", tensor)
//                    .fetch("Bottleneck_BatchNorm/batchnorm/add_1")
//                    .run()
//                    .get(0)
//                    .expect(Float.class);
//            embedding = new float[(int) output.shape()[1]];
//            output.copyTo(embedding);
//        } catch (Exception e) {
//            e.printStackTrace();
//        }
//        return embedding;
//    }
//
//    private Tensor normalizeImage(Mat mat) {
//        Imgproc.cvtColor(mat, mat, Imgproc.COLOR_BGR2RGB);
//        mat.convertTo(mat, CvType.CV_32F);
//        Core.divide(mat, Scalar.all(255.0f), mat);
//        Tensor.of(mat, mat.reshape(1, 160));
//        return Tensor.create(mat.reshape(1, 160, 160, 3));
//    }
//
//    private static byte[] readAllBytesOrExit(Path path) {
//        try {
//            return Files.readAllBytes(path);
//        } catch (IOException e) {
//            e.printStackTrace();
//            System.exit(-1);
//        } catch (IOException e) {
//            throw new RuntimeException(e);
//        }
//        return null;
//    }
//
//}