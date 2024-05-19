import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Main {
    static {
        try {
            System.load("C:/Users/chama/Downloads/opencv/build/java/x64/opencv_java490.dll");
        } catch (UnsatisfiedLinkError e) {
            System.err.println("Native code library failed to load.\n" + e);
            System.exit(1);
        }
    }

    public static void main(String[] args) throws IOException, ClassNotFoundException {
        // Cargar las imágenes y etiquetas
        List<ImageData> imageDataList = DataLoader.loadImages();

        // Verificar que se hayan cargado las imágenes
        if (imageDataList.isEmpty()) {
            throw new RuntimeException("No se han cargado imágenes.");
        }

        // Convertir las imágenes y las etiquetas a arrays
        double[][] images = new double[imageDataList.size()][];
        double[][] labels = new double[imageDataList.size()][];
        for (int i = 0; i < imageDataList.size(); i++) {
            ImageData imageData = imageDataList.get(i);
            Mat image = imageData.getImage();
            images[i] = matToDoubleArray(image);
            labels[i] = labelToOneHot(imageData.getLabel());
        }

        // Verificar que las etiquetas estén correctamente convertidas
        for (double[] label : labels) {
            if (label.length != labelToIndex.size()) {
                throw new IllegalArgumentException("One-hot label size does not match number of classes.");
            }
        }

        // Crear la red neuronal
        int[] layerSizes = {4096, 100, labelToIndex.size()}; // Ajustar según las necesidades
        NeuralNetwork nn = new NeuralNetwork(layerSizes);

        // Definir el tamaño del lote
        int batchSize = 32;

        // Entrenar la red neuronal
        nn.train(images, labels, 100 /* Número de épocas */, 0.1 /* Tasa de aprendizaje */, batchSize);

// Guardar la red neuronal
        nn.save("neural_network.ser");

// Cargar la red neuronal
        NeuralNetwork loadedNN = NeuralNetwork.load("neural_network.ser");
    }

    private static double[] matToDoubleArray(Mat mat) {
        Mat convertedMat = new Mat();
        mat.convertTo(convertedMat, CvType.CV_64F);
        int size = (int) convertedMat.total() * convertedMat.channels();
        double[] array = new double[size];
        convertedMat.get(0, 0, array);
        return array;
    }

    private static Map<String, Integer> labelToIndex = new HashMap<>();

    static {
        String[] animals = {
                "antelope", "badger", "bat", "bear", "bee", "beetle", "bison", "boar", "butterfly", "cat",
                "caterpillar", "chimpanzee", "cockroach", "cow", "coyote", "crab", "crow", "deer", "dog",
                "dolphin", "donkey", "dragonfly", "duck", "eagle", "elephant", "flamingo", "fly", "fox",
                "goat", "goldfish", "goose", "gorilla", "grasshopper", "hamster", "hare", "hedgehog",
                "hippopotamus", "hornbill", "horse", "hummingbird", "hyena", "jellyfish", "kangaroo", "koala",
                "ladybugs", "leopard", "lion", "lizard", "lobster", "mosquito", "moth", "mouse", "octopus",
                "okapi", "orangutan", "otter", "owl", "ox", "oyster", "panda", "parrot", "pelecaniformes",
                "penguin", "pig", "pigeon", "porcupine", "possum", "raccoon", "rat", "reindeer", "rhinoceros",
                "sandpiper", "seahorse", "seal", "shark", "sheep", "snake", "sparrow", "squid", "squirrel",
                "starfish", "swan", "tiger", "turkey", "turtle", "whale", "wolf", "wombat", "woodpecker", "zebra"
        };

        for (int i = 0; i < animals.length; i++) {
            labelToIndex.put(animals[i], i);
        }
    }

    // ...

    private static double[] labelToOneHot(String label) {
        double[] oneHot = new double[labelToIndex.size()];
        int index = labelToIndex.get(label);
        oneHot[index] = 1.0;
        return oneHot;
    }
}
