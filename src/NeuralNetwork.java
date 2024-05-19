import java.io.*;

public class NeuralNetwork  implements Serializable {
    private Layer[] layers;
    private double[] lastInputs;

    public NeuralNetwork(int[] layerSizes) {
        layers = new Layer[layerSizes.length];
        for (int i = 0; i < layerSizes.length; i++) {
            int inputSize = i == 0 ? layerSizes[i] : layerSizes[i - 1];
            layers[i] = new Layer(layerSizes[i], inputSize);
        }
    }

    public double[] feedForward(double[] inputs) {
        lastInputs = inputs.clone(); // Store a copy of the inputs
        double[] outputs = inputs;
        for (Layer layer : layers) {
            outputs = layer.feedForward(outputs);
        }
        return outputs;
    }

    public void backPropagate(double[] expectedOutput, double learningRate) {
        for (int i = layers.length - 1; i >= 0; i--) {
            Layer layer = layers[i];
            double[] errors = new double[layer.getNeurons().length];

            if (i == layers.length - 1) { // Output layer
                for (int j = 0; j < layer.getNeurons().length; j++) {
                    Neuron neuron = layer.getNeurons()[j];
                    errors[j] = expectedOutput[j] - neuron.getOutput();
                }
            } else { // Hidden layer
                Layer nextLayer = layers[i + 1];
                for (int j = 0; j < layer.getNeurons().length; j++) {
                    double error = 0.0;
                    for (Neuron nextNeuron : nextLayer.getNeurons()) {
                        error += nextNeuron.getDelta() * nextNeuron.getWeights()[j];
                    }
                    errors[j] = error;
                }
            }

            for (int j = 0; j < layer.getNeurons().length; j++) {
                Neuron neuron = layer.getNeurons()[j];
                neuron.setDelta(errors[j] * ActivationFunction.sigmoidDerivative(neuron.getOutput()));
            }
        }

        double[] inputs = layers[0].feedForward(lastInputs); // Input layer inputs (not used)
        for (int i = 0; i < layers.length; i++) {
            if (i != 0) {
                inputs = layers[i - 1].feedForward(inputs);
            }
            Layer layer = layers[i];
            for (Neuron neuron : layer.getNeurons()) {
                neuron.updateWeights(inputs, learningRate);
            }
        }
    }

    public void train(double[][] trainingInputs, double[][] trainingOutputs, int epochs, double learningRate, int batchSize) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            System.out.println("Starting epoch " + (epoch + 1) + " / " + epochs);
            for (int i = 0; i < trainingInputs.length; i += batchSize) {
                int end = Math.min(i + batchSize, trainingInputs.length);
                for (int j = i; j < end; j++) {
                    if (trainingOutputs[j].length != layers[layers.length - 1].getNeurons().length) {
                        throw new IllegalArgumentException("Expected output size does not match the number of output neurons.");
                    }
                    feedForward(trainingInputs[j]);
                    backPropagate(trainingOutputs[j], learningRate);
                }
                System.out.println("Processed batch " + (i / batchSize + 1) + " / " + (trainingInputs.length / batchSize + 1));
            }
            System.out.println("Completed epoch " + (epoch + 1));
        }
    }

    public void save(String filename) throws IOException {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filename))) {
            oos.writeObject(this);
        }
    }

    public static NeuralNetwork load(String filename) throws IOException, ClassNotFoundException {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filename))) {
            return (NeuralNetwork) ois.readObject();
        }
    }

}
