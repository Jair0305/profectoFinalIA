import java.util.Random;

public class ActivationFunction {
    public static double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    public static double sigmoidDerivative(double x) {
        return x * (1 - x);
    }
}

class Neuron {
    private double[] weights;
    private double output;
    private double delta;

    public double getWeight(int index) {
        if (index >= 0 && index < weights.length) {
            return weights[index];
        } else {
            throw new ArrayIndexOutOfBoundsException("Index " + index + " out of bounds for length " + weights.length);
        }
    }

    public Neuron(int inputSize) {
        weights = new double[inputSize + 1]; // +1 for bias
        Random rand = new Random();
        for (int i = 0; i < weights.length; i++) {
            weights[i] = rand.nextDouble() - 0.5;
        }
    }

    public double feedForward(double[] inputs) {
        double sum = weights[weights.length - 1]; // bias
        for (int i = 0; i < inputs.length; i++) {
            sum += weights[i] * inputs[i];
        }
        output = ActivationFunction.sigmoid(sum);
        return output;
    }

    public void updateWeights(double[] inputs, double learningRate) {
        for (int i = 0; i < inputs.length; i++) {
            weights[i] += learningRate * delta * inputs[i];
        }
        weights[weights.length - 1] += learningRate * delta; // update bias
    }

    public double getOutput() {
        return output;
    }

    public void setDelta(double delta) {
        this.delta = delta;
    }

    public double getDelta() {
        return delta;
    }

    public double[] getWeights() {
        return weights;
    }
}

class Layer {
    private Neuron[] neurons;

    public Layer(int numberOfNeurons, int inputSize) {
        neurons = new Neuron[numberOfNeurons];
        for (int i = 0; i < numberOfNeurons; i++) {
            neurons[i] = new Neuron(inputSize);
        }
    }

    public double[] feedForward(double[] inputs) {
        double[] outputs = new double[neurons.length];
        for (int i = 0; i < neurons.length; i++) {
            outputs[i] = neurons[i].feedForward(inputs);
        }
        return outputs;
    }



    public Neuron[] getNeurons() {
        return neurons;
    }
}

