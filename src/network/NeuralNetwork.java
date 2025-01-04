package network;

import network.activationFunctions.ActivationFunction;
import network.costFunctions.CostFunction;

import java.io.*;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class NeuralNetwork implements Serializable {
    @Serial
    private static final long serialVersionUID = 1L;

    private final Layer[] layers;
    private final int[] numLayers;
    private final ActivationFunction activationFunction;
    private final CostFunction costFunction;

    private String saveDirectoryPath;


    public NeuralNetwork(int[] numLayers, ActivationFunction activationFunction, CostFunction costFunction, String saveDirectoryPath) {
        this.numLayers = numLayers;
        this.activationFunction = activationFunction;
        this.costFunction = costFunction;

        Random random = new Random();

        layers = new Layer[numLayers.length - 1];
        for (int i = 1; i < numLayers.length; i++) {
            layers[i - 1] = new Layer(numLayers[i - 1], numLayers[i], random, activationFunction, costFunction);
        }

        selectSaveDirectory(saveDirectoryPath);

    }

    private void selectSaveDirectory(String saveDirectoryPath) {
        int i = 0;


        while (new File(saveDirectoryPath + "-" + i).exists()){
            i++;
        }

        this.saveDirectoryPath = saveDirectoryPath + "-" + i;
        new File(this.saveDirectoryPath).mkdir();
    }

    public double[] calculateOutputs(double[] inputActivations) {
        for (Layer layer : layers) {
            inputActivations = layer.calculateOutputActivations(inputActivations);
        }
        return inputActivations;

    }

    public void clearAllGradients() {
        for (Layer layer : layers) {
            layer.initializeGradients();
        }
    }

    public double dataPointsCost(DataPoint[] data) {
        double cost = 0;

        for (DataPoint dataPoint : data) {
            cost += costFunction.dataPointCost(dataPoint, calculateOutputs(dataPoint.getInputActivation()));
        }

        return cost / data.length;
    }

    public void save(int epoch) throws IOException {

        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(saveDirectoryPath + "/epoch-" + epoch + ".ser"))) {
            oos.writeObject(this);
            System.out.println("Saved trained network of epoch " + epoch + " to file path: " + saveDirectoryPath + "/epoch-" + epoch + ".ser");
        }
    }

    public static NeuralNetwork load(String filePath) throws IOException, ClassNotFoundException {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filePath))) {
            return (NeuralNetwork) ois.readObject();
        }
    }


    public double[] calculateInputActivationByDesiredOutput(double[] desiredOutputActivation){
        for (int i = layers.length - 1; i >= 0; i--) {
            Layer layer = layers[i];


        }

        return new double[desiredOutputActivation.length];
    }



    public Layer[] getLayers() {
        return layers;
    }

    public int[] getNumLayers() {
        return numLayers;
    }

    public ActivationFunction getActivationFunction() {
        return activationFunction;
    }

    public CostFunction getCostFunction() {
        return costFunction;
    }
}
