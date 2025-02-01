package de.jonathanebeling.neuralnetwork.network;

import de.jonathanebeling.neuralnetwork.activationFunctions.ActivationFunction;
import de.jonathanebeling.neuralnetwork.costFunctions.CostFunction;
import de.jonathanebeling.neuralnetwork.data.DataPoint;
import de.jonathanebeling.neuralnetwork.data.TrainingDataManager;
import de.jonathanebeling.neuralnetwork.utils.MathUtils;

import java.io.*;
import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class NeuralNetwork implements Serializable {
    @Serial
    private static final long serialVersionUID = 1L;

    private final Layer[] layers;
    private final int[] numLayers;
    private final ActivationFunction activationFunction;
    private final CostFunction costFunction;

    private String saveDirectoryPath;

    private int trainedEpochs = 0;

    private boolean earlyStopping = true;
    private int earlyStoppingPatience = 5;
    private int patienceCounter = 0;
    private double minValidationError = Double.MAX_VALUE;


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
            System.out.println("Saved trained neural network of epoch " + epoch + " to file path: " + saveDirectoryPath + "/epoch-" + epoch + ".ser");
        }
    }

    public static NeuralNetwork load(String filePath) throws IOException, ClassNotFoundException {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filePath))) {
            return (NeuralNetwork) ois.readObject();
        }
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


    public double train(DataPoint[] dataPoints, double learningRate) {
        NetworkTrainingInstance networkTrainingInstance = new NetworkTrainingInstance(this);

        double totalError = 0;

        for (DataPoint datapoint : dataPoints) {
            totalError += networkTrainingInstance.trainOnDatapoint(datapoint);
        }

        networkTrainingInstance.applyAllGradients(learningRate / dataPoints.length);

        return totalError / dataPoints.length;
    }




    public void trainMiniBatchAsync(TrainingDataManager dataManager, double learningRate, int minibatchSize, int epochs) {

        minValidationError = Double.MAX_VALUE;
        patienceCounter = 0;

        DataPoint[] trainingData = dataManager.getTrainingData();
        DataPoint[] validationData = dataManager.getValidationData();

        if (!dataManager.hasValidationData() && earlyStopping)
            System.out.println("Can't use early stopping if there is no validation data set!");

        System.out.println("===== BEGINNING TRAINING OF NEURAL NETWORK =====");
        System.out.println("Training data points: " + trainingData.length);
        System.out.println("Cross validation data points: " + validationData.length);
        System.out.println("Learning rate: " + learningRate);
        System.out.println("Mini Batch Size: " + minibatchSize);
        System.out.println("Epochs: " + epochs);

        // Thread-Pool erstellen
        int availableProcessors = Runtime.getRuntime().availableProcessors();
        try (ExecutorService executor = Executors.newFixedThreadPool(availableProcessors)) {

            for (int epoch = 0; epoch < epochs; epoch++) {


                System.out.println("===== EPOCH " + trainedEpochs + " STARTING =====");

                dataManager.resetTrainingsData();
                trainingData = dataManager.getTrainingData();

                long startMillis = System.currentTimeMillis();
                int batchCount = trainingData.length / minibatchSize;
                double totalError = 0;

                try {
                    // Liste von Futures für parallele Berechnungen
                    Future<Double>[] futures = new Future[batchCount];

                    for (int batch = 0; batch < batchCount; batch++) {
                        int startPos = batch * minibatchSize;
                        int endPos = Math.min((batch + 1) * minibatchSize, trainingData.length);

                        DataPoint[] batchData = Arrays.copyOfRange(trainingData, startPos, endPos);


                        // Übergebe die Berechnung an einen Thread
                        futures[batch] = executor.submit(() -> train(batchData, learningRate));


                    }

                    // Sammle die Ergebnisse aus den Threads
                    for (Future<Double> future : futures) {
                        totalError += future.get(); // Blockiert, bis das Ergebnis verfügbar ist
                    }


                } catch (Exception e) {
                    e.printStackTrace();
                }

                System.out.println("Finished in " + (System.currentTimeMillis() - startMillis) + "ms ");
                System.out.println("Training error rate of epoch: " + MathUtils.roundDecimalPoints(totalError / batchCount, 4));






                boolean stop = validateEpochTraining(dataManager);
                trainedEpochs += 1;
                if (stop) {
                    break;
                }

            }

            // Executor-Service herunterfahren
            executor.shutdown();
        }
    }

    private boolean validateEpochTraining(TrainingDataManager dataManager){
        if (dataManager.hasValidationData()) {
                    double crossValidationError = dataPointsCost(dataManager.getValidationData());
                    System.out.println("Validation error of epoch: " + MathUtils.roundDecimalPoints(crossValidationError, 4));

                    if (crossValidationError < minValidationError) {
                        minValidationError = crossValidationError;
                        patienceCounter = 0;

                        System.out.println("Lowest cross validation error until now: " + minValidationError);
                        try {
                            System.out.println("Saving neural network of this epoch due to lowest cross validation error rate");
                            save(trainedEpochs);
                        } catch (IOException e) {
                            throw new RuntimeException(e);
                        }
                    } else {
                        patienceCounter++;
                        System.out.println("Not beaten lowest cross validation error since " + patienceCounter + " epochs!");
                    }

                    if (patienceCounter >= earlyStoppingPatience) {
                        System.out.println("Stopping training in epoch " + trainedEpochs + " due to lack of cross validation error rate improvement!");
                        return true;
                    }
                }

        System.out.println(" ");

        return false;
    }
}
