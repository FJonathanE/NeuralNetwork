package network.neuralNetworkTrainer;

import network.DataPoint;
import network.Layer;
import network.NetworkTrainingInstance;
import network.NeuralNetwork;
import utils.TrainingDataManager;
import utils.MathUtils;

import java.io.IOException;
import java.util.Arrays;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public abstract class TrainingAlgorithm {

    private boolean earlyStopping = true;
    private int earlyStoppingPatience = 5;
    private int patienceCounter = 0;
    private double minValidationError = Double.MAX_VALUE;

    abstract double updateAllGradients(DataPoint datapoint, NetworkTrainingInstance network);

    public double train(DataPoint[] dataPoints, double learningRate, NeuralNetwork network) {
        NetworkTrainingInstance networkTrainingInstance = new NetworkTrainingInstance(network);


        double totalError = 0;

        for (DataPoint datapoint : dataPoints) {
            totalError += updateAllGradients(datapoint, networkTrainingInstance);
        }

        networkTrainingInstance.applyAllGradients(learningRate / dataPoints.length);

        return totalError / dataPoints.length;
    }


    /**
     * Trains neural network on training data in mini batches
     *
     * @param dataManager   Training data manager for training data
     * @param learningRate  Learning rate
     * @param minibatchSize Size of mini batches
     * @param epochs        Amount of epochs the neural network should be trained on
     */

//    public void trainMiniBatch(NeuralNetwork network, TrainingDataManager dataManager, double learningRate, int minibatchSize, int epochs) {
//
//        DataPoint[] trainingData = dataManager.getTrainingData(true);
//        DataPoint[] validationData = dataManager.getValidationData(false);
//
//        if (!dataManager.hasValidationData() && earlyStopping)
//            System.out.println("Can't use early stopping if there is no validation data set!");
//
//        System.out.println("===== BEGINNING TRAINING OF NEURAL NETWORK =====");
//
//        System.out.println("Training data points: " + trainingData.length);
//        System.out.println("Cross validation data points: " + validationData.length);
//        System.out.println("Learning rate: " + learningRate);
//        System.out.println("Mini Batch Size: " + minibatchSize);
//        System.out.println("Epochs: " + epochs);
//
//
//        if (dataManager.hasValidationData()) {
//
//            System.out.println("Using cross validation.");
//            System.out.println("Training data points: " + trainingData.length);
//            System.out.println("Cross validation data points: " + validationData.length);
//
//        } else {
//            System.out.println("Training data points: " + validationData.length);
//        }
//
//        System.out.println(" ");
//
//        for (int epoch = 0; epoch < epochs; epoch++) {
//
//            System.out.println("===== EPOCH " + (epoch + 1) + " STARTING =====");
//
//            trainingData = dataManager.getTrainingData(true);
//
//            long startMillis = System.currentTimeMillis();
//            int batchCount = trainingData.length / minibatchSize;
//            double totalError = 0;
//
//            for (int batch = 0; batch < batchCount; batch++) {
//                int startPos = batch * minibatchSize;
//                int endPos = (batch + 1) * minibatchSize;
//
//                DataPoint[] batch_data = Arrays.copyOfRange(trainingData, startPos, endPos);
//                double error = train(batch_data, learningRate, network);
//                totalError += error;
//            }
//
//            System.out.println("Finished in " + (System.currentTimeMillis() - startMillis) + "ms ");
//            System.out.println("Training error rate of epoch: " + MathUtils.roundDecimalPoints(totalError / batchCount, 4));
//
//            try {
//                System.out.println("Saving network of this epoch");
//                network.save(epoch);
//            } catch (IOException e) {
//                throw new RuntimeException(e);
//            }
//
//            if (dataManager.hasValidationData()) {
//                double crossValidationError = network.dataPointsCost(validationData);
//                System.out.println("Validation error of epoch: " + MathUtils.roundDecimalPoints(crossValidationError, 4));
//                if (shouldStop(crossValidationError)) {
//                    System.out.println("Stopping training in epoch " + (epoch + 1) + " due to lack of cross validation error rate improvement!");
//                    break;
//                }
//            }
//
//            System.out.println(" ");
//
//
//        }
//
//    }


    public void trainMiniBatchAsync(NeuralNetwork network, TrainingDataManager dataManager, double learningRate, int minibatchSize, int epochs) {

        DataPoint[] trainingData = dataManager.getTrainingData(true);
        DataPoint[] validationData = dataManager.getValidationData(false);

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

                System.out.println("===== EPOCH " + epoch + " STARTING =====");

                trainingData = dataManager.getTrainingData(true);

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
                        futures[batch] = executor.submit(() -> train(batchData, learningRate, network));


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






                if (dataManager.hasValidationData()) {
                    double crossValidationError = network.dataPointsCost(validationData);
                    System.out.println("Validation error of epoch: " + MathUtils.roundDecimalPoints(crossValidationError, 4));

                    if (crossValidationError < minValidationError) {
                        minValidationError = crossValidationError;
                        patienceCounter = 0;

                        System.out.println("Lowest cross validation error until now: " + minValidationError);
                        try {
                            System.out.println("Saving network of this epoch due to lowest cross validation error rate");
                            network.save(epoch);
                        } catch (IOException e) {
                            throw new RuntimeException(e);
                        }
                    } else {
                        patienceCounter++;
                        System.out.println("Not beaten lowest cross validation error since " + patienceCounter + " epochs!");
                    }

                    if (patienceCounter >= earlyStoppingPatience) {
                        System.out.println("Stopping training in epoch " + (epoch + 1) + " due to lack of cross validation error rate improvement!");
                        break;
                    }
                }

                System.out.println(" ");
            }

            // Executor-Service herunterfahren
            executor.shutdown();
        }
    }


    public boolean isEarlyStopping() {
        return earlyStopping;
    }

    public void setEarlyStopping(boolean earlyStopping) {
        this.earlyStopping = earlyStopping;
    }

    public int getEarlyStoppingPatience() {
        return earlyStoppingPatience;
    }

    public void setEarlyStoppingPatience(int earlyStoppingPatience) {
        this.earlyStoppingPatience = earlyStoppingPatience;
    }
}
