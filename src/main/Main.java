package main;

import network.DataPoint;
import network.NeuralNetwork;
import network.activationFunctions.SigmoidActivation;
import utils.TrainingDataManager;
import network.activationFunctions.ReLuActivation;
import network.costFunctions.SumOfSquaredErrorsCost;
import network.neuralNetworkTrainer.BackPropagationTraining;
import utils.DisplayHelper;
import utils.MnistDataReader;

import java.io.IOException;
import java.util.Random;

public class Main {


    private static final String trainingDataPath = "src/data/train-images.idx3-ubyte";
    private static final String trainingDataLabelPath = "src/data/train-labels.idx1-ubyte";
    private static final String testDataPath = "src/data/t10k-images.idx3-ubyte";
    private static final String testDataLabelPath = "src/data/t10k-labels.idx1-ubyte";


    public static void main(String[] args) throws IOException, ClassNotFoundException {
        DisplayHelper printer = new DisplayHelper();
//

        TrainingDataManager dataManager = new TrainingDataManager(getDataPoints(trainingDataPath, trainingDataLabelPath), 0.05);
        dataManager.setShuffleTrainingsData(true);
        dataManager.setMaxRandomTrainingDataRotationAnlge(20);
        dataManager.setMaxRandomTrainingDataTranslation(2);
        dataManager.setTrainingDataNoiseFactor(0.1);
        dataManager.resetTrainingsData();


        NeuralNetwork network = train(dataManager);
        test(network);




//            NeuralNetwork network1 = NeuralNetwork.load("src/saved_networks/relu-better-generalisation-1/epoch-39.ser");
//            test(network1);
//            NeuralNetwork network2 = NeuralNetwork.load("src/saved_networks/network-14.ser");
//            test(network2);





//        NeuralNetwork network1 = NeuralNetwork.load("src/saved_networks/network9-233398651.ser");
//        test(network1);
//        NeuralNetwork network2 = NeuralNetwork.load("src/saved_networks/network8-587897618.ser");
//        test(network2);


    }

    private static DataPoint[] getDataPoints(String dataPath, String labelPath) {
        MnistDataReader reader = new MnistDataReader();
        DataPoint[] data;
        try {
            data = reader.readData(dataPath, labelPath);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        return data;
    }


    private static void test(NeuralNetwork network) {
        DataPoint[] testData = getDataPoints(testDataPath, testDataLabelPath);


        double error = network.dataPointsCost(testData);

        DisplayHelper printer = new DisplayHelper();

        System.out.println(" ");
        System.out.println("Total error rate of test set: " + error);
        printer.printNumberOfWrongs(network, testData);



    }


    private static NeuralNetwork train(TrainingDataManager dataManager) {
        int[] numLayers = {784, 300, 200, 200, 10};
        NeuralNetwork network = new NeuralNetwork(numLayers, new ReLuActivation(), new SumOfSquaredErrorsCost(), "networks/temporary/big-relu");

        BackPropagationTraining propagationTraining = new BackPropagationTraining();
        propagationTraining.setEarlyStopping(true);
        propagationTraining.setEarlyStoppingPatience(5);

        propagationTraining.trainMiniBatchAsync(network, dataManager, 0.02, 200, 200);




//        propagationTraining.train(dataManager.getTrainingData(false), 0.5, network);
//        test(network);



        return network;
    }

//    private static NeuralNetwork train(TrainingDataManager dataManager) {
//        int[] numLayers = {784, 40, 10};
//        NeuralNetwork network = new NeuralNetwork(numLayers, new SigmoidActivation(), new SumOfSquaredErrorsCost());
//
//        BackPropagationTraining propagationTraining = new BackPropagationTraining();
//
//        propagationTraining.trainMiniBatch(network, dataManager, 2, 64, 20);
//
//        return network;
//    }

}
