package de.jonathanebeling.neuralnetwork.main;

import de.jonathanebeling.neuralnetwork.data.DataPoint;
import de.jonathanebeling.neuralnetwork.network.NeuralNetwork;
import de.jonathanebeling.neuralnetwork.data.TrainingDataManager;
import de.jonathanebeling.neuralnetwork.activationFunctions.ReLuActivation;
import de.jonathanebeling.neuralnetwork.costFunctions.SumOfSquaredErrorsCost;
import de.jonathanebeling.neuralnetwork.utils.DisplayHelper;
import de.jonathanebeling.neuralnetwork.data.MnistDataReader;

import java.io.IOException;

public class Main {


    private static final String trainingDataPath = "src/data/train-images.idx3-ubyte";
    private static final String trainingDataLabelPath = "src/data/train-labels.idx1-ubyte";
    private static final String testDataPath = "src/data/t10k-images.idx3-ubyte";
    private static final String testDataLabelPath = "src/data/t10k-labels.idx1-ubyte";


    public static void main(String[] args) throws IOException, ClassNotFoundException {
        DisplayHelper printer = new DisplayHelper();
//

        TrainingDataManager dataManager = new TrainingDataManager(getDataPoints(trainingDataPath, trainingDataLabelPath), 0.05);
        dataManager.setTestData(getDataPoints(testDataPath, testDataLabelPath));


        dataManager.setShuffleTrainingsData(true);
//        dataManager.setMaxRandomTrainingDataRotationAngle(10);
//        dataManager.setMaxRandomTrainingDataTranslation(2);
        dataManager.setTrainingDataNoiseFactor(0.1);
        dataManager.resetTrainingsData();


        NeuralNetwork network = train(dataManager);


        network.test(dataManager.getTestData());


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




    private static NeuralNetwork train(TrainingDataManager dataManager) {
        int[] numLayers = {784, 100, 100, 10};
        NeuralNetwork network = new NeuralNetwork(numLayers, new ReLuActivation(), new SumOfSquaredErrorsCost(), "networks/temporary/test");

        network.trainMiniBatchAsync(dataManager, 0.01, 50, 20);
        network.trainMiniBatchAsync(dataManager, 0.005, 50, 20);

        return network;
    }

}
