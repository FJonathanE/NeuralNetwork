package de.jonathanebeling.neuralnetwork.main;

import de.jonathanebeling.neuralnetwork.network.NeuralNetwork;
import de.jonathanebeling.neuralnetwork.data.TrainingDataManager;
import de.jonathanebeling.neuralnetwork.activationFunctions.ReLuActivation;
import de.jonathanebeling.neuralnetwork.costFunctions.SumOfSquaredErrorsCost;
import de.jonathanebeling.neuralnetwork.utils.DisplayHelper;

public class Main {


    private static final String TRAINING_DATA_PATH = "src/data/train-images.idx3-ubyte";
    private static final String TRAINING_LABEL_PATH = "src/data/train-labels.idx1-ubyte";
    private static final String TEST_DATA_PATH = "src/data/t10k-images.idx3-ubyte";
    private static final String TEST_LABEL_PATH = "src/data/t10k-labels.idx1-ubyte";


    public static void main(String[] args) {
        DisplayHelper printer = new DisplayHelper();


        // Trainingsdaten einrichten

        TrainingDataManager dataManager = TrainingDataManager.fromMnistData(0.05,
                TRAINING_DATA_PATH, TRAINING_LABEL_PATH, TEST_DATA_PATH, TEST_LABEL_PATH);


        dataManager.setShuffleTrainingsData(true);
        dataManager.setTrainingDataNoiseFactor(0.1);

//        dataManager.setMaxRandomTrainingDataRotationAngle(10);
//        dataManager.setMaxRandomTrainingDataTranslation(2);




        // Neuronales Netz einrichten

        int[] numLayers = {784, 100, 100, 10};

        NeuralNetwork network = new NeuralNetwork(numLayers, new ReLuActivation(), new SumOfSquaredErrorsCost(),
                "networks/temporary/test");



        // Neuronales Netz trainieren

        network.trainMiniBatchAsync(dataManager, 0.01, 50, 20);



        // Neuronales Netz an Testdaten testen

        network.test(dataManager.getTestData());


    }
}
