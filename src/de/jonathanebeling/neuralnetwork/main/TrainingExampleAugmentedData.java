package de.jonathanebeling.neuralnetwork.main;

import de.jonathanebeling.neuralnetwork.activation_functions.ReLuActivation;
import de.jonathanebeling.neuralnetwork.cost_functions.SumOfSquaredErrorsCost;
import de.jonathanebeling.neuralnetwork.data.TrainingDataManager;
import de.jonathanebeling.neuralnetwork.network.NeuralNetwork;

public class TrainingExampleAugmentedData {

    private static final String TRAINING_DATA_PATH = "data/train-images.idx3-ubyte";
    private static final String TRAINING_LABEL_PATH = "data/train-labels.idx1-ubyte";
    private static final String TEST_DATA_PATH = "data/t10k-images.idx3-ubyte";
    private static final String TEST_LABEL_PATH = "data/t10k-labels.idx1-ubyte";

    public static void main(String[] args) {
        trainingExampleAugmentedData();
    }

    private static void trainingExampleAugmentedData(){

        // TrainingDataManager mit Trainingsdaten einrichten
        TrainingDataManager dataManager = TrainingDataManager.fromMnistData(0.05,
                TRAINING_DATA_PATH, TRAINING_LABEL_PATH, TEST_DATA_PATH, TEST_LABEL_PATH);

        dataManager.setShuffleTrainingsData(true);
        dataManager.setTrainingDataNoiseFactor(0.2);
        dataManager.setMaxRandomTrainingDataRotationAngle(20);
        dataManager.setMaxRandomTrainingDataTranslation(3);


        int[] numLayers = {784, 200, 200, 10};

        // Neuronales Netz einrichten
        NeuralNetwork network = new NeuralNetwork(numLayers, new ReLuActivation(), new SumOfSquaredErrorsCost(),
                "networks/temporary/test");

        // Neuronales Netz trainieren
        network.trainMiniBatchAsync(dataManager, 0.01, 5, 3);


        // Neuronales Netz an Testdaten testen
        network.test(dataManager.getTestData());


    }
}
