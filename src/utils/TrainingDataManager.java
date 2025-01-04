package utils;

import network.DataPoint;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class TrainingDataManager {

    private DataPoint[] totalData;

    private DataPoint[] trainingData;
    private DataPoint[] validationData;

    private double validationDataPercentage;
    private DataPoint[] testData;

    /**
     * @param totalData                DataPoints to be used for training and perhaps part of it for cross validation.
     * @param validationDataPercentage Percentage of data to be used not for training but for cross validation. Set as 0 for no cross validation. Must be lower than 1.
     */
    public TrainingDataManager(DataPoint[] totalData, double validationDataPercentage) {
        totalData = shuffleData(totalData);
        this.totalData = totalData;
        this.validationDataPercentage = validationDataPercentage;


        boolean crossValidating = false;

        if (validationDataPercentage > 0 && validationDataPercentage < 1) crossValidating = true;

        if (crossValidating) {


            int index = (int) ((1 - validationDataPercentage) * totalData.length);
            trainingData = Arrays.copyOfRange(totalData, 0, index);
            validationData = Arrays.copyOfRange(totalData, index, totalData.length);
        } else {
            trainingData = totalData;
            validationData = new DataPoint[0];
        }

        System.out.println("Training data points: " + trainingData.length);
        System.out.println("Cross validation data points: " + validationData.length);

    }


    /**
     *
     * @param noiseFactor Lower than 0.1 for small noise and bigger than 0.1 for big noise
     */
    public void addTrainingDataNoise(double noiseFactor){

        System.out.println("Adding noise to training data with a noise factor of: " + noiseFactor);

        Random random = new Random();

        for (DataPoint dataPoint : trainingData) {
            double[] inputActivation = dataPoint.getInputActivation();

            for (int i = 0; i < inputActivation.length; i++) {
                double noise = random.nextGaussian() * noiseFactor;
                inputActivation[i] = Math.min(1, Math.max(0, inputActivation[i] + noise));
            }

            dataPoint.setInputActivation(inputActivation);
        }


    }

    public void setTestData(DataPoint[] testData) {
        this.testData = testData;
    }

    public DataPoint[] shuffleData(DataPoint[] dataPoints) {
        List<DataPoint> list = Arrays.asList(dataPoints);
        Collections.shuffle(list);

        return list.toArray(new DataPoint[0]);
    }

    private DataPoint[] shuffleData(DataPoint[] dataPoints, boolean shuffle) {
        if (shuffle) return shuffleData(dataPoints);
        else return dataPoints;
    }


    public DataPoint[] getTrainingData(boolean shuffle) {
        return shuffleData(trainingData, shuffle);
    }

    public DataPoint[] getValidationData(boolean shuffle) {
        return shuffleData(validationData, shuffle);
    }

    public DataPoint[] getTestData(boolean shuffle) {
        return shuffleData(testData, shuffle);
    }

    public boolean hasValidationData() {
        return (validationData.length != 0);
    }
}
