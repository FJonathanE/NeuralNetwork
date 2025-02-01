package de.jonathanebeling.neuralnetwork.data;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static de.jonathanebeling.neuralnetwork.utils.MathUtils.from1dTo2d;
import static de.jonathanebeling.neuralnetwork.utils.MathUtils.from2dTo1d;

public class TrainingDataManager {

    private final DataPoint[] totalData;

    private final DataPoint[] unmodifiedTrainingData;
    private DataPoint[] trainingData;
    private DataPoint[] validationData;

    private final double validationDataPercentage;
    private DataPoint[] testData;

    private boolean shuffleTrainingsData = false;
    private double maxRandomTrainingDataRotationAngle = 0;
    private int maxRandomTrainingDataTranslation = 0;
    private double trainingDataNoiseFactor = 0;

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
            unmodifiedTrainingData = Arrays.copyOfRange(totalData, 0, index);
            validationData = Arrays.copyOfRange(totalData, index, totalData.length);
        } else {
            unmodifiedTrainingData = totalData;
            validationData = new DataPoint[0];
        }


        resetTrainingsData();
    }

    public void resetTrainingsData() {

        DataPoint[] dataPoints = shuffleData(unmodifiedTrainingData, shuffleTrainingsData);
        if (maxRandomTrainingDataRotationAngle != 0) rotateDataPoints(dataPoints, maxRandomTrainingDataRotationAngle);
        if (maxRandomTrainingDataTranslation != 0) translateDataPoints(dataPoints, maxRandomTrainingDataTranslation);
        if (trainingDataNoiseFactor != 0) addRandomNoise(dataPoints, trainingDataNoiseFactor);

        trainingData = dataPoints;
    }

    public void setTrainingDataNoiseFactor(double trainingDataNoiseFactor) {
        this.trainingDataNoiseFactor = trainingDataNoiseFactor;
    }

    public void setMaxRandomTrainingDataTranslation(int maxRandomTrainingDataTranslation) {
        this.maxRandomTrainingDataTranslation = maxRandomTrainingDataTranslation;
    }

    public void setMaxRandomTrainingDataRotationAngle(double maxRandomTrainingDataRotationAnlge) {
        this.maxRandomTrainingDataRotationAngle = maxRandomTrainingDataRotationAnlge;
    }

    public void setShuffleTrainingsData(boolean shuffleTrainingsData) {
        this.shuffleTrainingsData = shuffleTrainingsData;
    }

    /**
     *
     * @param noiseFactor Lower than 0.1 for small noise and bigger than 0.1 for big noise
     */
    private void addRandomNoise(DataPoint[] dataPoints, double noiseFactor){

        System.out.println("Adding noise to training data with a noise factor of: " + noiseFactor);

        int numThreads = Runtime.getRuntime().availableProcessors();
        try (ExecutorService executor = Executors.newFixedThreadPool(numThreads)) {

            int chunkSize = (int) Math.ceil((double) dataPoints.length / numThreads);
            for (int thread = 0; thread < numThreads; thread++) {

                int start = thread * chunkSize;
                int end = Math.min(start + chunkSize, dataPoints.length);

                executor.submit(() -> {
                    Random random = new Random();

                    for (int i = start; i < end; i++){
                        DataPoint dataPoint = dataPoints[i];

                        double[] inputActivation = dataPoint.getInputActivation();

                        for (int j = 0; j < inputActivation.length; j++) {
                            double noise = random.nextGaussian() * noiseFactor;
                            inputActivation[j] = Math.min(1, Math.max(0, inputActivation[j] + noise));
                        }

                        dataPoint.setInputActivation(inputActivation);
                    }
                });

            }

            executor.shutdown();
            executor.awaitTermination(Long.MAX_VALUE, TimeUnit.MILLISECONDS);
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }


    }




    public void setTestData(DataPoint[] testData) {
        this.testData = testData;
    }

    public DataPoint[] shuffleData(DataPoint[] dataPoints) {
        System.out.println("Shuffling data");

        List<DataPoint> list = Arrays.asList(dataPoints);
        Collections.shuffle(list);

        return list.toArray(new DataPoint[0]);
    }


    public DataPoint[] getTrainingData() {
        return trainingData.clone();
    }

    public DataPoint[] getTotalData() {
        return totalData;
    }

    public DataPoint[] getUnmodifiedTrainingData() {
        return unmodifiedTrainingData;
    }

    public double getValidationDataPercentage() {
        return validationDataPercentage;
    }

    public boolean isShuffleTrainingsData() {
        return shuffleTrainingsData;
    }

    public double getMaxRandomTrainingDataRotationAnlge() {
        return maxRandomTrainingDataRotationAngle;
    }

    public int getMaxRandomTrainingDataTranslation() {
        return maxRandomTrainingDataTranslation;
    }

    public double getTrainingDataNoiseFactor() {
        return trainingDataNoiseFactor;
    }

    private DataPoint[] shuffleData(DataPoint[] dataPoints, boolean shuffle){
        if (shuffle) return shuffleData(dataPoints);
        else return  dataPoints;
    }

    public DataPoint[] getValidationData() {
        return validationData.clone();
    }

    public DataPoint[] getTestData() {
        return testData.clone();
    }

    public boolean hasValidationData() {
        return (validationData.length != 0);
    }

    /**
     * Rotates the training data input activations by a random angle within the specified range.
     *
     * @param maxRotationAngle The maximum angle (in degrees) for rotation. Rotations will be in the range [-maxRotationAngle, maxRotationAngle].
     */
    private void rotateDataPoints(DataPoint[] dataPoints, double maxRotationAngle) {
        System.out.println("Rotating training data with a maximum rotation angle of: " + maxRotationAngle + " degrees");

        Random random = new Random();
        double maxAngleRadians = Math.toRadians(maxRotationAngle);

        for (DataPoint dataPoint : dataPoints) {
            double[] inputActivation = dataPoint.getInputActivation();
            int size = (int) Math.sqrt(inputActivation.length); // Assuming input data is square (e.g., 28x28).

            if (size * size != inputActivation.length) {
                throw new IllegalArgumentException("Input activation data is not square.");
            }

            // Convert the flat array to a 2D array for easier rotation.
            double[][] image = from1dTo2d(inputActivation, size, size);

            // Generate a random angle in radians.
            double angle = (random.nextDouble() * 2 - 1) * maxAngleRadians;

            // Rotate the image and flatten it back.
            double[][] rotatedImage = rotateImage(image, angle);
            double[] rotatedInputActivation = from2dTo1d(rotatedImage);

            // Update the DataPoint with the rotated activations.
            dataPoint.setInputActivation(rotatedInputActivation);
        }
    }

    /**
     * Rotates a 2D array (image) by the specified angle.
     *
     * @param image The 2D array representing the image.
     * @param angle The angle in radians to rotate the image.
     * @return The rotated 2D array.
     */
    private double[][] rotateImage(double[][] image, double angle) {
        int size = image.length;
        double[][] rotatedImage = new double[size][size];
        int centerX = size / 2;
        int centerY = size / 2;

        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) {
                // Calculate coordinates relative to the center.
                int relativeX = x - centerX;
                int relativeY = y - centerY;

                // Apply rotation transformation.
                int newX = (int) Math.round(relativeX * Math.cos(angle) - relativeY * Math.sin(angle)) + centerX;
                int newY = (int) Math.round(relativeX * Math.sin(angle) + relativeY * Math.cos(angle)) + centerY;

                // Check bounds and assign the value if valid.
                if (newX >= 0 && newX < size && newY >= 0 && newY < size) {
                    rotatedImage[y][x] = image[newY][newX];
                } else {
                    rotatedImage[y][x] = 0.0; // Fill out-of-bounds with a default value (e.g., 0).
                }
            }
        }

        return rotatedImage;
    }

    /**
     * Translates the training data input activations by a random number of pixels within the specified range.
     *
     * @param maxTranslation The maximum number of pixels to shift in any direction. Translations will be in the range [-maxTranslation, maxTranslation].
     */
    private void translateDataPoints(DataPoint[] dataPoints, int maxTranslation) {
        System.out.println("Translating training data with a maximum translation of: " + maxTranslation + " pixels");

        Random random = new Random();

        int debug = 0;

        for (DataPoint dataPoint : unmodifiedTrainingData) {
            double[] inputActivation = dataPoint.getInputActivation();
            int size = (int) Math.sqrt(inputActivation.length); // Assuming input data is square (e.g., 28x28).

            if (size * size != inputActivation.length) {
                throw new IllegalArgumentException("Input activation data is not square.");
            }

            // Convert the flat array to a 2D array for easier manipulation.
            double[][] image = from1dTo2d(inputActivation, size, size);

            // Generate random translations for x and y.
            int translateX = random.nextInt(2 * maxTranslation) - maxTranslation;
            int translateY = random.nextInt(2 * maxTranslation) - maxTranslation;


            // Translate the image and flatten it back.

            double[][] translatedImage = image;
            try {
                translatedImage = translateImage(image, translateX, translateY);
            }catch (RuntimeException exception){
                exception.printStackTrace();
            }
            double[] translatedInputActivation = from2dTo1d(translatedImage);

            // Update the DataPoint with the translated activations.
            dataPoint.setInputActivation(translatedInputActivation);


            debug++;
        }
    }

    /**
     * Translates a 2D array (image) by the specified number of pixels in x and y directions.
     *
     * @param image The 2D array representing the image.
     * @param translateX The number of pixels to shift in the x direction.
     * @param translateY The number of pixels to shift in the y direction.
     * @return The translated 2D array.
     */
    public double[][] translateImage(double[][] image, int translateX, int translateY) {
        int size = image.length;
        double[][] translatedImage = new double[size][size];


        for (int y = 0; y < size; y++) {

            for (int x = 0; x < size; x++) {

                int newX = x + translateX;
                int newY = y + translateY;


                // Check bounds and assign the value if valid.
                if (newX >= 0 && newX < size && newY >= 0 && newY < size) {
                    translatedImage[y][x] = image[newY][newX];
                } else {
                    translatedImage[y][x] = 0.0; // Fill out-of-bounds with a default value (e.g., 0).
                }
            }
        }

        return translatedImage;
    }

}
