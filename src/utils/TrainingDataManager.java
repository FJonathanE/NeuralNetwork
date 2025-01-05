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
//        totalData = shuffleData(totalData);
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

    /**
     * Rotates the training data input activations by a random angle within the specified range.
     *
     * @param maxRotationAngle The maximum angle (in degrees) for rotation. Rotations will be in the range [-maxRotationAngle, maxRotationAngle].
     */
    public void rotateTrainingData(double maxRotationAngle) {
        System.out.println("Rotating training data with a maximum rotation angle of: " + maxRotationAngle + " degrees");

        Random random = new Random();
        double maxAngleRadians = Math.toRadians(maxRotationAngle);

        for (DataPoint dataPoint : trainingData) {
            double[] inputActivation = dataPoint.getInputActivation();
            int size = (int) Math.sqrt(inputActivation.length); // Assuming input data is square (e.g., 28x28).

            if (size * size != inputActivation.length) {
                throw new IllegalArgumentException("Input activation data is not square.");
            }

            // Convert the flat array to a 2D array for easier rotation.
            double[][] image = flatTo2DArray(inputActivation, size, size);

            // Generate a random angle in radians.
            double angle = (random.nextDouble() * 2 - 1) * maxAngleRadians;

            // Rotate the image and flatten it back.
            double[][] rotatedImage = rotateImage(image, angle);
            double[] rotatedInputActivation = flatten2DArray(rotatedImage);

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
    public void translateTrainingData(int maxTranslation) {
        System.out.println("Translating training data with a maximum translation of: " + maxTranslation + " pixels");

        Random random = new Random();

        int debug = 0;

        for (DataPoint dataPoint : trainingData) {
            double[] inputActivation = dataPoint.getInputActivation();
            int size = (int) Math.sqrt(inputActivation.length); // Assuming input data is square (e.g., 28x28).

            if (size * size != inputActivation.length) {
                throw new IllegalArgumentException("Input activation data is not square.");
            }

            // Convert the flat array to a 2D array for easier manipulation.
            double[][] image = flatTo2DArray(inputActivation, size, size);

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
            double[] translatedInputActivation = flatten2DArray(translatedImage);

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

    /**
     * Converts a flat array to a 2D array.
     *
     * @param flatArray The flat array to convert.
     * @param rows The number of rows in the 2D array.
     * @param cols The number of columns in the 2D array.
     * @return The 2D array.
     */
    public double[][] flatTo2DArray(double[] flatArray, int rows, int cols) {
        double[][] array2D = new double[rows][cols];
        for (int i = 0; i < flatArray.length; i++) {
            array2D[i / cols][i % cols] = flatArray[i];
        }
        return array2D;
    }

    /**
     * Flattens a 2D array to a flat array.
     *
     * @param array2D The 2D array to flatten.
     * @return The flat array.
     */
    public double[] flatten2DArray(double[][] array2D) {
        int rows = array2D.length;
        int cols = array2D[0].length;
        double[] flatArray = new double[rows * cols];
        for (int y = 0; y < rows; y++) {
            for (int x = 0; x < cols; x++) {
                flatArray[y * cols + x] = array2D[y][x];
            }
        }
        return flatArray;
    }
}
