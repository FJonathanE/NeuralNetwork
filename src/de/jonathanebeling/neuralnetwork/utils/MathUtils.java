package de.jonathanebeling.neuralnetwork.utils;

import java.util.Arrays;
import java.util.Comparator;

public class MathUtils {

    public static double roundDecimalPoints(double input, int decimalPoints){
        double i = Math.pow(10, decimalPoints);
        return Math.round(input * i) / i;
    }

    public static int[] getSortedIndices(double[] array) {
        // Paare von Index und Wert erstellen
        Integer[] indices = new Integer[array.length];
        for (int i = 0; i < array.length; i++) {
            indices[i] = i;
        }

        // Sortiere Indizes basierend auf den Werten im Array
        Arrays.sort(indices, new Comparator<Integer>() {
            @Override
            public int compare(Integer i1, Integer i2) {
                return Double.compare(array[i2], array[i1]); // Absteigend sortieren
            }
        });

        // Indizes als int[] zurückgeben
        return Arrays.stream(indices).mapToInt(Integer::intValue).toArray();
    }

    /**
     * Converts a flat array to a 2D array.
     *
     * @param flatArray The flat array to convert.
     * @param rows The number of rows in the 2D array.
     * @param cols The number of columns in the 2D array.
     * @return The 2D array.
     */
    public static double[][] from1dTo2d(double[] flatArray, int rows, int cols) {
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
    public static double[] from2dTo1d(double[][] array2D) {
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


    public static double[] multiplyElementwise(double[] array1, double[] array2){
        if (array1.length != array2.length) throw new IllegalArgumentException("Arrays have to be of same length");

        double[] output = new double[array1.length];

        for (int index = 0; index < array1.length; index++) {
            output[index] = array1[index] * array2[index];
        }

        return output;
    }

    public static double sum(double[] array){
        double output = 0;

        for (double d : array) {
            output += d;
        }

        return output;
    }

    public static double sumMultipliedArrays(double[] array1, double[] array2){
        if (array1.length != array2.length) throw new IllegalArgumentException("Arrays have to be of same length");

        double output = 0;

        for (int index = 0; index < array1.length; index++) {
            output += array1[index] * array2[index];
        }

        return output;
    }

    
}
