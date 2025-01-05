package utils;

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
}
