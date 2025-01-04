package utils;

public class MathUtils {

    public static double roundDecimalPoints(double input, int decimalPoints){
        double i = Math.pow(10, decimalPoints);
        return Math.round(input * i) / i;
    }
}
