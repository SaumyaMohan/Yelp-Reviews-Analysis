
public class Utilities {
    private Utilities() {
    }

    private static String[] frenchWords = {" bien ", " aliments ", " mal ", " ne pas "};

    public static boolean hasFrenchWord(String text) {
        String textLowerCase = text.toLowerCase();

        for (String frenchWord : frenchWords) {
            if (textLowerCase.contains(frenchWord)) {
                return true;
            }
        }
        return false;
    }
}