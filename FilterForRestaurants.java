import com.google.gson.Gson;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashSet;
import java.util.Scanner;
import java.util.Set;

public class FilterForRestaurants { 
    private static final String OUTPUT_FILE = "restaurant_reviews.txt";
    private static final String NEWLINE = System.getProperty("line.separator");
    private String[] reviewFiles = {"xaa", "xab", "xac", "xad", "xae", "xaf", "xag"};
    private Set<String> idList = new HashSet<>();

    public static void main(String[] args) {
        new FilterForRestaurants().run(args[0], args[1]);
    }

    void run(String reviewFilesParentPath, String restaurantIdFile) {
        try {
            FileWriter writer = new FileWriter(new File(OUTPUT_FILE));
            Scanner scannerID = new Scanner(new File(restaurantIdFile));

            while (scannerID.hasNext()) {
                idList.add(scannerID.nextLine().trim());
            }
            scannerID.close();

            for (String reviewFile : reviewFiles) {
                Scanner scanner = new Scanner(new File(reviewFilesParentPath + "\\" + reviewFile));
                scanner.nextLine();
                while (scanner.hasNext()) {
                    String str = scanner.nextLine();
                    if (!scanner.hasNext()) {
                        break;
                    }
                    Gson g = new Gson();
                    Review review = g.fromJson(str, Review.class);

                    if (idList.contains(review.getBusiness_id()) && !Utilities.hasFrenchWord(review.getText())) {
                        String textNormalized = normalize(review.getText());
                        review.setText(textNormalized);

                        writer.write("\"" + review.getText().trim() + "\"\t" + review.getStars().toString().trim() + NEWLINE);
                    }
                }
                scanner.close();
            }
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private String normalize(String str) {
        String textNormalized = str.replaceAll("\\n",". ");
        textNormalized = textNormalized.replaceAll("\\t", " ");
        return textNormalized;
    }
}
