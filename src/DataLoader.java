import org.opencv.core.Mat;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class DataLoader {
    private static final String DATASET_DIR = "C:/Users/chama/Downloads/ds/animals";

    public static List<ImageData> loadImages() throws IOException {
        List<ImageData> images = new ArrayList<>();
        Files.walk(Paths.get(DATASET_DIR))
                .filter(Files::isRegularFile)
                .forEach(path -> {
                    String label = path.getParent().getFileName().toString();
                    try {
                        Mat image = ImagePreprocessor.preprocessImage(path.toString());
                        images.add(new ImageData(image, label));
                        System.out.println("Loaded image from " + path + " with label " + label);
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                });
        System.out.println("Loaded " + images.size() + " images");
        return images;
    }
}

class ImageData {
    private Mat image;
    private String label;

    public ImageData(Mat image, String label) {
        this.image = image;
        this.label = label;
    }

    public Mat getImage() {
        return image;
    }

    public String getLabel() {
        return label;
    }
}