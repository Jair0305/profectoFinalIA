import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

public class ImagePreprocessor {
    private static final int IMG_WIDTH = 64;
    private static final int IMG_HEIGHT = 64;

    public static Mat preprocessImage(String imagePath) {
        // Leer la imagen
        Mat image = Imgcodecs.imread(imagePath, Imgcodecs.IMREAD_GRAYSCALE);

        // Redimensionar la imagen
        Imgproc.resize(image, image, new Size(IMG_WIDTH, IMG_HEIGHT));

        // Normalizar los valores de los píxeles
        image.convertTo(image, CvType.CV_32F);
        Core.divide(image, new Scalar(255), image);

        return image;
    }
}