import cv2
import tensorflow as tf
import tensorflow_hub as hub


class PreprocessImage(object):
    @staticmethod
    def slice_image_quad(image: cv2.typing.MatLike) -> cv2.typing.MatLike:
        (h, w) = image.shape[:2]
        cv2.imshow('Original', image)
        # compute the center coordinate of the image
        (cX, cY) = (w // 2, h // 2)
        # crop the image into four parts which will be labelled as
        # top left, top right, bottom left, and bottom right.
        q1 = image[0:cY, 0:cX]
        q2 = image[0:cY, cX:w]
        q3 = image[cY:h, 0:cX]
        q4 = image[cY:h, cX:w]

        cv2.imwrite("../data/experimental/image_q1.jpg", q1)
        cv2.imwrite("../data/experimental/image_q2.jpg", q2)
        cv2.imwrite("../data/experimental/image_q3.jpg", q3)
        cv2.imwrite("../data/experimental/image_q4.jpg", q4)

    @staticmethod
    def slice_image(image: cv2.typing.MatLike) -> cv2.typing.MatLike:
        (h, w) = image.shape[:2]

        # cv2.imshow('Original', image)

        # compute the center coordinate of the image
        (cX3, cY3) = (w // 3, h // 3)

        split_images = []
        p11 = image[0:cY3, 0:cX3]
        p12 = image[0:cY3, cX3:cX3 + cX3]
        p13 = image[0:cY3, cX3 + cX3:w]
        split_images.append(p11)
        split_images.append(p12)
        split_images.append(p13)

        p21 = image[cY3:cY3 + cY3, 0:cX3]
        p22 = image[cY3:cY3 + cY3, cX3:cX3 + cX3]
        p23 = image[cY3:cY3 + cY3, cX3 + cX3:w]
        split_images.append(p21)
        split_images.append(p22)
        split_images.append(p23)

        p31 = image[cY3 + cY3:w, 0:cX3]
        p32 = image[cY3 + cY3:w, cX3:cX3 + cX3]
        p33 = image[cY3 + cY3:w, cX3 + cX3:w]
        split_images.append(p31)
        split_images.append(p32)
        split_images.append(p33)

        count = 1
        for elem in split_images:
            image_path = f"../data/experimental/image_{count}.jpg"
            cv2.imwrite(image_path, elem)
            count += 1

    @staticmethod
    def preprocessing(image_plot):
        imageSize = (tf.convert_to_tensor(image_plot.shape[:-1]) // 4) * 4
        cropped_image = tf.image.crop_to_bounding_box(
            img, 0, 0, imageSize[0], imageSize[1])
        preprocessed_image = tf.cast(cropped_image, tf.float32)
        return tf.expand_dims(preprocessed_image, 0)

    @staticmethod
    def upscale(image):
        model = hub.load("https://tfhub.dev/captain-pool/esrgan-tf2/1")
        preprocessed_image = PreprocessImage.preprocessing(image)  # Preprocess the LR Image
        new_image = model(preprocessed_image)  # Runs the model
        # returns the size of the original argument that is given as input
        return tf.squeeze(new_image) / 255.0


if __name__ == '__main__':
    # Read the image
    img = cv2.imread("../data/experimental/861_1717039867.jpg")
    # PreprocessImage.slice_image(img)
    PreprocessImage.upscale(img)
