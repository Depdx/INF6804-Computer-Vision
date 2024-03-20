import cv2
import os
import natsort


def main() -> None:
    image_folder = "data/frames"
    video_name = "data/video.mp4"

    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    images = natsort.natsorted(images)  # Sort the images by name

    # Read the first image to get the width and height
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(
        video_name, cv2.VideoWriter_fourcc(*"mp4v"), 30, (width, height)
    )

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()


if __name__ == "__main__":
    main()
