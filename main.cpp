#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    cv::CascadeClassifier faceCascade, eyesCascade, smileCascade;
    faceCascade.load("D:/haarcascades/haarcascade_frontalface_alt.xml");
    eyesCascade.load("D:/haarcascades/haarcascade_eye_tree_eyeglasses.xml");
    smileCascade.load("D:/haarcascades/haarcascade_smile.xml");

    cv::VideoCapture cap("D:/uwu.mp4");
    if (!cap.isOpened()) {
        std::cerr << "error" << std::endl;
        return -1;
    }

    int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    cv::VideoWriter video("D:/output.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), cap.get(cv::CAP_PROP_FPS),
        cv::Size(frameWidth, frameHeight));

    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "error empty frame" << std::endl;
            break;
        }

        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::Rect> faces, eyes, smiles;
        faceCascade.detectMultiScale(gray, faces, 1.1, 5, 0, cv::Size(30, 30));
        for (const auto& face : faces) {
            cv::rectangle(frame, face, cv::Scalar(255, 180, 0), 2);

            cv::Mat faceROI = gray(face);

            eyesCascade.detectMultiScale(faceROI, eyes, 1.1, 5, 0, cv::Size(20, 20));
            for (const auto& eye : eyes) {
                cv::Point center(face.x + eye.x + eye.width / 2, face.y + eye.y + eye.height / 2);
                int radius = cvRound((eye.width + eye.height) * 0.25);
                cv::circle(frame, center, radius, cv::Scalar(200, 200, 0), 2);
            }

            smileCascade.detectMultiScale(faceROI, smiles, 2.0, 30, 0, cv::Size(10, 10));
            for (const auto& smile : smiles) {
                cv::rectangle(frame, cv::Point(face.x + smile.x, face.y + smile.y),
                    cv::Point(face.x + smile.x + smile.width, face.y + smile.y + smile.height),
                    cv::Scalar(255, 0, 0), 2);
            }
        }

        video.write(frame);
        cv::imshow("UwU", frame);

        if (cv::waitKey(10) == 'q') {
            break;
        }
    }

    cap.release();
    video.release();
    cv::destroyAllWindows();

    return 0;
}
