#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <omp.h>
using namespace cv;
using namespace std;
void detectFaces(Mat& frame, CascadeClassifier& faceCascade, vector<cv::Rect>& faces) 
{
    Mat grayFrame;
    cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
    faceCascade.detectMultiScale(grayFrame, faces, 2, 3, 0, Size(30, 30));
    for (const auto& face : faces) 
    {
        rectangle(frame, face, Scalar(0, 0, 255), 2);
    }
}

void detectEyes(Mat& frame, vector<Rect>& faces, CascadeClassifier& eyeCascade, int maxEyesPerFace)
{
    for (auto& face : faces) 
    {
        Mat faceROI = frame(face);
        vector<Rect> eyes;
        eyeCascade.detectMultiScale(faceROI, eyes, 3, 8, 0, Size(30, 30));
        if (eyes.size() > maxEyesPerFace)
        {
            eyes.resize(maxEyesPerFace);
        }

        for (size_t j = 0; j < eyes.size(); ++j)
        {
            Point eye_center(face.x + eyes[j].x + eyes[j].width / 2, face.y + eyes[j].y + eyes[j].height / 2);
            int radius = cvRound((eyes[j].width + eyes[j].height) * 0.25);
            circle(frame, eye_center, radius, Scalar(255, 0, 0), 2);
        }
    }
}

void detectSmiles(const Mat& frame, vector<cv::Rect>& faces, CascadeClassifier& smileCascade) {
    for (auto& face : faces) 
    {
        Mat faceROI = frame(face);
        vector<cv::Rect> smiles;
        smileCascade.detectMultiScale(faceROI, smiles, 3, 35, 0, Size(30, 30));
        for (size_t k = 0; k < smiles.size(); ++k)
        {
            Point pt1(faces[0].x + smiles[k].x, faces[0].y + smiles[k].y);
            Point pt2(faces[0].x + smiles[k].x + smiles[k].width, faces[0].y + smiles[k].y + smiles[k].height);
            rectangle(frame, pt1, pt2, Scalar(0, 0, 255), 2);
        }
    }
}

int main() {
    CascadeClassifier faceCascade, eyeCascade, smileCascade;
    faceCascade.load("haarcascade_frontalface_alt.xml");
    eyeCascade.load("haarcascade_eye.xml");
    smileCascade.load("haarcascade_smile.xml");
    VideoCapture cap("video.mp4");
    cap.isOpened();  
    vector<Mat> processedFrames;
    Mat frame;
    while (cap.read(frame)) 
    {    
#pragma omp parallel sections num_threads(8) 
        {
#pragma omp section
        vector<Rect> faces;
        detectFaces(frame, faceCascade, faces);      
#pragma omp section
            detectEyes(frame, faces, eyeCascade,2);
#pragma omp section
            detectSmiles(frame, faces, smileCascade);
        }
        processedFrames.push_back(frame.clone());
    }
    cap.release();
    int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    VideoWriter video("output.mp4",VideoWriter::fourcc('m', 'p', '4', 'v'), fps, Size(frameWidth, frameHeight));
    for (const auto& frame : processedFrames) 
    {
        video.write(frame);
        imshow("Output Video", frame);
        if (waitKey(30) == 27) 
        {
            break;
        }
    }
    video.release();
    destroyAllWindows();

    return 0;
}
