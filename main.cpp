#include <iostream>
#include <string>
#include <cmath>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

class ImObj {
public:
	Mat src;
	Mat edges;
	Mat clean_edges;
	Mat smoothed;
	Mat lines;
};

std::pair<double, double> calc_gradient(Mat image, int row, int col)
{
	int tempX = image.at<int>(row+1, col) - image.at<int>(row - 1, col);
	int tempY = image.at<int>(row, col+1) - image.at<int>(row, col - 1);

	double mag = sqrt(pow(tempX, 2) + pow(tempY, 2));
	double direction = atan2(tempY, tempX);

	if(direction < 0)
	{
		direction += 2*M_PI;
	}

	return std::pair<double, double>(mag, direction);
}

int main(int argc, char** argv){

	ImObj box;
	ImObj pentagon;

	string fn;

	cout << "Enter path of images: ";
	cin >> fn;

	box.src = imread(fn + "box.pgm", 0);
	box.edges = imread(fn + "boxCanny2.pgm", 0);
	box.clean_edges = imread(fn + "boxCannyClean.pgm", 0);
	box.smoothed = imread(fn + "boxSmoothed2.pgm", 0);

	pentagon.src = imread(fn + "pentagon.pgm", 0);
	pentagon.edges = imread(fn + "pentagonCanny2.pgm", 0);
	pentagon.clean_edges = imread(fn + "pentagonCannyClean", 0);
	pentagon.smoothed = imread(fn + "pentagonSmooth2.pgm", 0);
	cvtColor(pentagon.src, pentagon.lines, CV_GRAY2BGR);


	for(int i = 0; i < pentagon.edges.rows; ++i)
	{
		for(int j = 0; j < pentagon.edges.cols; ++j)
		{
			if(pentagon.edges.at<int>(i, j))
			{
				std::pair<double, double> grad = calc_gradient(pentagon.smoothed, i, j);
//				cout << "Gradient Mag: " << grad.first << endl;
//				cout << "Gradient Direction: " << grad.second << endl << endl;
			}
		}
	}

	vector<Vec2f> lines;
	HoughLines(pentagon.src, lines, 1, CV_PI/180, 150, 0, 0);

	for(size_t i = 0; i < lines.size(); ++i)
	{
		float rho = lines[i][0], theta = lines[i][i];
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a*rho, y0 = b*rho;
		pt1.x = cvRound(x0 + 1000*(-b));
		pt1.y = cvRound(y0 + 1000*(a));
		pt2.x = cvRound(x0 - 1000*(-b));
		pt2.y = cvRound(y0 - 1000*(a));
		line(pentagon.lines, pt1, pt2, Scalar(0,0,255), 3, CV_AA);
	}

	imshow("Detected Lines", pentagon.lines);
	waitKey();

	return 0;
}
