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
	Mat gradDirect;
	Mat gradMag;
};

std::pair<double, double> calc_gradient(Mat image, int row, int col)
{
	float tempX = image.at<int>(row+1, col) - image.at<int>(row - 1, col);
	float tempY = image.at<int>(row, col+1) - image.at<int>(row, col - 1);

	double mag = sqrt(pow(tempX, 2) + pow(tempY, 2));
	double direction = atan(tempY/tempX);

	return std::pair<double, double>(mag, direction);
}


std::vector<int> findThetaPeaks(std::vector< std::pair<int, std::vector<int> > > hist)
{
	std::vector<int> peaks;

	for(int i = 1; i < hist.size() - 1; ++i)
	{
		if(hist[i].first > hist[i-1].first && hist[i].first > hist[i+1].first)
		{
			peaks.push_back(i);
		}
	}

	return peaks;
}

std::vector<int> findRhoPeaks(std::vector<int> hist)
{
	std::vector<int> peaks;

	for(int i = 1; i < hist.size() - 1; ++i)
	{
		if(hist[i] > hist[i-1] && hist[i] > hist[i+1])
		{
			peaks.push_back(i);
		}
	}

	return peaks;
}



int main(int argc, char** argv){

	/////// READ IN THE IMAGES TO CUSTOM CLASS ///////
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
	pentagon.gradDirect = pentagon.src;
	pentagon.gradMag = pentagon.src;
	cvtColor(pentagon.src, pentagon.lines, CV_GRAY2BGR);


	/////// GLOBAL VARIABLE INTITIALIZATIONS ///////
	int edgeThresh = 45;
	int thetaBins = 35;
	int rhoBins = 10;
	float thetaBinSize = M_PI / thetaBins;
	float rhoMax = sqrt(pow(pentagon.src.rows, 2) + pow(pentagon.src.cols, 2));
	float rhoBinSize = rhoMax / rhoBins;
	std::vector<std::pair<int, std::vector<int> > > houghTrans(thetaBins);
	for (int i = 0; i < houghTrans.size(); ++i)
	{
		houghTrans[i] = std::pair<int, std::vector<int> >(0, std::vector<int>(rhoBins));
	}

	/////// BUILD THE THETA HISTOGRAM FOR THE PENTAGON IMAGE ///////
	vector<vector<pair<int, int> > > gradDirHist(thetaBins);
	for(int i = 0; i < pentagon.edges.rows; ++i)
	{
		for(int j = 0; j < pentagon.edges.cols; ++j)
		{
			if(pentagon.edges.at<int>(i, j))
			{
				std::pair<double, double> grad = calc_gradient(pentagon.smoothed, i, j);

				for(int k = 0; k < thetaBins; ++k)
				{
					float thetaCalc = (-(M_PI / 2)) + k * thetaBinSize;
					float rhoCalc = i*cos(thetaCalc) + j*sin(thetaCalc);
					if(k == 0)
					{
						if(grad.second <= (-(M_PI / 2) + thetaBinSize))
						{
							houghTrans[k].first += 1;

							for(int h = 0; h < rhoBins; ++h)
							{
								if(h == 0)
								{
									if(rhoCalc <= rhoBinSize)
									{
										houghTrans[k].second[h] += 1;
										break;
									}
								}

								if(rhoCalc > rhoBinSize * h && rhoCalc <= rhoBinSize * (h+1))
								{
									houghTrans[k].second[h] += 1;
									break;
								}
							}
							break;
						}
					}

					if(grad.second > (-(M_PI / 2) + (thetaBinSize * k)) && grad.second <= (-(M_PI / 2) + (thetaBinSize * (k + 1))))
					{
						houghTrans[k].first += 1;
						for(int h = 0; h < rhoBins; ++h)
						{
							if(h == 0)
							{
								if(rhoCalc <= rhoBinSize)
								{
									houghTrans[k].second[h] += 1;
									break;
								}
							}

							if(rhoCalc > rhoBinSize * h && rhoCalc <= rhoBinSize * (h+1))
							{
								houghTrans[k].second[h] += 1;
								break;
							}
						}
						break;
					}
				}
			}
		}
	}



	/////// EVALUATE THE HISTOGRAMS OF THETA AND RHO ///////
	std::vector<int> thetaPeaks = findThetaPeaks(houghTrans);

	for(int i = 0; i < thetaPeaks.size(); ++i)
	{
		float theta = (-(M_PI / 2)) + thetaBinSize * thetaPeaks[i];
		std::vector<int> rhoPeaks = findRhoPeaks(houghTrans[thetaPeaks[i]].second);

		for(int j = 0; j < rhoPeaks.size(); ++j)
		{
			if(houghTrans[thetaPeaks[i]].second[rhoPeaks[j]] < edgeThresh)
			{
				continue;
			}

			float rho = rhoBinSize * rhoPeaks[j];
			Point pt1, pt2;
			double a = cos(theta), b = sin(theta);
			double x0 = a*rho, y0 = b*rho;
			pt1.x = cvRound(x0 + 1000*(-b));
			pt1.y = cvRound(y0 + 1000*(a));
			pt2.x = cvRound(x0 - 1000*(-b));
			pt2.y = cvRound(y0 - 1000*(a));
			line(pentagon.lines, pt1, pt2, Scalar(0,0,255), 1, CV_AA);
		}
	}


	/////// DISPLAY THE IMAGE WITH THE LINES DRAWN ////////
	imshow("Pentagon Lines", pentagon.lines);
	waitKey();

	return 0;
}
