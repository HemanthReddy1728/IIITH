#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

Mat DualGradientEnergyMat(Mat image)
{
    int row = image.rows, col = image.cols;
    Mat DGEMatrix(row, col, CV_64F);
    Vec3b left, right, top, bottom;

    for(int i = 0; i < row; i++)
    {
        for(int j = 0; j < col; j++)
        {
            left = image.at<Vec3b>(i, max(j - 1, 0));
            right = image.at<Vec3b>(i, min(j + 1, col - 1));
            top = image.at<Vec3b>(max(i - 1, 0), j);
            bottom = image.at<Vec3b>(min(i + 1, row - 1), j);
            DGEMatrix.at<double>(i, j) = sqrt((pow(left[0] - right[0], 2) + pow(left[1] - right[1], 2) + pow(left[2] - right[2], 2)) + (pow(top[0] - bottom[0], 2) + pow(top[1] - bottom[1], 2) + pow(top[2] - bottom[2], 2)));
        }
    }

    return DGEMatrix;
}

Mat MinCostMat(Mat DGEMatrix)
{
    int row = DGEMatrix.rows, col = DGEMatrix.cols;
    Mat MCMatrix(row, col, CV_64F);
    DGEMatrix.copyTo(MCMatrix);

    for (int i = 1; i < row; i++)  // i != 0
    {
        for (int j = 0; j < col; j++) 
        {
            MCMatrix.at<double>(i, j) += min(min(DGEMatrix.at<double>(i - 1, max(j - 1, 0)), DGEMatrix.at<double>(i - 1, min(j + 1, col - 1))), DGEMatrix.at<double>(i - 1, j));
        }
    }

    return MCMatrix;
}

double* FindSeam(Mat MCMatrix) 
{
    int row = MCMatrix.rows, col = MCMatrix.cols;
    double last_row_min = DBL_MAX;
    double* Seam = new double[row];
    
    for (int curCol = 0; curCol < col; curCol++) 
    {
        if (MCMatrix.at<double>(row - 1, curCol) < last_row_min) 
        {
            last_row_min = MCMatrix.at<double>(row - 1, curCol);
            Seam[row - 1] = curCol;
        }
    }
    
    for (int currRow = row - 2; currRow >= 0; currRow--) 
    {
        int currCol = Seam[currRow + 1];
        double left = MCMatrix.at<double>(currRow, currCol - 1), center = MCMatrix.at<double>(currRow, currCol), right = MCMatrix.at<double>(currRow, currCol + 1);
        double MinAmongTrio = min(min(left, right), center);
        
        Seam[currRow] = currCol;
        if (currCol > 0 && MinAmongTrio == left) 
        {
            Seam[currRow] = currCol - 1;
        } 
        else if (currCol < col - 1 && MinAmongTrio == right) 
        {
            Seam[currRow] = currCol + 1;
        }
    }

    return Seam;
}

Mat SeamRemovalVertical(int newWidth, Mat originalImage)
{
    while (newWidth < originalImage.cols) 
    {
        int row = originalImage.rows, col = originalImage.cols;
        double* Seam = FindSeam(MinCostMat(DualGradientEnergyMat(originalImage)));

        for (int i = 0; i < row; i++) 
        {
            for (int column = Seam[i]; column < col - 1; column++) 
            {
                originalImage.at<Vec3b>(i, column) = originalImage.at<Vec3b>(i, column + 1);
            }
        }

        originalImage = originalImage.colRange(0, col - 1);
    }

    return originalImage;
}

int main() 
{
    string inputImagePath;
    int newWidth, newHeight;

    // cout << "Enter the new width and height: ";
    cin >> inputImagePath >> newWidth >> newHeight;

    Mat originalImage = imread(inputImagePath);
    if (originalImage.empty()) 
    {
        cout << "Could not read the image." << endl;
        return -1;
    }

    if (newWidth >= originalImage.cols || newHeight >= originalImage.rows) 
    {
        cout << "New dimensions should be smaller than the original image." << endl;
        return -1;
    }

    Mat VertSeamRemvImage = SeamRemovalVertical(newWidth, originalImage), VertSeamRemvTranspImage;
    transpose(VertSeamRemvImage, VertSeamRemvTranspImage);

    Mat HorzVertSeamsRemvTranspImage = SeamRemovalVertical(newHeight, VertSeamRemvTranspImage), resizedImage;
    transpose(HorzVertSeamsRemvTranspImage, resizedImage);

    // namedWindow("Resized Image", WINDOW_NORMAL);
    // imshow("Resized Image", originalImage);
    // waitKey(0);
    imwrite("output.jpeg", resizedImage);
    // destroyAllWindows();


    return 0;
}

