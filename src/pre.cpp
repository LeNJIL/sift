#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
#include "../include/pre.h"

namespace cv_pre {

    vector<matchPoint> compute_macth(Keypoint *key1, Keypoint *key2, float maxloss) {
        double rate_thres = 0.5;
        vector<matchPoint> mac;
        Keypoint *k1 = key1, *k2 = key2;
        while (k1) {
            k2 = key2;
            Point2i m1(0,0), m2(0,0);
            float loss1 = 1000000, loss2 = 100000000;
            while (k2) {
                float loss = 0;
                float *f1 = k1->descrip;
                float *f2 = k2->descrip;
                for (int i = 0; i < 128; i++) {
                    loss += pow(*f1++ - *f2++, 2);
                }
                loss = sqrt(loss);
                if (loss <= loss1) {
                    loss2 = loss1;
                    loss1 = loss;
                    m2 = m1;
                    m1 = Point2i(k2->col, k2->row);
                }
                else if (loss < loss2) {
                    loss2 = loss;
                    m2 = Point2i(k2->col, k2->row);
                }
                k2=k2->next;
            }
            if (loss1 < maxloss) {
                if (loss1 <(rate_thres*loss2)) {
                    mac.push_back(matchPoint(Point2i(k1->col, k1->row), m1));
                    //cout << (loss1) << endl;
                }
            }
            k1=k1->next;
        }
        return mac;
    }


    //下采样原来的图像，返回缩小2倍尺寸的图像
    Mat halfSizeImage(Mat im)
    {
        int w = im.cols / 2;
        int h = im.rows / 2;
        Mat imnew(h, w, CV_32FC1);
        for (int r = 0; r < h; r++)
            for (int c = 0; c < w; c++)
                imnew.ptr<float>(r)[c] = im.ptr<float>(2 * r)[2 * c];
        return imnew;
    }

    void doubleSizeImageColor(Mat im,Mat imnew)
    {
        int w = im.cols * 2;
        int h = im.rows * 2;
        for (int r = 0; r < h; r++){
            for (int c = 0; c < w; c++) {
                imnew.ptr<float>(r)[3 * c] = (float)im.ptr<float>(r / 2)[3 * (c / 2)];
                imnew.ptr<float>(r)[3 * c+1] = (float)im.ptr<float>(r / 2)[3 * (c / 2) + 1];
                imnew.ptr<float>(r)[3 * c+2] = (float)im.ptr<float>(r / 2)[3 * (c / 2) + 2];
            }
        }
        //cout << 1;
    }

    //上采样原来的图像，返回放大2倍尺寸的图像
    Mat doubleSizeImage(Mat im)
    {
        int w = im.cols * 2;
        int h = im.rows * 2;
        Mat imnew(h, w, CV_32FC1);
        for (int r = 0; r < h; r++)
            for (int c = 0; c < w; c++)
                imnew.ptr<float>(r)[c] = im.ptr<float>( r/2)[c/2];
        return imnew;
    }

    //上采样原来的图像，返回放大2倍尺寸的线性插值图像
    Mat doubleSizeImage2(Mat im)
    {
        int w = im.cols * 2;
        int h = im.rows * 2;
        Mat imnew(h, w, CV_32FC1);
        for (int r = 0; r < h; r++)
            for (int c = 0; c < w; c++)
                imnew.ptr<float>(r)[c] = im.ptr<float>(r / 2)[c / 2];

        /*
        A B C
        E F G
        H I J
        pixels A C H J are pixels from original image
        pixels B E G I F are interpolated pixels
        */
        // interpolate pixels B and I
        for (int r = 0; r < h; r += 2)
            for (int c = 1; c < w - 1; c += 2)
                imnew.ptr<float>(r)[c] = 0.5*(imnew.ptr<float>(r)[c - 1] + imnew.ptr<float>(r)[c + 1]);
        // interpolate pixels E and G
        for (int r = 1; r < h - 1; r += 2)
            for (int c = 0; c < w; c += 2)
                imnew.ptr<float>(r)[c] = 0.5*(imnew.ptr<float>(r-1)[c] + imnew.ptr<float>(r+1)[c]);
        // interpolate pixel F
        for (int r = 1; r < h - 1; r += 2)
            for (int c = 1; c < w - 1; c += 2)
                imnew.ptr<float>(r)[c] = 0.25*(imnew.ptr<float>(r - 1)[c] + imnew.ptr<float>(r + 1)[c] + imnew.ptr<float>(r)[c - 1] + imnew.ptr<float>(r)[c + 1]);
        return imnew;
    }

    //双线性插值，返回像素间的灰度值
    float getPixelBI(Mat im, float col, float row)
    {
        int irow = (int)row, icol = (int)col;   //实部
        float rfrac, cfrac;                     //虚部
        int width = im.cols;
        int height = im.rows;
        if (irow < 0 || irow >= height
            || icol < 0 || icol >= width)
            return 0;
        if (row > height - 1)
            row = height - 1;
        if (col > width - 1)
            col = width - 1;
        rfrac = (row - (float)irow);
        cfrac = (col - (float)icol);

        float row1 = 0, row2 = 0;
        if (cfrac > 0) {
            row1 = (1 - cfrac)*im.ptr<float>(irow)[icol] + cfrac*im.ptr<float>(irow)[icol + 1];
        }
        else {
            row1 = im.ptr<float>(irow)[icol];
        }
        if (rfrac > 0) {
            if (cfrac > 0) {
                row2 = (1 - cfrac)*im.ptr<float>(irow + 1)[icol] + cfrac*im.ptr<float>(irow + 1)[icol + 1];
            }
            else row2 = im.ptr<float>(irow + 1)[icol];
        }
        else {
            return row1;
        }
        return ((1 - rfrac)*row1 + rfrac*row2);
    }

    //矩阵归一化
    void normalizeMat(Mat mat)
    {
        float sum = 0;

        for (unsigned int r = 0; r < mat.rows; r++)
            for (unsigned int c = 0; c < mat.cols; c++)
                sum += mat.ptr<float>(r)[c];
        for (unsigned int r = 0; r < mat.rows; r++)
            for (unsigned int c = 0; c < mat.cols; c++)
                mat.ptr<float>(r)[c] = mat.ptr<float>(r)[c] / sum;
    }

    //向量归一化
    void normalizeVec(float* vec, int dim)
    {
        unsigned int i;
        float sum = 0;
        for (i = 0; i < dim; i++)
            sum += vec[i];
        for (i = 0; i < dim; i++)
            vec[i] /= sum;
    }

    //得到向量   L2-范数
    float GetVecNorm(float* vec, int dim)
    {
        float sum = 0.0;
        for (unsigned int i = 0; i<dim; i++)
            sum += vec[i] * vec[i];
        return sqrt(sum);
    }

    //产生1D高斯核
    float* GaussianKernel1D(float sigma, int dim)
    {
        float *kern = (float*)malloc(dim * sizeof(float));
        float s2 = sigma * sigma;
        int c = dim / 2;
        float m = 1.0 / (sqrt(2.0 * CV_PI) * sigma);
        double v;
        for (int i = 0; i < (dim + 1) / 2; i++)
        {
            v = m * exp(-(1.0*i*i) / (2.0 * s2));
            kern[c + i] = v;
            kern[c - i] = v;
        }
        return kern;
    }

    //产生2D高斯核矩阵
    Mat GaussianKernel2D(float sigma)
    {
        int dim = (int)max(3.0, 2.0 * GAUSSKERN *sigma + 1.0);
        if (dim % 2 == 0)
            dim++;
        Mat mat(dim, dim, CV_32FC1);
        float s2 = sigma * sigma;
        int c = dim / 2;
        float m = 1.0 / (sqrt(2.0 * CV_PI) * sigma);    //前方系数
        for (int i = 0; i < (dim + 1) / 2; i++)
        {
            for (int j = 0; j < (dim + 1) / 2; j++)
            {
                float v = m * exp(-(1.0*i*i + 1.0*j*j) / (2.0 * s2));
                mat.ptr<float>(c + i)[c + j] = v;
                mat.ptr<float>(c - i)[c + j] = v;
                mat.ptr<float>(c + i)[c - j] = v;
                mat.ptr<float>(c - i)[c - j] = v;
            }
        }

        return mat;
    }

    //x方向像素处作卷积
    float ConvolveLocWidth(float* kernel, int dim, Mat src, int x, int y)
    {
        unsigned int i;
        float pixel = 0;
        int col;
        int cen = dim / 2;
        for (i = 0; i < dim; i++)
        {
            col = x + (i - cen);
            if (col < 0) {
                col = 0;
                continue;
            }

            if (col >= src.cols) {
                col = src.cols - 1;
                continue;
            }
            pixel += kernel[i] * src.ptr<float>(y)[col];;
        }
        if (pixel > 1)
            pixel = 1;
        return pixel;
    }

    //x方向作卷积
    void Convolve1DWidth(float* kern, int dim, Mat src, Mat dst)
    {
        unsigned int i, j;

        for (j = 0; j < src.rows; j++)
        {
            for (i = 0; i < src.cols; i++)
            {
                //printf("%d, %d/n", i, j);
                dst.ptr<float>(j)[i] = ConvolveLocWidth(kern, dim, src, i, j);
            }
        }
    }

    //y方向像素处作卷积
    float ConvolveLocHeight(float* kernel, int dim, Mat src, int x, int y)
    {
        unsigned int j;
        float pixel = 0;
        int cen = dim / 2;
        //printf("ConvolveLoc(): Applying convoluation at location (%d, %d)/n", x, y);
        for (j = 0; j < dim; j++)
        {
            int row = y + (j - cen);
            if (row < 0)
                row = 0;
            if (row >= src.rows)
                row = src.rows - 1;
            pixel += kernel[j] * src.ptr<float>(row)[x];;
        }
        if (pixel > 1)
            pixel = 1;
        return pixel;
    }

    //y方向作卷积
    void Convolve1DHeight(float* kern, int dim, Mat src, Mat dst)
    {
        unsigned int i, j;
        for (j = 0; j < src.rows; j++)
        {
            for (i = 0; i < src.cols; i++)
            {
                //printf("%d, %d/n", i, j);
                dst.ptr<float>(j)[i] = ConvolveLocHeight(kern, dim, src, i, j);
            }
        }
    }

    //卷积模糊图像
    int BlurImage(Mat src, Mat dst, float sigma)
    {
        float* convkernel;
        int dim = (int)max(3.0, 2.0 * GAUSSKERN * sigma + 1.0);
        // make dim odd
        if (dim % 2 == 0)
            dim++;
        Mat tempMat = Mat(src.rows, src.cols, CV_32FC1);
        convkernel = GaussianKernel1D(sigma, dim);
        Convolve1DWidth(convkernel, dim, src, tempMat);
        Convolve1DHeight(convkernel, dim, tempMat, dst);
        return dim;
    }




}
