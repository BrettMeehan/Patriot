//
//  main.cpp
//  Patriot
//
//  Created by Brett Meehan on 7/8/16.
//  Copyright © 2016 Brett Meehan. All rights reserved.
//

#include <iostream>
#include <cmath>
#include <cassert>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/nonfree/nonfree.hpp>

using namespace cv;

// 1. adjust color sensitivity manually
// 2. increase image contrast to identify color boundaries more easily
// 3. search for stars and blue background (candidate points are RED circles)
// 4. search for stripe pattern(s)         (candidate points are GREEN circles)
// 5. if distance between one pair of stripes and stars is close enough/matches scale, draw connecting line (BLUE)

const int CONTRAST_LEVEL = 8;// use 6 or 8
const int RED_SENSITIVITY = 130;
const int WHITE_SENSITIVITY = 100;
const int DARK_BLUE_SENSITIVITY = 110;
const int MIN_STARS = 2;
const int STAR_BACKGROUND_SCALE_DIFFERENCE = 2;
const int MIN_STRIPES = 4;
const int STRIPE_SCALE_DIFFERENCE = 2;
const int STAR_TO_STRIPE_SCALE_DIFFERENCE = 3;
const int MAX_STRIPE_SCALE_DISTANCE_FACTOR = 10;// multiply by stripe scale to get max distance to possible star location

enum PIXEL_TYPE {
    WHITE,
    RED,
    BLUE,
    OTHER
};

void findStarsAndStripes(Mat_<Vec3b> &, std::vector<std::pair<std::pair<int, int>, int>> &, std::vector<std::pair<std::pair<int, int>, int>> &, bool);
std::pair<bool, int> similarAltColorWidthsAndScale(const std::vector<std::pair<PIXEL_TYPE, int>> &, int);
void findFlagsSURF(Mat &, Mat &);

inline bool isWhite(const Vec3b & pixel) {
    return pixel[0] > WHITE_SENSITIVITY && pixel[1] > WHITE_SENSITIVITY && pixel[2] > WHITE_SENSITIVITY;
}

inline bool isRed(const Vec3b & pixel) {
    return pixel[0] < 100 && pixel[1] < 100 && pixel[2] > RED_SENSITIVITY;
}

inline bool isBlue(const Vec3b & pixel) {
    //return pixel[0] - pixel[1] > 15 && pixel[0] - pixel[2] > 15 && pixel[0] < 110;
    return (pixel[0] > pixel[1] && pixel[0] > pixel[2] && pixel[0] < DARK_BLUE_SENSITIVITY) || (pixel[0] < 25 && pixel[1] < 25 && pixel[2] < 25);
}

inline double distance(const Point & point1, const Point & point2) {
    return sqrt(pow(point1.x - point2.x, 2) + pow(point1.y - point2.y, 2));
}

int main(int argc, char** argv) {
    
    if(argc < 3) {
        std::cout << "Usage: ./Patriot imgSceneName SURFtrainingFlagName" << std::endl;
        return -1;
    }
    
    std::string imgSceneName = argv[1];//"/Users/Brett/Desktop/flags/flag3.jpg";
    std::string SURFtrainingFlagName = argv[2];//"/Users/Brett/Desktop/flags/flag2.jpg";
    assert(imgSceneName != SURFtrainingFlagName); // images can't be the same; otherwise SURF throws an exception
    Mat image = imread(imgSceneName, CV_LOAD_IMAGE_COLOR);   // Read the file
    
    if(!image.data) {                             // Check for invalid input
        std::cout <<  "Could not open or find the image" << std::endl;
        return -1;
    }
    CV_Assert(image.depth() == CV_8U && image.channels() == 3);
    
    // compare against standard SURF detection using code from OpenCV tutorial
    Mat img_object = imread(SURFtrainingFlagName, CV_LOAD_IMAGE_COLOR );
    Mat img_scene = image.clone();
    assert(!std::equal(img_object.begin<uchar>(), img_object.end<uchar>(), img_scene.begin<uchar>()));// images can't be the same; otherwise SURF throws an exception
    findFlagsSURF(img_object, img_scene);
    
    
    
    // Patriot detection
    // increase image contrast to identify color boundaries more easily
    Mat contrastImage = image.clone();
    Mat kernel = (Mat_<double>(3, 3) << 0, -1,  0
                                       -1,  CONTRAST_LEVEL, -1,
                                        0, -1,  0);
    filter2D(image, contrastImage, image.depth(), kernel);

    
    std::vector<std::pair<std::pair<int, int>, int>> candidateStarCoordsAndScale, candidateStripeCoordsAndScale;
    Mat_<Vec3b> _image = contrastImage;
    bool horizontal = true;
    // search by row
    findStarsAndStripes(_image, candidateStarCoordsAndScale, candidateStripeCoordsAndScale, horizontal);
    // search by column
    findStarsAndStripes(_image, candidateStarCoordsAndScale, candidateStripeCoordsAndScale, !horizontal);
    
    // draw RED circles around star candidate points
    for(int i = 0; i < candidateStarCoordsAndScale.size(); ++i) {
        //std::cout << candidateStarCoordsAndScale[i].first.first << ", " << candidateStarCoordsAndScale[i].first.second << ": " << candidateStarCoordsAndScale[i].second << std::endl;
        
        int r = 10, thickness = 1, lineType = 8;
        circle( image,
               Point(candidateStarCoordsAndScale[i].first.second, candidateStarCoordsAndScale[i].first.first),
               r,
               Scalar( 0, 0, 255 ),
               thickness,
               lineType );
    }
    //std::cout << "----------" << std::endl;
    
    // draw GREEN circles around stripe candidate points
    for(int i = 0; i < candidateStripeCoordsAndScale.size(); ++i) {
        //std::cout << candidateStripeCoordsAndScale[i].first.first << ", " << candidateStripeCoordsAndScale[i].first.second << ": " << candidateStripeCoordsAndScale[i].second << std::endl;
        int r = 10, thickness = 1, lineType = 8;
        circle( image,
               Point(candidateStripeCoordsAndScale[i].first.second, candidateStripeCoordsAndScale[i].first.first),
               r,
               Scalar( 0, 255, 0 ),
               thickness,
               lineType );
    }
    
    // draw BLUE lines between properly scaled star and stripe candidate areas
    for(int i = 0; i < candidateStarCoordsAndScale.size(); ++i) {
        for(int j = 0; j < candidateStripeCoordsAndScale.size(); ++j) {
            Point starLoc = Point(candidateStarCoordsAndScale[i].first.second, candidateStarCoordsAndScale[i].first.first);
            Point stripeLoc = Point(candidateStripeCoordsAndScale[j].first.second, candidateStripeCoordsAndScale[j].first.first);
            double dist = distance(starLoc, stripeLoc);
            int starScale = candidateStarCoordsAndScale[i].second;
            int stripeScale = candidateStripeCoordsAndScale[j].second;
            //std::cout << "dist: " << dist << " ,  starScale: " << starScale << ",  stripeScale: " << stripeScale << std::endl;
            if(dist <= MAX_STRIPE_SCALE_DISTANCE_FACTOR*stripeScale && stripeScale <= STAR_TO_STRIPE_SCALE_DIFFERENCE*starScale)
                line(image, starLoc, stripeLoc, Scalar(255, 0, 0));
        }
    }
    
    namedWindow("Patriot: Good matches", WINDOW_AUTOSIZE);
    imshow("Patriot: Good matches", image);
    
    
    waitKey(0);                                          // Wait for a keystroke in the window
    return 0;
}

// does a row-by-row or column-by-column scan for alternating color patterns
void findStarsAndStripes(Mat_<Vec3b> & _image, std::vector<std::pair<std::pair<int, int>, int>> & starCoordsAndScale, std::vector<std::pair<std::pair<int, int>, int>> & stripeCoordsAndScale, bool horizontal) {
    int outerLimit, innerLimit;
    if(horizontal) {
        outerLimit = _image.rows;
        innerLimit = _image.cols;
    } else {
        outerLimit = _image.cols;
        innerLimit = _image.rows;
    }
    for(int i = 0; i < outerLimit; ++i) {
        std::vector<std::pair<PIXEL_TYPE, int>> BWlengths, RWlengths;
        PIXEL_TYPE prevPixel = OTHER, currentPixel = OTHER;
        int lastColorStartIndex = 0;
        for(int j = 0; j < innerLimit; ++j) {
            Vec3b pixel = horizontal ? _image(i, j) : _image(j, i);
            if(isWhite(pixel))
                currentPixel = WHITE;
            else if(isRed(pixel))
                currentPixel = RED;
            else if(isBlue(pixel))
                currentPixel = BLUE;
            else
                currentPixel = OTHER;
            if(currentPixel != prevPixel) {
                // find stars
                // only care about adjacent Blue and White chunks
                if((currentPixel == WHITE || currentPixel == BLUE) && prevPixel != WHITE && prevPixel != BLUE)
                    BWlengths.clear();
                int BWcolorLength = j - lastColorStartIndex;
                if((prevPixel == WHITE || prevPixel == BLUE) && (currentPixel == WHITE || currentPixel == BLUE)) {
                    BWlengths.push_back(std::pair<PIXEL_TYPE, int>(currentPixel, BWcolorLength));
                    // half of the lengths should be white stars
                    if(BWlengths.size()/2 == MIN_STARS) {
                        std::pair<bool, int> validAndScale = similarAltColorWidthsAndScale(BWlengths, STAR_BACKGROUND_SCALE_DIFFERENCE);
                        if(validAndScale.first) { // use midpoint of scanning direction?
                            std::pair<int, int> coordinate = horizontal ? std::pair<int, int>(i, j) : std::pair<int, int>(j, i);
                            starCoordsAndScale.push_back(std::pair<std::pair<int, int>, int>(coordinate, validAndScale.second));
                            BWlengths.clear();
                        } else
                            BWlengths.erase(BWlengths.begin());
                    }
                }
                
                // find stripes
                // only care about adjacent Red and White chunks-RED and WHITE should alternate at least 4 times, e.g. RWRW or WRWR
                if((currentPixel == WHITE || currentPixel == RED) && prevPixel != WHITE && prevPixel != RED)
                    RWlengths.clear();
                int RWcolorLength = j - lastColorStartIndex;
                if((prevPixel == WHITE || prevPixel == RED) && (currentPixel == WHITE || currentPixel == RED)) {
                    RWlengths.push_back(std::pair<PIXEL_TYPE, int>(currentPixel, RWcolorLength));
                    if(RWlengths.size() == MIN_STRIPES) {
                        std::pair<bool, int> validAndScale = similarAltColorWidthsAndScale(RWlengths, STRIPE_SCALE_DIFFERENCE);
                        if(validAndScale.first) { // use midpoint of scanning direction?
                            std::pair<int, int> coordinate = horizontal ? std::pair<int, int>(i, j) : std::pair<int, int>(j, i);
                            stripeCoordsAndScale.push_back(std::pair<std::pair<int, int>, int>(coordinate, validAndScale.second));
                            RWlengths.clear();
                        } else
                            RWlengths.erase(RWlengths.begin());
                    }
                }
                lastColorStartIndex = j;
            }
            prevPixel = currentPixel;
        }
    }
}

// O(3n/2) algorithm for finding both max and min
std::pair<bool, int> similarAltColorWidthsAndScale(const std::vector<std::pair<PIXEL_TYPE, int>> & colorWidths, const int scaleDifference) {
    assert(colorWidths.size() >= 2);
    
    // first find min and max widths
    const int numColorWidths = colorWidths.size();
    int min, max, i;
    if(numColorWidths % 2 == 0) {
        if(colorWidths[0].second < colorWidths[1].second) {
            min = colorWidths[0].second;
            max = colorWidths[1].second;
        } else {
            min = colorWidths[1].second;
            max = colorWidths[0].second;
        }
        i = 1;
    } else {
        min = max = colorWidths[0].second;
        i = 2;
    }
    for(; i < numColorWidths - 1; i += 2) {
        int colorWidth1 = colorWidths[i].second;
        int colorWidth2 = colorWidths[i + 1].second;
        if(colorWidth1 < colorWidth2) {
            if(colorWidth1 < min)
                min = colorWidth1;
            if(colorWidth2 > max)
                max = colorWidth2;
        } else {
            if(colorWidth1 > max)
                max = colorWidth1;
            if(colorWidth2 < min)
                min = colorWidth2;
        }
    }
    return std::pair<bool, int>(max <= scaleDifference*min, max); // account for folded flags and such
}

// standard SURF detection using code from OpenCV tutorial
void findFlagsSURF(Mat & img_object, Mat & img_scene) {
    assert(img_object.data && img_scene.data);
    
    //-- Step 1: Detect the keypoints using SURF Detector
    int minHessian = 400;
    
    SurfFeatureDetector detector( minHessian );
    
    std::vector<KeyPoint> keypoints_object, keypoints_scene;
    
    detector.detect( img_object, keypoints_object );
    detector.detect( img_scene, keypoints_scene );
    
    //-- Step 2: Calculate descriptors (feature vectors)
    SurfDescriptorExtractor extractor;
    
    Mat descriptors_object, descriptors_scene;
    
    extractor.compute( img_object, keypoints_object, descriptors_object );
    extractor.compute( img_scene, keypoints_scene, descriptors_scene );
    
    //-- Step 3: Matching descriptor vectors using FLANN matcher
    FlannBasedMatcher matcher;
    std::vector< DMatch > matches;
    matcher.match( descriptors_object, descriptors_scene, matches );
    
    double max_dist = 0; double min_dist = 100;
    
    //-- Quick calculation of max and min distances between keypoints
    for( int i = 0; i < descriptors_object.rows; i++ )
    { double dist = matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
    }
    
    //printf("-- Max dist : %f \n", max_dist );
    //printf("-- Min dist : %f \n", min_dist );
    
    //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
    std::vector< DMatch > good_matches;
    
    for( int i = 0; i < descriptors_object.rows; i++ )
    { if( matches[i].distance < 3*min_dist )
    { good_matches.push_back( matches[i]); }
    }
    
    Mat img_matches;
    drawMatches( img_object, keypoints_object, img_scene, keypoints_scene,
                good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    
    //-- Localize the object
    std::vector<Point2f> obj;
    std::vector<Point2f> scene;
    
    for( int i = 0; i < good_matches.size(); i++ )
    {
        //-- Get the keypoints from the good matches
        obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
        scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
    }
    
    Mat H = findHomography( obj, scene, CV_RANSAC );
    
    //-- Get the corners from the image_1 ( the object to be "detected" )
    std::vector<Point2f> obj_corners(4);
    obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( img_object.cols, 0 );
    obj_corners[2] = cvPoint( img_object.cols, img_object.rows ); obj_corners[3] = cvPoint( 0, img_object.rows );
    std::vector<Point2f> scene_corners(4);
    
    perspectiveTransform( obj_corners, scene_corners, H);
    
    //-- Draw lines between the corners (the mapped object in the scene - image_2 )
    line( img_matches, scene_corners[0] + Point2f( img_object.cols, 0), scene_corners[1] + Point2f( img_object.cols, 0), Scalar(0, 255, 0), 4 );
    line( img_matches, scene_corners[1] + Point2f( img_object.cols, 0), scene_corners[2] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
    line( img_matches, scene_corners[2] + Point2f( img_object.cols, 0), scene_corners[3] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
    line( img_matches, scene_corners[3] + Point2f( img_object.cols, 0), scene_corners[0] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
    
    //-- Show detected matches
    imshow( "SURF: Good Matches", img_matches );
}