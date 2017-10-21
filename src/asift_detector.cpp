
#include "asift_detector.h"

#include <math.h>

#include <sift.h>

#include "asift.h"
#include "asift_feature_container.h"
#include "asift_model.h"

using namespace std;
using namespace cv;
using namespace asift;


static const float PI_F = 3.1415927f;


Detector::Detector()
	: projectionBorderFactor ( PROJECTION_BORDER_FACTOR )
	, tilts                  ( TILTS                    )
	, tiltFactor             ( TILT_FACTOR              )
	, rotationsPerTilt       ( ROTATIONS_PER_TILT       )
{ }

void Detector::setProjectionBorderFactor(float factor) {
	if (factor < 0.f)
		throw domain_error("The projection border factor must be positive or zero.");

	this->projectionBorderFactor = factor;
}

void Detector::setTilts(int tilts) {
	if (tilts < 1)
		throw domain_error("The number of tilts must be positive.");

	this->tilts = tilts;
}

void Detector::setTiltFactor(float factor) {
	if (factor <= 1.f)
		throw domain_error("The tilt factor must be greater than 1.");

	this->tiltFactor = factor;
}

void Detector::setRotationsPerTilt(float rotations) {
	if (rotations < 1.f)
		throw domain_error("The rotations per tilt must be greater or equal 1.");

	this->rotationsPerTilt = rotations;
}

void Detector::setUpscale(bool upscale) {
	detector.setUpscale(upscale);
}

void Detector::setBorderWidth(int width) {
	detector.setBorderWidth(width);
}

void Detector::setIntervals(int intervals) {
	detector.setIntervals(intervals);
}

void Detector::setSigma(float sigma) {
	detector.setSigma(sigma);
}

void Detector::setSigmaInit(float sigmaInit) {
	detector.setSigmaInit(sigmaInit);
}

void Detector::setContrastThreshold(float threshold) {
	detector.setContrastThreshold(threshold);
}

void Detector::setCurvatureThreshold(float threshold) {
	detector.setCurvatureThreshold(threshold);
}

void Detector::setOrientationSigmaFactor(float factor) {
	detector.setOrientationSigmaFactor(factor);
}

void Detector::setOrientationRadiusFactor(float factor) {
	detector.setOrientationRadiusFactor(factor);
}

void Detector::setOrientationPeakRatio(float ratio) {
	detector.setOrientationPeakRatio(ratio);
}

void Detector::setOrientationSmoothIterations(int iterations) {
	detector.setOrientationSmoothIterations(iterations);
}

void Detector::setDescriptorSigmaFactor(float factor) {
	detector.setDescriptorSigmaFactor(factor);
}

void Detector::setDescriptorMagnitudeThreshold(float threshold) {
	detector.setDescriptorMagnitudeThreshold(threshold);
}

int Detector::detect(const Mat &img, vector<FeatureContainer> &features) const {
	// initialize routine

	this->img = &img;
	this->features = &features;
	init();

	// feature detection loop

	loop();

	// sum number of features

	int n = features.size();
	
	int nfeatures = 0;
	for (int i = 0; i < n; ++i)
		nfeatures += features[i].siftModel.size();

	return nfeatures;
}

int Detector::detect(const cv::Mat &img, asift::Model &model) const {
	return model.detect(*this, img);
}

void Detector::init() const {
	projCounter = 0;

	// sum number of simulations

	int simulations = 0;
	for (int t = 0; t < tilts; ++t) {
		float tilt = calcTilt(t);

		// if no tilt
		if (tilt == 1.f)
			simulations += 1; // only angle 0
		else
			simulations += calcRotations(tilt);
	}

	// reserve feature vector
	features->resize(simulations);
}

void Detector::loop() const {
	for (int t = 0; t < tilts; ++t) {
		float tilt = calcTilt(t);

		// if no tilt
		if (tilt == 1.f)
			step(tilt, 0.f, true);
		else {
			int rotations = calcRotations(tilt);
			float dangle = PI_F / rotations;

			float angle = 0.f;
			for (int r = 0; r < rotations; ++r) {
				// don't filter 0 and PI/2 angles
				bool noFilter = r == 0 || 2*r == rotations;

				step(tilt, angle, noFilter);
				angle += dangle;
			}
		}
	}
}

void Detector::step(float tilt, float angle, bool noFilter) const {
	nextContainer();
	container->tilt = tilt;
	container->angle = angle;

	project();

	container->siftModel.detect(detector, projection);

	if (!noFilter)
		filter();

	compensateKeyPoints();
}

void Detector::nextContainer() const {
	container = &(*features)[projCounter++];
}

void Detector::project() const {
	Mat tmp;
	
	// TODO use field instead of .8f :P
	float antiAliasing = .8f * container->tilt;

	projM = rotate(*img, tmp, container->angle);
	// 1-dimensional blur for anti aliasing
	GaussianBlur(tmp, tmp, Size(1, 0), antiAliasing, 0., BORDER_REPLICATE);
	tilt(tmp, projection, container->tilt);
	
	// apply tilt to perspective matrix
	projM.row(1) /= container->tilt;
	projMinv = projM.inv();
}

void Detector::filter() const {
	Size s = img->size();
	float w = (float) s.width;
	float h = (float) s.height;

	// get image corners of projection
	Point2f A = projectPoint( Point2f(0.f, 0.f) );
	Point2f B = projectPoint( Point2f(0.f,   h) );
	Point2f D = projectPoint( Point2f(  w, 0.f) );

	// x and y distances
	float ABx = A.x - B.x;
	float ABy = A.y - B.y;
	float DAx = D.x - A.x;
	float DAy = D.y - A.y;

	// length of line AB
	float AB = sqrt(ABx * ABx + ABy * ABy);
	// length of line DA
	float DA = sqrt(DAx * DAx + DAy * DAy);

	// Calculate the distances of parallel lines. Thus, less calcDistance
	// calls are needed.
	
	// AB times the distance between the lines AB and CD
	float e = calcDistance(A, B, ABx, ABy, D);
	// DA times the distance between the lines DA and BC
	float f = calcDistance(D, A, DAx, DAy, B);

	// The border factor is multiplied with line lengths. This allows the
	// calcDistance method to avoid dividing by A square root. Therefore the
	// square root only needs to be multiplied once instead of divide multiple
	// times.

	// AB times the projectionBorderFactor
	float ABborderFactor = AB * projectionBorderFactor;
	// DA times the projectionBorderFactor
	float DAborderFactor = DA * projectionBorderFactor;

	// the filter condition
	function<bool (const KeyPoint &kp)> condition =
		[this, &A, &B, &D, ABx, ABy, DAx, DAy, e, f, ABborderFactor, DAborderFactor]
		(const KeyPoint &kp) -> bool
	{
		// AB times the distance to the line AB
		float ABP = calcDistance(A, B, ABx, ABy, kp.pt);
		// DA times the distance to the line DA
		float DAP = calcDistance(D, A, DAx, DAy, kp.pt);
		// DA times the distance to the line BC
		float BCP = f - DAP;
		// AB times the distance to the line CD
		float CDP = e - ABP;

		float ABborder = ABborderFactor * kp.size;
		float DAborder = DAborderFactor * kp.size;

		// accept key point if within boundaries
		return
			ABP >= ABborder &&
			DAP >= DAborder &&
			BCP >= DAborder &&
			CDP >= ABborder;
	};

	// actual filtering
	container->siftModel.filter(condition);
}

void Detector::compensateKeyPoints() const {
	// mapper
	function<void (KeyPoint &kp)> mapper = [this] (KeyPoint &kp) {
		kp.pt = compensatePoint(kp.pt);
	};

	// actual mapping
	container->siftModel.map(mapper);
}

float Detector::calcTilt(int tiltIndex) const {
	return TILT_MIN * pow(tiltFactor, tiltIndex);
}

int Detector::calcRotations(float tilt) const {
	return cvRound(tilt * rotationsPerTilt + 1.f) / 2;
}

void Detector::tilt(const Mat &src, Mat &dst, float tilt) const {
	resize(src, dst, Size(), 1., 1./tilt, CV_INTER_AREA);
}

Mat Detector::rotate(const Mat &src, Mat &dst, float angle) const {
	Size s = src.size();
	int w = s.width;
	int h = s.height;

	float cosX  = cos(angle);
	float sinX  = sin(angle);
	float wCosX = w * cosX;
	float hCosX = h * cosX;
	float wSinX = w * sinX;
	float hSinX = h * sinX;

	// new size
	int wRot = cvRound( abs(wCosX) + abs(hSinX) );
	int hRot = cvRound( abs(wSinX) + abs(hCosX) );
	
	// The image will be rotated with the upper left corner as center point.
	// In result some areas of the image will be outside the target area.
	// Therefore a shift is calculated, so that the rotated image is fully visible.

	float dx = 0.f;
	float dy = 0.f;

	if (angle > 0)
		dy += wSinX;
	else
		dx -= hSinX;
	
	if (angle >= CV_PI / 2. || angle <= -CV_PI / 2) {
		dx -= wCosX;
		dy -= hCosX;
	}

	// The rotation and shift is applied to the perspective matrix.

	// create transformation matrix
	Mat M = (Mat_<float>(3, 3) <<
		 cosX, sinX,  dx,
		-sinX, cosX,  dy,
		  0.f,  0.f, 1.f);

	// actual transformation
	warpPerspective(src, dst, M, Size(wRot, hRot));

	return M;
}

Point2f Detector::projectPoint(const Point2f &pt) const {
	Mat src = (Mat_<float>(3, 1) << pt.x, pt.y, 1.f);
	Mat dst = projM * src;

	return dst(Rect(0, 0, 1, 2));
}

Point2f Detector::compensatePoint(const Point2f &pt) const {
	Mat dst = (Mat_<float>(3, 1) << pt.x, pt.y, 1.f);
	Mat src = projMinv * dst;

	return src(Rect(0, 0, 1, 2));
}

float Detector::calcDistance(
	const Point2f &A,
	const Point2f &B,
	      float   ABx,
	      float   ABy,
	const Point2f &P) const
{
	float APx = A.x - P.x;
	float APy = A.y - P.y;

	return ABy * APx - ABx * APy;
}

