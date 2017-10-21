#ifndef ASIFT_DETECTOR_H
#define ASIFT_DETECTOR_H


#include <vector>

#include <opencv2/opencv.hpp>

#include <sift_detector.h>


namespace asift {

struct FeatureContainer;

class Model;

///=================================================================================================
/// <summary>
/// The Detector extracts ASIFT-Features from images. Several (A)SIFT-Parameters can be configured
/// before Detection.
/// </summary>
/// 
/// <remarks> Jasper, 15.08.2012. </remarks>
///-------------------------------------------------------------------------------------------------
class Detector {

public:
	
/******************************************************************************************************* 
 ****** 0. Constructors ********************************************************************************
 *******************************************************************************************************/

	///=================================================================================================
	/// <summary>
	/// Default constructor. Sets default asift parameters.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	Detector();
			
/******************************************************************************************************* 
 ****** I. Public Methods ******************************************************************************
 ****** I.a Setters       ******************************************************************************
 *******************************************************************************************************/

	///=================================================================================================
	/// <summary>
	/// Sets whether an initial upscale will be performed. If set to true an additional octave "-1" is
	/// calculated. Width and height are doubled. If set to false the calculations begin in the 0th
	/// octave with original width and height.
	/// </summary>
	/// 
	/// <param name="upscale"> true to scale up. </param>
	///-------------------------------------------------------------------------------------------------
	void setUpscale(bool upscale);

	///=================================================================================================
	/// <summary>
	/// Sets the border width in which to ignore detected extrema.
	/// </summary>
	/// 
	/// <param name="width"> The width. (width >= 0) </param>
	///-------------------------------------------------------------------------------------------------
	void setBorderWidth(int width);

	///=================================================================================================
	/// <summary>
	/// Sets the number of sampled intervals per octave.
	/// </summary>
	/// 
	/// <param name="intervals">
	/// The number intervals an octave shall consist of. (intervals > INTERVALS_MIN)
	/// </param>
	///-------------------------------------------------------------------------------------------------
	void setIntervals(int intervals);

	///=================================================================================================
	/// <summary>
	/// Sets the standard deviation of the gaussian bluring which is doubled each octave.
	/// </summary>
	/// 
	/// <param name="sigma"> The standard deviation of each interval blurring. (sigma > 0)</param>
	///-------------------------------------------------------------------------------------------------
	void setSigma(float sigma);

	///=================================================================================================
	/// <summary>
	/// Sets the assumed standard deviation of the gaussian blur for the input image.
	/// </summary>
	/// 
	/// <param name="sigmaInit">
	/// The standard deviation of the initial input image blurring. (sigmaInit >= 0)
	/// </param>
	///-------------------------------------------------------------------------------------------------
	void setSigmaInit(float sigmaInit);
	
	///=================================================================================================
	/// <summary>
	/// Sets the contrast threshold. Key points with lower contrast are discarded.
	/// </summary>
	/// 
	/// <param name="threshold"> The threshold. (threshold > 0) </param>
	///-------------------------------------------------------------------------------------------------
	void setContrastThreshold(float threshold);

	///=================================================================================================
	/// <summary>
	/// Sets the curvature threshold of principle curvatures used for edge detection. In general the
	/// smaller the threshold the more sensitive the edge detection.
	/// </summary>
	/// 
	/// <param name="threshold"> The threshold. (threshold > 0) </param>
	///-------------------------------------------------------------------------------------------------
	void setCurvatureThreshold(float threshold);

	///=================================================================================================
	/// <summary>
	/// Sets the orientation sigma factor which is used to calculate the
	/// gaussian weight for the orientation histogram. The octave scale is
	/// multiplied by the factor resulting in the orientation sigma.
	/// </summary>
	/// 
	/// <param name="factor"> The factor. (factor > 0) </param>
	///-------------------------------------------------------------------------------------------------
	void setOrientationSigmaFactor(float factor);

	///=================================================================================================
	/// <summary>
	/// Sets the orientation radius factor which is used to calculate the region size for the
	/// orientation assignment. The radius of the region is determined by multiplying the octave scale
	/// by the factor.
	/// </summary>
	/// 
	/// <param name="factor"> The factor. (factor > 0) </param>
	///-------------------------------------------------------------------------------------------------
	void setOrientationRadiusFactor(float factor);

	///=================================================================================================
	/// <summary>
	/// Sets the orientation peak ratio. An orientation results in a new feature if its
	/// magnitude-maximum ratio is at least as high as the given one.
	/// </summary>
	/// 
	/// <param name="ratio"> The ratio. (ratio \in [0, 1]) </param>
	///-------------------------------------------------------------------------------------------------
	void setOrientationPeakRatio(float ratio);

	///=================================================================================================
	/// <summary>
	/// Sets the number of smooth iterations of the orientation histogramm.
	/// </summary>
	/// 
	/// <param name="iterations"> The number of iterations. (iterations >= 0) </param>
	///-------------------------------------------------------------------------------------------------
	void setOrientationSmoothIterations(int iterations);

	///=================================================================================================
	/// <summary>
	/// Sets the descriptor sigma factor which is used to calculate the size of a descriptor sub
	/// histogram relative to the octave scale.
	/// </summary>
	/// 
	/// <param name="factor"> The factor. (factor > 0) </param>
	///-------------------------------------------------------------------------------------------------
	void setDescriptorSigmaFactor(float factor);

	///=================================================================================================
	/// <summary>
	/// Sets the magnitude threshold for the descriptor vector after normalization.
	/// </summary>
	/// 
	/// <param name="threshold"> The threshold. (factor \in (0, 1)) </param>
	///-------------------------------------------------------------------------------------------------
	void setDescriptorMagnitudeThreshold(float threshold);
	
	///=================================================================================================
	/// <summary> Sets a projection border factor. </summary>
	///
	/// <param name="factor"> The factor. (factor => 0) </param>
	///-------------------------------------------------------------------------------------------------
	void setProjectionBorderFactor(float factor);

	///=================================================================================================
	/// <summary>
	/// Sets the number of tilts to be simulated.
	/// </summary>
	///
	/// <param name="tilts"> The tilts. (tilts >= 1) </param>
	///-------------------------------------------------------------------------------------------------
	void setTilts(int tilts);

	///=================================================================================================
	/// <summary>
	/// Sets a tilt factor. Used to calculate the next nilt from the current.
	/// </summary>
	///
	/// <param name="tiltFactor"> The tilt factor. (factor > 1) </param>
	///-------------------------------------------------------------------------------------------------
	void setTiltFactor(float factor);

	///=================================================================================================
	/// <summary> Sets the rotations per tilt. </summary>
	///
	/// <param name="rotations"> The rotations per tilt. (rotations >= 1) </param>
	///-------------------------------------------------------------------------------------------------
	void setRotationsPerTilt(float rotations);

/******************************************************************************************************* 
 ****** I.b Use Case Methods ***************************************************************************
 *******************************************************************************************************/

	///=================================================================================================
	/// <summary>
	/// Detects ASIFT features in the given input image. The key points and the descriptors are stored
	/// in feature containers for each projection. A key point descriptor pair will have the same index
	/// in their vectors.
	/// </summary>
	///
	/// <param name="img"> The input image. </param>
	/// <param name="features"> [in,out] The feature containers. </param>
	///
	/// <returns> The number of found features. </returns>
	///-------------------------------------------------------------------------------------------------
	int detect(const cv::Mat &img, std::vector<asift::FeatureContainer> &features) const;

	///=================================================================================================
	/// <summary>
	/// Detects ASIFT features in the given input image. The key points and the descriptors are stored
	/// in the given model.
	/// </summary>
	///
	/// <param name="img"> The input image. </param>
	/// <param name="model"> [in, out] The model in which to store the key points and descriptors. </param>
	///
	/// <returns> The number of found features. </returns>
	///-------------------------------------------------------------------------------------------------
	int detect(const cv::Mat &img, asift::Model &model) const;

/******************************************************************************************************* 
 ****** II. Fields    ***********************************************************************************
 ****** II.a Settings **********************************************************************************
 *******************************************************************************************************/

	///=================================================================================================
	/// <summary>
	/// The projection border factor.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	float projectionBorderFactor;

	///=================================================================================================
	/// <summary>
	/// The number of tilts to be simulated.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	int tilts;

	///=================================================================================================
	/// <summary>
	/// The tilt factor.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	float tiltFactor;

	///=================================================================================================
	/// <summary>
	/// The rotations per tilt.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	float rotationsPerTilt;

/******************************************************************************************************* 
 ****** II.b SIFT Detector *****************************************************************************
 *******************************************************************************************************/

	///=================================================================================================
	/// <summary>
	/// The SIFT detector.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	sift::Detector detector;
	
/******************************************************************************************************* 
 ****** II.c State               ***********************************************************************
 ****** II.c (1) Detection State ***********************************************************************
 *******************************************************************************************************/

	mutable const cv::Mat *img;

	mutable std::vector<asift::FeatureContainer> *features;
		
/******************************************************************************************************* 
 ****** II.c (2) Projection State **********************************************************************
 *******************************************************************************************************/

	///=================================================================================================
	/// <summary>
	/// Tracks the current projection number. Is used to identify the current feature container.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	mutable int projCounter;

	///=================================================================================================
	/// <summary>
	/// The current projection matrix.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	mutable cv::Mat projM;

	///=================================================================================================
	/// <summary>
	/// The current projection matrix inversed.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	mutable cv::Mat projMinv;

	///=================================================================================================
	/// <summary>
	/// The current image projection.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	mutable cv::Mat projection;

	///=================================================================================================
	/// <summary>
	/// The current feature container.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	mutable asift::FeatureContainer *container;
	
/******************************************************************************************************* 
 ****** III. Private Methodes          *****************************************************************
 ****** III.a Initializer and Clean Up *****************************************************************
 *******************************************************************************************************/

	///=================================================================================================
	/// <summary>
	/// Initializes the detection process.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	void init() const;
	
/******************************************************************************************************* 
 ****** III.b ASIFT Algorithm   ************************************************************************
 *******************************************************************************************************/

	///=================================================================================================
	/// <summary>
	/// Performs multiple steps until the final tilt and angle have been reached. Calculates the tilts
	/// and angles to be simulated and calls step.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	void loop() const;

	///=================================================================================================
	/// <summary>
	/// Project the input image with the given tilt and angle. Detects SIFT-Features of the projection.
	/// </summary>
	///
	/// <param name="tilt"> The tilt. </param>
	/// <param name="angle"> The angle. </param>
	/// <param name="noFilter"> If false filters SIFT-Features near the projected image border. </param>
	///-------------------------------------------------------------------------------------------------
	void step(float tilt, float angle, bool noFilter) const;
	
	///=================================================================================================
	/// <summary>
	/// Sets the next feature container.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	void nextContainer() const;

	///=================================================================================================
	/// <summary>
	/// Projects the input image.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	void project() const;
	
	///=================================================================================================
	/// <summary>
	/// Filters feature points near the projected image border.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	void filter() const;

	///=================================================================================================
	/// <summary>
	/// Compensate key points coordinates to match with the original image's coordinates.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	void compensateKeyPoints() const;

/******************************************************************************************************* 
 ****** III.c Helpers **********************************************************************************
 *******************************************************************************************************/
	
	///=================================================================================================
	/// <summary>
	/// Calculates the tilt with the given tilt index.
	/// </summary>
	///
	/// <param name="tiltIndex"> Zero-based index of the tilt. </param>
	///
	/// <returns> The calculated tilt. </returns>
	///-------------------------------------------------------------------------------------------------
	float calcTilt(int tiltIndex) const;

	///=================================================================================================
	/// <summary> Calculates the number of rotations of the given tilt. </summary>
	///
	/// <param name="tilt"> The tilt. </param>
	///
	/// <returns> The calculated rotations. </returns>
	///-------------------------------------------------------------------------------------------------
	int calcRotations(float tilt) const;

	///=================================================================================================
	/// <summary>
	/// Tilts the source image by the given tilt.
	/// </summary>
	///
	/// <param name="src"> The input image. </param>
	/// <param name="dst"> [in,out] The destination image. </param>
	/// <param name="tilt"> The tilt. </param>
	///-------------------------------------------------------------------------------------------------
	void tilt(const cv::Mat &src, cv::Mat &dst, float tilt) const;
	
	///=================================================================================================
	/// <summary>
	/// Rotates the source image by the given angle.
	/// </summary>
	///
	/// <param name="src"> The input image. </param>
	/// <param name="dst"> [in,out] destination image. </param>
	/// <param name="angle"> The angle. </param>
	/// 
	/// <returns> The perspective Matrix rotating the input image. </returns>
	///-------------------------------------------------------------------------------------------------
	cv::Mat rotate(const cv::Mat &src, cv::Mat &dst, float angle) const;

	///=================================================================================================
	/// <summary>
	/// Projects the given point.
	/// </summary>
	///
	/// <param name="pt"> The point to be projected. </param>
	///
	/// <returns> The projected point </returns>
	///-------------------------------------------------------------------------------------------------
	cv::Point2f projectPoint(const cv::Point2f &pt) const;

	///=================================================================================================
	/// <summary>
	/// Compensate the given point. This is inverse to the method projectPoint.
	/// </summary>
	///
	/// <param name="pt"> The point to be compensated. </param>
	///
	/// <returns> The compensated point. </returns>
	///-------------------------------------------------------------------------------------------------
	cv::Point2f compensatePoint(const cv::Point2f &pt) const;

	///=================================================================================================
	/// <summary>
	/// Calculates AB times the distance of the point P to the line connecting A and B, where
	/// AB is the distance between AB.
	/// </summary>
	///
	/// <param name="A"> Point A of the line. </param>
	/// <param name="B"> Point B of the line. </param>
	/// <param name="ABx"> The x-distance of A and B. </param>
	/// <param name="ABy"> The y-distance of A and B. </param>
	/// <param name="P"> Point P. </param>
	///
	/// <returns> The relative distance of P to the line connecting A and B. </returns>
	///-------------------------------------------------------------------------------------------------
	float calcDistance(
		const cv::Point2f &A,
		const cv::Point2f &B,
		      float       ABx,
		      float       ABy,
		const cv::Point2f &P) const;

};

}


#endif // ASIFT_DETECTOR_H