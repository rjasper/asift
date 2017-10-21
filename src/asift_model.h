#ifndef ASIFT_MODEL_H
#define ASIFT_MODEL_H


#include <functional>

#include <vector>
#include <list>

#include "asift_feature_container.h"


namespace sift {

class Model;

}


namespace asift {

class Detector;

///=================================================================================================
/// <summary>
/// The Model stores the key points and descriptors of an image object. You can compare a model to
/// others for object recognition.
/// asift_io.h provides I/O functions for this class.
/// </summary>
///
/// <remarks> Jasper, 06.09.2012. </remarks>
///-------------------------------------------------------------------------------------------------
class Model {

public:
		
/******************************************************************************************************* 
 ****** 0. Constructors and Destructors ****************************************************************
 *******************************************************************************************************/

	///=================================================================================================
	/// <summary>
	/// Default constructor. Initializes an empty set of key points and descriptors.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	Model();

	///=================================================================================================
	/// <summary>
	/// Move constructor.
	/// </summary>
	///
	/// <param name="model"> The model to be moved. </param>
	///-------------------------------------------------------------------------------------------------
	Model(Model &&model);

	///=================================================================================================
	/// <summary>
	/// Constructs a model from the given set of feature containers.
	/// </summary>
	///
	/// <param name="features"> The features. </param>
	///-------------------------------------------------------------------------------------------------
	Model(const std::vector<asift::FeatureContainer> &features);
	
	///=================================================================================================
	/// <summary>
	/// Constructs a model by moving the given set of feature containers.
	/// </summary>
	///
	/// <param name="features"> The features. </param>
	///-------------------------------------------------------------------------------------------------
	Model(const std::vector<asift::FeatureContainer> &&features);

	///=================================================================================================
	/// <summary>
	/// Move assignment operator.
	/// </summary>
	///
	/// <param name="rhs"> [in,out] The right hand side. </param>
	///
	/// <returns> Returns a reference to this object. </returns>
	///-------------------------------------------------------------------------------------------------
	Model &operator=(Model &&rhs);

/******************************************************************************************************* 
 ****** I. Public Methods ******************************************************************************
 ****** I.a Getters       ******************************************************************************
 *******************************************************************************************************/

	///=================================================================================================
	/// <summary>
	/// Queries if this object is empty.
	/// </summary>
	///
	/// <returns> true if empty, false if not. </returns>
	///-------------------------------------------------------------------------------------------------
	bool isEmpty() const;

	///=================================================================================================
	/// <summary>
	/// Gets the features.
	/// </summary>
	///
	/// <returns> The features. </returns>
	///-------------------------------------------------------------------------------------------------
	const std::vector<asift::FeatureContainer> &getFeatures() const;
	
	///=================================================================================================
	/// <summary>
	/// Gets the key points.
	/// </summary>
	///
	/// <returns> The key points. </returns>
	///-------------------------------------------------------------------------------------------------
	const std::vector<cv::KeyPoint> &getKeyPoints() const;

	///=================================================================================================
	/// <summary>
	/// Gets the descriptors.
	/// </summary>
	///
	/// <returns> The descriptors. </returns>
	///-------------------------------------------------------------------------------------------------
	const cv::Mat &getDescriptors() const;
	
	///=================================================================================================
	/// <summary>
	/// Returns the amount of features.
	/// </summary>
	///
	/// <returns> The amount of features. </returns>
	///-------------------------------------------------------------------------------------------------
	int size() const;

/******************************************************************************************************* 
 ****** I.b Setters ************************************************************************************
 *******************************************************************************************************/

	///=================================================================================================
	/// <summary> Sets the features of this model. </summary>
	///
	/// <param name="features"> The features. </param>
	///-------------------------------------------------------------------------------------------------
	void setFeatures(const std::vector<asift::FeatureContainer> &features);

	///=================================================================================================
	/// <summary> Moves the features of this model. </summary>
	///
	/// <param name="features"> The features. </param>
	///-------------------------------------------------------------------------------------------------
	void setFeatures(const std::vector<asift::FeatureContainer> &&features);

/******************************************************************************************************* 
 ****** I.c Use Case Methods ***************************************************************************
 *******************************************************************************************************/
	
	///=================================================================================================
	/// <summary>
	/// Extracts the model from an input stream.
	/// </summary>
	///
	/// <param name="in"> [in,out] The input stream. </param>
	///
	/// <returns> The input stream. </returns>
	///-------------------------------------------------------------------------------------------------
	void read(std::istream &in);

	///=================================================================================================
	/// <summary>
	/// Writes the model to an output stream.
	/// </summary>
	///
	/// <param name="out"> [in,out] The output stream. </param>
	///
	/// <returns> The output stream. </returns>
	///-------------------------------------------------------------------------------------------------
	void write(std::ostream &out) const;

	///=================================================================================================
	/// <summary>
	/// Detects a model from the given image using the given detector.
	/// </summary>
	///
	/// <param name="detector"> The detector. </param>
	/// <param name="img"> The image. </param>
	///
	/// <returns> The number of found features </returns>
	///-------------------------------------------------------------------------------------------------
	int detect(const asift::Detector &detector, const cv::Mat &img);
	
	///=================================================================================================
	/// <summary>
	/// Matches the given model against this object.
	/// </summary>
	///
	/// <param name="model"> The model. </param>
	/// <param name="matches"> [in,out] The matches. </param>
	///
	/// <returns> The number of matches </returns>
	///-------------------------------------------------------------------------------------------------
	int match(const Model &model, std::vector<cv::DMatch> &matches) const;
	
	///=================================================================================================
	/// <summary>
	/// Clears this object to its blank/initial state.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	void clear();

private:
			
/******************************************************************************************************* 
 ****** II Fields **************************************************************************************
 *******************************************************************************************************/

	/// <summary>
	/// The features of each projection.
	/// </summary>
	std::vector<asift::FeatureContainer> features;

	///=================================================================================================
	/// <summary>
	/// The key points. The indices comply with the rows of descriptors.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	std::vector<cv::KeyPoint> keyPoints;

	///=================================================================================================
	/// <summary>
	/// The descriptors. The rows comply with the indices of keyPoints.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	cv::Mat descriptors;

	/// <summary>
	/// The offsets of each projection's key point and descriptor indices. For proper matching the
	/// key points and descriptors need to be separated by their projections.
	/// </summary>
	std::vector<int> offsets;
	
/******************************************************************************************************* 
 ****** III.a Matching Algorithm ***********************************************************************
 *******************************************************************************************************/

	///=================================================================================================
	/// <summary>
	/// Matches the features of each projection pair.
	/// </summary>
	///
	/// <param name="model"> The model to match against. </param>
	/// <param name="pMatches"> [in,out] The matches of each projection pair (projection matches). </param>
	///
	/// <returns> The number of matches. </returns>
	///-------------------------------------------------------------------------------------------------
	int matchProjections(
		const Model &model,
		      std::vector<std::vector<cv::DMatch>> &pMatches) const;

	///=================================================================================================
	/// <summary>
	/// Flattens the given projection matches to a 1D vector. The offsets are the accumulated number of
	/// matches of each projection. Those are used to adjust the indices of each match in pMatches to
	/// the flattened version.
	/// </summary>
	///
	/// <param name="pMatches"> The matches of each projection pair. </param>
	/// <param name="trainOffsets"> The training model index offsets. </param>
	/// <param name="queryOffsets"> The query model index offsets. </param>
	/// <param name="flattened"> The flattened match pointer vector. </param>
	/// <param name="nm"> The number of matches. </param>
	///-------------------------------------------------------------------------------------------------
	void flatten(
		      std::vector<std::vector<cv::DMatch>> &pMatches,
		const std::vector<int>                     &trainOffsets,
		const std::vector<int>                     &queryOffsets,
		      std::vector<const cv::DMatch *>      &flattened,
		      int nm) const;

	///=================================================================================================
	/// <summary>
	/// Refines the matches by rejecting redundant matches and non-repetitve matches.
	/// </summary>
	///
	/// <param name="model"> The query model. </param>
	/// <param name="src"> The unrefined matches. </param>
	/// <param name="dst"> The refined matches. </param>
	///-------------------------------------------------------------------------------------------------
	void refineMatches(
		const Model &model,
		const std::vector<const cv::DMatch *> &src,
		      std::vector<const cv::DMatch *> &dst) const;

	///=================================================================================================
	/// <summary>
	/// Detects groups of similar points of matches. The point flag determines if the training or the
	/// query points shall be similar.
	/// </summary>
	///
	/// <param name="matches"> The matches to be grouped. </param>
	/// <param name="groups"> The grouped matches. </param>
	/// <param name="point"> Flag: TRAINING_POINT or QUERY_POINT </param>
	///-------------------------------------------------------------------------------------------------
	void detectGroups(
		const std::vector<const cv::DMatch *>              &matches,
		      std::vector<std::vector<const cv::DMatch *>> &groups,
		      int point) const;

	///=================================================================================================
	/// <summary>
	/// Extracts the points from the given matches. The point flag determines if the training
	/// or the query points are extracted.
	/// </summary>
	///
	/// <param name="matches"> The matches from which the points are extracted. </param>
	/// <param name="pts"> [in,out] The extracted points. </param>
	/// <param name="point"> Flag: TRAINING_POINT or QUERY_POINT. </param>
	///-------------------------------------------------------------------------------------------------
	void extractPoints(
		const std::vector<const cv::DMatch *> &matches,
		      cv::Mat &pts,
		      int point) const;

	///=================================================================================================
	/// <summary>
	/// Detects groups similar points. Two points are similar if their distance is below a certain
	/// threshold.
	/// </summary>
	///
	/// <param name="pts"> The points to be grouped. </param>
	/// <param name="similarPts"> [in,out] The similar points match vector. </param>
	///-------------------------------------------------------------------------------------------------
	void detectSimilarPoints(
		const cv::Mat &pts,
		      std::list<std::list<cv::DMatch>> &similarPts) const;

	///=================================================================================================
	/// <summary>
	/// Maps a group of similar point to a equivalent group of similar matches.
	/// </summary>
	///
	/// <param name="src"> The grouped points. </param>
	/// <param name="map"> The map of match pointers. </param>
	/// <param name="dst"> The grouped match pointers. </param>
	///-------------------------------------------------------------------------------------------------
	void mapGroups(
		const std::list<std::list<cv::DMatch>>             &src,
		const std::vector<const cv::DMatch *>              &map,
		      std::vector<std::vector<const cv::DMatch *>> &dst) const;

	///=================================================================================================
	/// <summary> Unifies a group of subgroup of matches. </summary>
	///
	/// <param name="group"> The group of matches. </param>
	///
	/// <returns> the unified match of the given group, or null if rejected. </returns>
	///-------------------------------------------------------------------------------------------------
	const cv::DMatch *unifyGroup(
		const std::vector<std::vector<const cv::DMatch *>> &group) const;

	///=================================================================================================
	/// <summary>
	/// Unifyies subgroup of matches.
	/// </summary>
	///
	/// <param name="subgroup"> The subgroup of matches. </param>
	///
	/// <returns> the unified match of the given subgroup. </returns>
	///-------------------------------------------------------------------------------------------------
	const cv::DMatch *unifySubgroup(
		const std::vector<const cv::DMatch *> &subgroup) const;

	///=================================================================================================
	/// <summary>
	/// Builds the matches from the given vector of match pointers.
	/// </summary>
	///
	/// <param name="pointers"> The match pointers. </param>
	/// <param name="matches"> The build matches. </param>
	///-------------------------------------------------------------------------------------------------
	void buildMatches(
		const std::vector<const cv::DMatch *> &pointers,
		      std::vector<cv::DMatch>         &matches) const;

/******************************************************************************************************* 
 ****** III.b Miscellaneous Methods ********************************************************************
 *******************************************************************************************************/

	///=================================================================================================
	/// <summary>
	/// Flattens features of this object. Also calculates the resulting offsets.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	void flattenFeatures();

};


/******************************************************************************************************* 
 ****** Inline Definitions *****************************************************************************
 *******************************************************************************************************/

inline bool Model::isEmpty() const {
	return features.size() == 0;
}

inline const std::vector<asift::FeatureContainer> &Model::getFeatures() const {
	return features;
}

inline const std::vector<cv::KeyPoint> &Model::getKeyPoints() const {
	return keyPoints;
}

inline const cv::Mat &Model::getDescriptors() const {
	return descriptors;
}

inline int Model::size() const {
	return keyPoints.size();
}

}


#endif // ASIFT_MODEL_H