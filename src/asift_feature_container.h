#ifndef ASIFT_FEATURE_CONTAINER_H
#define ASIFT_FEATURE_CONTAINER_H


#include <sift_model.h>


namespace asift {

///=================================================================================================
/// <summary> The Feature Container stores SIFT-Features of a single projection. </summary>
///
/// <remarks> Jasper, 06.09.2012. </remarks>
///-------------------------------------------------------------------------------------------------
struct FeatureContainer {

	/// <summary> The tilt of the projection. </summary>
	float tilt;

	/// <summary> The angle of the projection. </summary>
	float angle;

	/// <summary> The sift model storing the features of the projection. </summary>
	sift::Model siftModel;

};


}


#endif // ASIFT_FEATURE_CONTAINER_H