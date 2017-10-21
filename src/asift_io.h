#ifndef ASIFT_IO_H
#define ASIFT_IO_H


#include <iostream>
#include <vector>

#include "asift_model.h"


namespace asift {


struct FeatureContainer;


///=================================================================================================
/// <summary> Writes the features to an output stream. </summary>
///
/// <param name="out"> [in,out] The output stream. </param>
/// <param name="features"> The features. </param>
///-------------------------------------------------------------------------------------------------
void writeFeatures(
	      std::ostream                  &out,
	const std::vector<FeatureContainer> &features);

///=================================================================================================
/// <summary> Reads features from an input stream. </summary>
///
/// <param name="in"> [in,out] The input stream. </param>
/// <param name="features"> [in,out] The features. </param>
///-------------------------------------------------------------------------------------------------
void readFeatures(
	std::istream                  &in,
	std::vector<FeatureContainer> &features);

///=================================================================================================
/// <summary>
/// Writes a model to an output stream.
/// </summary>
///
/// <param name="out"> [in,out] The output stream. </param>
/// <param name="model"> The model. </param>
///
/// <returns> The output stream. </returns>
///-------------------------------------------------------------------------------------------------
inline std::ostream &operator<<(std::ostream &out, const Model &model) {
	model.write(out);
	return out;
}

///=================================================================================================
/// <summary>
/// Extracts a model from an input stream.
/// </summary>
///
/// <param name="in"> [in,out] The input stream. </param>
/// <param name="model"> [in,out] The model to be loaded from the stream. </param>
///
/// <returns> The input stream. </returns>
///-------------------------------------------------------------------------------------------------
inline std::istream &operator>>(std::istream &in, Model &model) {
	model.read(in);
	return in;
}


}


#endif // ASIFT_IO_H