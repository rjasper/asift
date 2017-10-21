
#include "asift_io.h"

#include <sift_io.h>

#include "asift_feature_container.h"

using namespace std;
using namespace sift;


namespace asift {


void writeFeatures(
	      ostream                  &out,
	const vector<FeatureContainer> &features)
{
	out << features.size() << endl;

	vector<FeatureContainer>::const_iterator
		&begin = features.begin(),
		&end   = features.end(),
		fc;

	for (fc = begin; fc != end; ++fc) {
		out << fc->tilt <<" "<< fc->angle << endl;
		out << fc->siftModel;
	}
}

void readFeatures(
	istream                  &in,
	vector<FeatureContainer> &features)
{
	int n;
	in >> n;
	features.resize(n);

	vector<FeatureContainer>::iterator
		&begin = features.begin(),
		&end   = features.end(),
		fc;

	for (fc = begin; fc != end; ++fc) {
		in >> fc->tilt >> fc->angle;
		in >> fc->siftModel;
	}
}


}
