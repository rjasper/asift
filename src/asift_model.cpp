#include "asift_model.h"

#include <stdint.h>
#include <float.h>

#include "asift.h"
#include "asift_feature_container.h"
#include "asift_io.h"
#include "asift_detector.h"

#include "sift_model.h"

using namespace std;
using namespace cv;
using namespace asift;


enum {
	QUERY_POINT,
	TRAINING_POINT
};


Model::Model() { }

Model::Model(Model &&model) {
	*this = move(model);
}

Model::Model(const vector<FeatureContainer> &features)
	: features ( features )
{
	flattenFeatures();
}

Model::Model(const vector<FeatureContainer> &&features)
	: features ( move(features) )
{
	flattenFeatures();
}

void Model::setFeatures(const vector<FeatureContainer> &features) {
	clear(); // may be not necessary but doesn't hurt

	this->features = features;

	flattenFeatures();
}

void Model::setFeatures(const vector<FeatureContainer> &&features) {
	clear(); // may be not necessary but doesn't hurt

	this->features = move(features);

	flattenFeatures();
}

Model &Model::operator=(Model &&rhs) {
	// don't move to yourself o_O
	if (&rhs == this)
		return *this;

	features    = move(rhs.features);
	keyPoints   = move(rhs.keyPoints);
	offsets     = move(rhs.offsets);
	descriptors = rhs.descriptors; // doesn't support move yet

	rhs.descriptors.release();

	return *this;
}

int Model::detect(const asift::Detector &detector, const cv::Mat &img) {
	clear(); // may be not necessary but doesn't hurt

	int n = detector.detect(img, features);

	flattenFeatures();

	return n;
}

int Model::match(const Model &model, std::vector<cv::DMatch> &matches) const {
	// matches between projections (projection matches)
	vector<vector<DMatch>> pMatches;
	// flattened projection matches
	vector<const DMatch *> flattened, refined;

	// match each projection pair's features
	int n = matchProjections(model, pMatches); // returns number of matches
	// flatten the nested matches (2D to 1D)
	flatten(pMatches, offsets, model.offsets, flattened, n);
	// remove redundant matches
	refineMatches(model, flattened, refined);
	// convert pointer array to object array
	buildMatches(refined, matches);

	return matches.size();
}

void Model::clear() {
	features.clear();
	keyPoints.clear();
	descriptors.release();
	offsets.clear();
}

int Model::matchProjections(
	const Model &model,
	      vector<vector<DMatch>> &pMatches) const
{
	// aliases
	const vector<FeatureContainer> &f1 = features;
	const vector<FeatureContainer> &f2 = model.features;
	
	// number of projections
	int np1 = f1.size();
	int np2 = f2.size();

	// matches between projections (projection matches)
	pMatches.resize(np1 * np2);

	// number of matches (counter)
	int nm = 0;
	// for each projection of the train model
	int offset = 0;
	for (int i = 0; i < np1; ++i) {
		const sift::Model &sm1 = f1[i].siftModel;

		// for each projection of the query model
		for (int j = 0; j < np2; ++j) {
			const sift::Model &sm2 = f2[j].siftModel;
			vector<DMatch> &pm = pMatches[offset + j];

			sm1.match(sm2, pm);

			nm += pm.size();
		}

		offset += np2;
	}

	return nm;
}

void Model::flatten(
	      vector<vector<DMatch>> &pMatches,
	const vector<int>            &trainOffsets,
	const vector<int>            &queryOffsets,
	      vector<const DMatch *> &flattened,
	      int nm) const
{
	// number of projections
	int np1 = trainOffsets.size();
	int np2 = queryOffsets.size();

	flattened.resize(nm);

	vector<const DMatch *>::iterator m = flattened.begin();
	vector<vector<DMatch>>::iterator pm = pMatches.begin();
	// for each projection of the train model
	for (int i = 0; i < np1; ++i) {
		// for each projection of the query model
		for (int j = 0; j < np2; ++j) {
			// number of the projection pair's matches
			int npm = pm->size();

			// for each match
			for (int k = 0; k < npm; ++k) {
				DMatch &match = (*pm)[k];
				
				// adjust key point indices to flattened key points
				match.trainIdx += trainOffsets[i];
				match.queryIdx += queryOffsets[j];

				// copy match from current projection to flattened matches
				*m = &match;

				++m;
			}

			++pm;
		}
	}
}

void Model::flattenFeatures() {
	// number of projections
	int np = features.size();

	offsets.resize(np);

	// count key points
	int n = 0;
	for (int i = 0; i < np; ++i) {
		offsets[i] = n;
		n += features[i].siftModel.getKeyPoints().size();
	}

	// prepare flattened data structures
	keyPoints.resize(n);
	descriptors = Mat(n, sift::DESCRIPTOR_LENGTH, CV_32F);
	
	// iterator of the flattened key points
	vector<KeyPoint>::iterator &kFlat = keyPoints.begin();
	// for each projection
	for (int i = 0; i < np; ++i) {
		// aliases
		const FeatureContainer &f = features[i];
		const vector<KeyPoint> &k = f.siftModel.getKeyPoints();
		const Mat              &d = f.siftModel.getDescriptors();

		vector<KeyPoint>::const_iterator
			&begin = k.begin(),
			&end   = k.end(),
			ki;
		// for each key point
		for (ki = begin; ki != end; ++ki) {
			*kFlat = *ki;
			++kFlat;
		}
		
		// copy descriptors
		int rows = d.rows;
		Rect roi(0, offsets[i], sift::DESCRIPTOR_LENGTH, rows);
		d.copyTo( descriptors(roi) );
	}
}

void Model::refineMatches(
	const Model &model,
	const vector<const DMatch *> &src,
	      vector<const DMatch *> &dst) const
{
	// list of accepted matches
	list<const DMatch *> refined;
	// groups of matches with similar query points
	vector<vector<const DMatch *>> qryPtGroups;

	// detect similar query point groups
	model.detectGroups(src, qryPtGroups, QUERY_POINT);

	vector<vector<const DMatch *>>::iterator
		&qbegin = qryPtGroups.begin(),
		&qend   = qryPtGroups.end(),
		qryGroup;
	// for each group of similar query points
	for (qryGroup = qbegin; qryGroup != qend; ++qryGroup) {
		// groups of matches with similar training points
		vector<vector<const DMatch *>> trnPtGroups;

		// detect similar training point subgroups
		detectGroups(*qryGroup, trnPtGroups, TRAINING_POINT);

		// select one match or none from the current subgroups
		const DMatch *unified = unifyGroup(trnPtGroups);

		if (unified)
			refined.push_back(unified);
	}

	// convert list to vector
	dst.assign( refined.begin(), refined.end() );
}

void Model::extractPoints(
	const vector<const DMatch *> &matches,
	      Mat &pts,
		  int point) const
{
	int n = matches.size();

	pts = Mat(n, 2, CV_32F);

	if (point == QUERY_POINT) {
		// for each match
		for (int i = 0; i < n; ++i) {
			const DMatch   &m  = *matches[i];
			const KeyPoint &kp = keyPoints[m.queryIdx];
			const Point2f  &pt = kp.pt;

			pts.at<float>(i, 0) = pt.x;
			pts.at<float>(i, 1) = pt.y;
		}
	} else { // point == TRAINING_POINT
		// for each match
		for (int i = 0; i < n; ++i) {
			const DMatch   &m  = *matches[i];
			const KeyPoint &kp = keyPoints[m.trainIdx];
			const Point2f  &pt = kp.pt;

			pts.at<float>(i, 0) = pt.x;
			pts.at<float>(i, 1) = pt.y;
		}
	}
}

void Model::detectGroups(
	const vector<const DMatch *> &matches,
	      vector<vector<const DMatch *>> &groups,
	      int point) const
{
	Mat pts; // extracted points
	list<list<DMatch>> similarPts;

	extractPoints(matches, pts, point);
	detectSimilarPoints(pts, similarPts);
	mapGroups(similarPts, matches, groups);
}

void Model::detectSimilarPoints(
	const Mat &pts,
	      list<list<DMatch>> &similarPts) const
{
	// number of points
	int n = pts.rows;

	vector<vector<DMatch>> similarPtMatches;
	FlannBasedMatcher flann;

	// find neighbors of each point
	// note that each point will at least find itself (lucky point ;)
	flann.radiusMatch(pts, pts, similarPtMatches, MATCH_SIMILAR_POINTS_RADIUS);

	// The following loop will group the neighbors. The resulting list of
	// groups will include each point only once.

	// marks points which are already grouped
	vector<bool> grouped(n, false);
	// for each point
	for (int i = 0; i < n; ++i) {
		const vector<DMatch> &nbrs = similarPtMatches[i];

		// discard point if already grouped
		if (grouped[i])
			continue;

		// current point's number of similarPtMatches
		int m = nbrs.size();
		// current group (avoiding copying a large list afterwards)
		similarPts.push_back(list<DMatch>());
		list<DMatch> &group = similarPts.back();

		// for each neighbor (including the current point)
		for (int j = 0; j < m; ++j) {
			const DMatch &nbr = nbrs[j];
			int idx = nbr.trainIdx; // index of the neighbor

			// discard point if already grouped
			if (grouped[idx])
				continue;

			group.push_back(nbr);
			grouped[idx] = true;
		}
	}
}

void Model::mapGroups(
	const list<list<DMatch>>             &src,
	const vector<const DMatch *>         &map,
	      vector<vector<const DMatch *>> &dst) const
{
	dst.resize( src.size() );

	// prepare outer iterators
	list<list<DMatch>>::const_iterator
		&sbegin = src.begin(),
		&send   = src.end(),
		s; // src group
	vector<vector<const DMatch *>>::iterator
		&dbegin = dst.begin(),
		d; // dst group

	// for each group
	for (s = sbegin, d = dbegin; s != send; ++s, ++d) {
		d->resize( s->size() );

		// prepare inner iterators
		list<DMatch>::const_iterator
			&ssbegin = s->begin(),
			&ssend   = s->end(),
			ss;
		vector<const DMatch *>::iterator
			&ddbegin = d->begin(),
			dd;
		// for each match
		for (ss = ssbegin, dd = ddbegin; ss != ssend; ++ss, ++dd)
			*dd = map[ss->trainIdx];
	}
}

const DMatch *Model::unifyGroup(const vector<vector<const DMatch *>> &group) const {
	// currently selected match to represent the group
	const DMatch *sel = nullptr;
	// current minimum value
	float min = FLT_MAX;

	vector<vector<const DMatch *>>::const_iterator
		&begin = group.begin(),
		&end   = group.end(),
		subgroup;
	// for each subgroup
	for (subgroup = begin; subgroup != end; ++subgroup) {
		// drop non-repetitive matches
		if (subgroup->size() < MATCH_MIN_SUBGROUP_SIZE)
			continue;

		const DMatch *unifiedSubgroup = unifySubgroup(*subgroup);
		float value = unifiedSubgroup->distance;

		if (value < min) {
			min = value;
			sel = unifiedSubgroup;
		}
	}

	return sel;
}

const DMatch *Model::unifySubgroup(const vector<const DMatch *> &subgroup) const {
	// currently selected match to represent the group
	const DMatch *sel = nullptr;
	// current minimum value
	float min = FLT_MAX;

	vector<const DMatch *>::const_iterator
		&begin = subgroup.begin(),
		&end   = subgroup.end(),
		match;
	// for each match
	for (match = begin; match != end; ++match) {
		float value = (*match)->distance;

		if (value < min) {
			min = value;
			sel = *match;
		}
	}

	return sel;
}

void Model::buildMatches(
	const vector<const DMatch *> &pointers,
	      vector<DMatch>         &matches) const
{
	matches.resize( pointers.size() );

	vector<const DMatch *>::const_iterator
		&pbegin = pointers.begin(),
		&pend   = pointers.end(),
		ptr;
	vector<DMatch>::iterator
		&mbegin = matches.begin(),
		match;
	// for each pointer
	for (ptr = pbegin, match = mbegin; ptr != pend; ++ptr, ++match)
		*match = **ptr;
}

void Model::read(istream &in) {
	readFeatures(in, features);
	flattenFeatures();
}

void Model::write(ostream &out) const {
	writeFeatures(out, features);
}
