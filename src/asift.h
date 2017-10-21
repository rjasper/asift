#ifndef ASIFT_H
#define ASIFT_H


#include <math.h>


namespace asift {


/// <summary> The projection border factor. </summary>
const float PROJECTION_BORDER_FACTOR = 6.f * sqrt(2.f);
	
/// <summary> The number of tilts to be simulated. </summary>
const int TILTS = 7;

/// <summary> The tilt factor. </summary>
const float TILT_FACTOR = sqrt(2.f);

/// <summary> The tilt minimum. </summary>
const int TILT_MIN = 1;

/// <summary> The rotations per tilt. </summary>
const float ROTATIONS_PER_TILT = 5.f;

/// <summary> The radius wherein points are considered similar. </summary>
const float MATCH_SIMILAR_POINTS_RADIUS = 2.f; // sqrt(2.f);

/// <summary> The minimum subgroup size. Smaller subgroups are rejected. </summary>
const int MATCH_MIN_SUBGROUP_SIZE = 2;


}


#endif // ASIFT_H