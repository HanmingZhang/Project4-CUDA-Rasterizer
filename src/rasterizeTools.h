/**
 * @file      rasterizeTools.h
 * @brief     Tools/utility functions for rasterization.
 * @authors   Yining Karl Li
 * @date      2012-2015
 * @copyright University of Pennsylvania
 */

#pragma once

#include <cmath>
#include <glm/glm.hpp>
#include <util/utilityCore.hpp>
//
//struct AABB {
//    glm::vec3 min;
//    glm::vec3 max;
//};

/**
 * Multiplies a glm::mat4 matrix and a vec4.
 */
//__host__ __device__ static
//glm::vec3 multiplyMV(glm::mat4 m, glm::vec4 v) {
//    return glm::vec3(m * v);
//}

// CHECKITOUT
/**
 * Finds the axis aligned bounding box for a given triangle.
 */
//__host__ __device__ static
//AABB getAABBForTriangle(const glm::vec3 tri[3]) {
//    AABB aabb;
//    aabb.min = glm::vec3(
//            min(min(tri[0].x, tri[1].x), tri[2].x),
//            min(min(tri[0].y, tri[1].y), tri[2].y),
//            min(min(tri[0].z, tri[1].z), tri[2].z));
//    aabb.max = glm::vec3(
//            max(max(tri[0].x, tri[1].x), tri[2].x),
//            max(max(tri[0].y, tri[1].y), tri[2].y),
//            max(max(tri[0].z, tri[1].z), tri[2].z));
//    return aabb;
//}

// CHECKITOUT
/**
 * Calculate the signed area of a given triangle.
 */
__host__ __device__ static
float calculateSignedArea(glm::vec3 tri[3]) {
    return 0.5 * ((tri[2].x - tri[0].x) * (tri[1].y - tri[0].y) - (tri[1].x - tri[0].x) * (tri[2].y - tri[0].y));
}

// CHECKITOUT
/**
 * Helper function for calculating barycentric coordinates.
 */
__host__ __device__ static
float calculateBarycentricCoordinateValue(glm::vec2 a, glm::vec2 b, glm::vec2 c, glm::vec3 tri[3]) {
    glm::vec3 baryTri[3];
    baryTri[0] = glm::vec3(a, 0);
    baryTri[1] = glm::vec3(b, 0);
    baryTri[2] = glm::vec3(c, 0);
    return calculateSignedArea(baryTri) / calculateSignedArea(tri);
}

// CHECKITOUT
/**
 * Calculate barycentric coordinates.
 */
__host__ __device__ static
glm::vec3 calculateBarycentricCoordinate(glm::vec3 tri[3], glm::vec2 point) {
    float beta  = calculateBarycentricCoordinateValue(glm::vec2(tri[0].x, tri[0].y), point, glm::vec2(tri[2].x, tri[2].y), tri);
    float gamma = calculateBarycentricCoordinateValue(glm::vec2(tri[0].x, tri[0].y), glm::vec2(tri[1].x, tri[1].y), point, tri);
    float alpha = 1.0 - beta - gamma;
    return glm::vec3(alpha, beta, gamma);
}

// CHECKITOUT
/**
 * Check if a barycentric coordinate is within the boundaries of a triangle.
 */
__host__ __device__ static
bool isBarycentricCoordInBounds(const glm::vec3 barycentricCoord) {
    return barycentricCoord.x >= 0.0 && barycentricCoord.x <= 1.0 &&
           barycentricCoord.y >= 0.0 && barycentricCoord.y <= 1.0 &&
           barycentricCoord.z >= 0.0 && barycentricCoord.z <= 1.0;
}

// CHECKITOUT
/**
 * For a given barycentric coordinate, compute the corresponding z position
 * (i.e. depth) on the triangle.
 */
//__host__ __device__ static
//float getZAtCoordinate(const glm::vec3 barycentricCoord, const glm::vec3 tri[3]) {
//    return -(barycentricCoord.x * tri[0].z
//           + barycentricCoord.y * tri[1].z
//           + barycentricCoord.z * tri[2].z);
//}

//__device__
//int conductDepthTest(int screen_space_x, int screen_space_y, glm::vec3 t1, glm::vec3 t2, glm::vec3 t3, int fragmentIdx, int* depth) {
//	//IMPORTANT! 
//	//Should interpolate Z (depth) value first
//	//z (depth) value is in camera(eye) space
//	float lerp_depth = depthValuePersCorrectionLerp(glm::vec3(screen_space_x, screen_space_y, 0.f), t1, t2, t3);
//
//	// OK... atomicMin only works for Int
//	// but we want more accuracy for our depth value
//	// so TRICK here!
//	// multiply a really large number to get accuracy
//	// just pay attention integar is between -2147483648 - 2147483647
//	// 10000 may be a acceptable number
//	int lerp_depth_int = (int)(lerp_depth * 10000.0f);
//
//	// Atomic depth buffer writing
//	int old = depth[fragmentIdx];
//	int assumed;
//
//	do {
//		assumed = old;
//		old = atomicMin(&depth[fragmentIdx], lerp_depth_int);
//	} while (assumed != old);
//
//	return lerp_depth_int;
//}

// test whether position p is in the triangle formed by p1, p2, p3
__device__
bool isPosInTriange(glm::vec3 p,
	glm::vec3 p1, glm::vec3 p2, glm::vec3 p3) {
	glm::vec3 v(p[0], p[1], 0.f);
	glm::vec3 v1(p1[0], p1[1], 0.f);
	glm::vec3 v2(p2[0], p2[1], 0.f);
	glm::vec3 v3(p3[0], p3[1], 0.f);

	float s = 0.5f * glm::length(glm::cross(v1 - v2, v3 - v2));
	float s1 = 0.5f * glm::length(glm::cross(v - v2, v3 - v2));
	float s2 = 0.5f * glm::length(glm::cross(v - v3, v1 - v3));
	float s3 = 0.5f * glm::length(glm::cross(v - v1, v2 - v1));

	return glm::abs(s1 + s2 + s3 - s) < 0.1f;
}


// p here should be glm::vec3 (ScreenSpace.x, ScreenSpace.y, 0)
// p1, p2, p3 here should be glm::vec3(ScreenSpace.x, ScreenSpace.y, EyeSpace.z)
__device__
float depthValuePersCorrectionLerp(glm::vec3 p,
	glm::vec3 p1, glm::vec3 p2, glm::vec3 p3)
{
	glm::vec3 v1(p1[0], p1[1], 0.f);
	glm::vec3 v2(p2[0], p2[1], 0.f);
	glm::vec3 v3(p3[0], p3[1], 0.f);


	float s = 0.5f * glm::length(glm::cross(v1 - v2, v3 - v2));
	float s1 = 0.5f * glm::length(glm::cross(p - v2, v3 - v2));
	float s2 = 0.5f * glm::length(glm::cross(p - v3, v1 - v3));
	float s3 = 0.5f * glm::length(glm::cross(p - v1, v2 - v1));

	return 1.0f / ((s1 / (p1[2] * s)) + (s2 / (p2[2] * s)) + (s3 / (p3[2] * s)));
}

// p, p1, p2, p3 here should be glm::vec3(ScreenSpace.x, ScreenSpace.y, EyeSpace.z)
__device__
glm::vec2 vec2AttributePersCorrectionLerp(glm::vec3 p,
	glm::vec3 p1, glm::vec3 p2, glm::vec3 p3,
	glm::vec2 attribute1, glm::vec2 attribute2, glm::vec2 attribute3)
{
	glm::vec3 v(p[0], p[1], 0.f);
	glm::vec3 v1(p1[0], p1[1], 0.f);
	glm::vec3 v2(p2[0], p2[1], 0.f);
	glm::vec3 v3(p3[0], p3[1], 0.f);


	float s = 0.5f * glm::length(glm::cross(v1 - v2, v3 - v2));
	float s1 = 0.5f * glm::length(glm::cross(v - v2, v3 - v2));
	float s2 = 0.5f * glm::length(glm::cross(v - v3, v1 - v3));
	float s3 = 0.5f * glm::length(glm::cross(v - v1, v2 - v1));

	return p[2] * ((attribute1 / p1[2]) * (s1 / s)
		+ (attribute2 / p2[2]) * (s2 / s)
		+ (attribute3 / p3[2]) * (s3 / s));
}

// p, p1, p2, p3 here should be glm::vec3(ScreenSpace.x, ScreenSpace.y, EyeSpace.z)
__device__
glm::vec3 vec3AttributePersCorrectionLerp(glm::vec3 p,
	glm::vec3 p1, glm::vec3 p2, glm::vec3 p3,
	glm::vec3 attribute1, glm::vec3 attribute2, glm::vec3 attribute3)
{
	glm::vec3 v(p[0], p[1], 0.f);
	glm::vec3 v1(p1[0], p1[1], 0.f);
	glm::vec3 v2(p2[0], p2[1], 0.f);
	glm::vec3 v3(p3[0], p3[1], 0.f);


	float s = 0.5f * glm::length(glm::cross(v1 - v2, v3 - v2));
	float s1 = 0.5f * glm::length(glm::cross(v - v2, v3 - v2));
	float s2 = 0.5f * glm::length(glm::cross(v - v3, v1 - v3));
	float s3 = 0.5f * glm::length(glm::cross(v - v1, v2 - v1));

	return p[2] * ((attribute1 / p1[2]) * (s1 / s)
		+ (attribute2 / p2[2]) * (s2 / s)
		+ (attribute3 / p3[2]) * (s3 / s));
}


__device__
bool isPointOnTriangleEdge(glm::vec2 p, glm::vec3 t1, glm::vec3 t2, glm::vec3 t3) {

	glm::vec3 tris[3];
	tris[0] = t1;
	tris[1] = t2;
	tris[2] = t3;

	glm::vec3 barycentricCoord = calculateBarycentricCoordinate(tris, p);

	float Epsilon = 0.1f;

	if (glm::abs(barycentricCoord.x) < Epsilon) {
		if (barycentricCoord.y >= 0.0f && barycentricCoord.y <= 1.0f &&
			barycentricCoord.z >= 0.0f && barycentricCoord.z <= 1.0f) {
			return true;
		}
	}
	else if (glm::abs(barycentricCoord.y) < Epsilon) {
		if (barycentricCoord.x >= 0.0f && barycentricCoord.x <= 1.0f &&
			barycentricCoord.z >= 0.0f && barycentricCoord.z <= 1.0f) {
			return true;
		}
	}
	else if (glm::abs(barycentricCoord.z) < Epsilon) {
		if (barycentricCoord.y >= 0.0f && barycentricCoord.y <= 1.0f &&
			barycentricCoord.x >= 0.0f && barycentricCoord.x <= 1.0f) {
			return true;
		}
	}

	return false;
}