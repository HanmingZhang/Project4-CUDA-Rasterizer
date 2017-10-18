/**
 * @file      rasterize.cu
 * @brief     CUDA-accelerated rasterization pipeline.
 * @authors   Skeleton code: Yining Karl Li, Kai Ninomiya, Shuai Shao (Shrek)
 * @date      2012-2016
 * @copyright University of Pennsylvania & STUDENT
 */

#include <cmath>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/random.h>
#include <util/checkCUDAError.h>
#include <util/tiny_gltf_loader.h>
#include "rasterizeTools.h"
#include "rasterize.h"
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/remove.h>

// TODO : do performance test here

// --------------------------------------------
// Shared memory use in Gaussian Blur
#define GAUSSIANBLUR_SHAREDMEMORY

// --------------------------------------------
// SSAA
//#define SSAAx2

// --------------------------------------------
// MSAA remains some problems in our project...
// Fragment buffer size is fixed(width * height)
// Ideally, dynamic size fragment buffer is wanted

// In my case, I directly use SSAA fragment size in MSAA (2 * width * 2 * height)
// and this stupid way cause its performance worse than SSAA....
// Besides, some unwanted artifacts appear here

//#define MSAAx2

// --------------------------------------------
// Backface culling
// Pipeline way (remove unwanted primitive using thrust::remove_if)
//#define BACKFACE_CULLING_IN_PIPELINE
// naive way (directly do test in rasterizer)
//#define BACKFACE_CULLING_IN_RASTERIZER

// --------------------------------------------
// Correct color interpolation between points on a primitive
//#define CORRECT_COLOR_LERP


// --------------------------------------------
//#define BILINEAR_TEXTURE_FILTER

namespace {

	typedef unsigned short VertexIndex;
	typedef glm::vec3 VertexAttributePosition;
	typedef glm::vec3 VertexAttributeNormal;
	typedef glm::vec2 VertexAttributeTexcoord;
	typedef unsigned char TextureData;

	typedef unsigned char BufferByte;

	enum PrimitiveType{
		Point = 1,
		Line = 2,
		Triangle = 3
	};

	struct VertexOut {
		glm::vec4 pos;

		// TODO: add new attributes to your VertexOut
		// The attributes listed below might be useful, 
		// but always feel free to modify on your own

		 glm::vec3 eyePos;	// eye space position used for shading
		 glm::vec3 eyeNor;	// eye space normal used for shading, cuz normal will go wrong after perspective transformation

#ifdef CORRECT_COLOR_LERP
		 glm::vec3 col;
#endif 

		 glm::vec2 texcoord0;
		 TextureData* dev_diffuseTex = NULL;
		int diffuseTexWidth, diffuseTexHeight;
		// ...
	};

	struct Primitive {
		PrimitiveType primitiveType = Triangle;	// C++ 11 init
		VertexOut v[3];
	};

	struct Fragment {
		glm::vec3 color; // color == (texcoord0 + dev_diffuseTex)

		// TODO: add new attributes to your Fragment
		// The attributes listed below might be useful, 
		// but always feel free to modify on your own

		glm::vec3 eyePos;	// eye space position used for shading
		glm::vec3 eyeNor;

		// We have directily read color from texture and store color in glm::vec3 color
		// so we don't want uv and texture anymore

		//glm::vec2 texcoord0;
		//TextureData* dev_diffuseTex;
		// ...
	};

	struct PrimitiveDevBufPointers {
		int primitiveMode;	//from tinygltfloader macro
		PrimitiveType primitiveType;
		int numPrimitives;
		int numIndices;
		int numVertices;

		// Vertex In, const after loaded
		VertexIndex* dev_indices;
		VertexAttributePosition* dev_position;
		VertexAttributeNormal* dev_normal;
		VertexAttributeTexcoord* dev_texcoord0;

		// Materials, add more attributes when needed
		TextureData* dev_diffuseTex;
		int diffuseTexWidth;
		int diffuseTexHeight;
		// TextureData* dev_specularTex;
		// TextureData* dev_normalTex;
		// ...

		// Vertex Out, vertex used for rasterization, this is changing every frame
		VertexOut* dev_verticesOut;

		// TODO: add more attributes when needed
	};

}

static std::map<std::string, std::vector<PrimitiveDevBufPointers>> mesh2PrimitivesMap;


static int width = 0;
static int height = 0;

static int totalNumPrimitives = 0;
static Primitive *dev_primitives = NULL;


#ifdef BACKFACE_CULLING_IN_PIPELINE
static Primitive *dev_primitives_after_backfaceCulling = NULL;
#endif 

static Fragment *dev_fragmentBuffer = NULL;

static glm::vec3 *dev_framebuffer = NULL;


//Used in post-processing
static glm::vec3 *dev_framebuffer1 = NULL;
static glm::vec3 *dev_framebuffer_DownScaleBy10 = NULL;
static glm::vec3 *dev_framebuffer_DownScaleBy10_2 = NULL;


static int * dev_depth = NULL;	// you might need this buffer when doing depth test



/**
 * Kernel that writes the image to the OpenGL PBO directly.
 */
__global__ 
void sendImageToPBO(uchar4 *pbo, int w, int h, glm::vec3 *image, int framebufferEdgeOffset, int downScale_w, int downScaleRate) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * w);

    if (x < w && y < h) {

		int framebufferIndex;

		if (downScaleRate == 1)
		{
			framebufferIndex = x + (y * w);
		}
		else {
			// for downscale frame buffer debug
			framebufferIndex = (x / downScaleRate) + framebufferEdgeOffset + (((y / downScaleRate) + framebufferEdgeOffset) * (downScale_w + 2 * framebufferEdgeOffset));
		}

        glm::vec3 color;
        color.x = glm::clamp(image[framebufferIndex].x, 0.0f, 1.0f) * 255.0;
        color.y = glm::clamp(image[framebufferIndex].y, 0.0f, 1.0f) * 255.0;
        color.z = glm::clamp(image[framebufferIndex].z, 0.0f, 1.0f) * 255.0;
        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

#if defined(SSAAx2) || defined(MSAAx2)

// w, h should be Nx downscale image size
// image should be supersample buffer data
__global__
void sendImageToPBO_AAxN(uchar4 *pbo, int w, int h, glm::vec3 *image, int SSAA_Rate) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < w && y < h) {

		int index = x + (y * w);

		float totalSubSampleNumber = (float)SSAA_Rate * (float)SSAA_Rate;

		glm::vec3 color = glm::vec3(0.0f);
		for (int i = 0; i < SSAA_Rate; i++) {
			for (int j = 0; j < SSAA_Rate; j++) {
				int subSamplePixelX = SSAA_Rate * x + i;
				int subSamplePixelY = SSAA_Rate * y + j;

				int subSamplePixelIndex = subSamplePixelX + (subSamplePixelY * w * SSAA_Rate);
				color.x += glm::clamp(image[subSamplePixelIndex].x, 0.0f, 1.0f) * 255.0;
				color.y += glm::clamp(image[subSamplePixelIndex].y, 0.0f, 1.0f) * 255.0;
				color.z += glm::clamp(image[subSamplePixelIndex].z, 0.0f, 1.0f) * 255.0;
			}
		}
		
		// Each thread writes one pixel location in the texture (textel)
		pbo[index].w = 0;
		pbo[index].x = color.x / totalSubSampleNumber;
		pbo[index].y = color.y / totalSubSampleNumber;
		pbo[index].z = color.z / totalSubSampleNumber;
	}
}
#endif


/** 
* Writes fragment colors to the framebuffer
*/
__global__
void render(int w, int h, glm::vec3 lightPos, Fragment *fragmentBuffer, glm::vec3 *framebuffer, int renderMode, int framebufferEdgeOffset) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < w && y < h) {
		int index = x + (y * w);

		// TODO: add your fragment shader code here
		Fragment thisFragment = fragmentBuffer[index];

		// whole Triangle render mode
		if (renderMode == 1) {
			// Lambert shading

			glm::vec3 lightVec = lightPos - thisFragment.eyePos;
			lightVec = glm::normalize(lightVec);

			float light_cosTheta = glm::min(glm::max(glm::dot(thisFragment.eyeNor, lightVec), 0.0f), 1.0f);

			float ambientTerm = 0.6f;

			float light_power = 3.0f;

			float light_intensity = light_power * light_cosTheta + ambientTerm; // add ambient term so that we can still see points that are not lit by point light 

			framebuffer[index] = light_intensity * thisFragment.color;

		}

		// wireframe or point mode
		if (renderMode == 2 || renderMode == 3) {
			framebuffer[index] = thisFragment.color;
		}
    }
}

// Post-processing stage
__global__
void horizontalGaussianBlur(int w, int h, glm::vec3 *framebuffer_in, glm::vec3 *framebuffer_out, int framebufferEdgeOffset) {
	
#ifdef GAUSSIANBLUR_SHAREDMEMORY
	//array size should be blocksize.y * (framebufferEdgeOffset + blocksize.x + framebufferEdgeOffset)
	//In our case -> 8 * (5 + 8 + 5) -> 144
	__shared__ glm::vec3 framebuffer_in_shared[144];

#endif
	
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < w && y < h) {
		
		int framebufferIndex = (x + framebufferEdgeOffset) + ((y + framebufferEdgeOffset) * (w + 2 * framebufferEdgeOffset));
		framebuffer_out[framebufferIndex] = glm::vec3(0.f);

#ifdef GAUSSIANBLUR_SHAREDMEMORY
		// framebufferEdgeOffset + blocksize.x + framebufferEdgeOffset
		// 18 -> 5 + 8 + 5
		int index = threadIdx.y * 18 + threadIdx.x + 5;

		if (threadIdx.x == 0) {
			framebuffer_in_shared[index - 5] = framebuffer_in[framebufferIndex - 5];
			framebuffer_in_shared[index - 4] = framebuffer_in[framebufferIndex - 4];
			framebuffer_in_shared[index - 3] = framebuffer_in[framebufferIndex - 3];
			framebuffer_in_shared[index - 2] = framebuffer_in[framebufferIndex - 2];
			framebuffer_in_shared[index - 1] = framebuffer_in[framebufferIndex - 1];
		}

		if (threadIdx.x == blockDim.x - 1) {
			framebuffer_in_shared[index + 1] = framebuffer_in[framebufferIndex + 1];
			framebuffer_in_shared[index + 2] = framebuffer_in[framebufferIndex + 2];
			framebuffer_in_shared[index + 3] = framebuffer_in[framebufferIndex + 3];
			framebuffer_in_shared[index + 4] = framebuffer_in[framebufferIndex + 4];
			framebuffer_in_shared[index + 5] = framebuffer_in[framebufferIndex + 5];
		}

		framebuffer_in_shared[index] = framebuffer_in[framebufferIndex];

		__syncthreads();

		framebuffer_out[framebufferIndex] += framebuffer_in_shared[index - 5] * 0.0093f;
		framebuffer_out[framebufferIndex] += framebuffer_in_shared[index - 4] * 0.028002f;
		framebuffer_out[framebufferIndex] += framebuffer_in_shared[index - 3] * 0.065984f;
		framebuffer_out[framebufferIndex] += framebuffer_in_shared[index - 2] * 0.121703f;
		framebuffer_out[framebufferIndex] += framebuffer_in_shared[index - 1] * 0.175713f;
		framebuffer_out[framebufferIndex] += framebuffer_in_shared[index] * 0.198596f;
		framebuffer_out[framebufferIndex] += framebuffer_in_shared[index + 1] * 0.175713f;
		framebuffer_out[framebufferIndex] += framebuffer_in_shared[index + 2] * 0.121703f;
		framebuffer_out[framebufferIndex] += framebuffer_in_shared[index + 3] * 0.065984f;
		framebuffer_out[framebufferIndex] += framebuffer_in_shared[index + 4] * 0.028002f;
		framebuffer_out[framebufferIndex] += framebuffer_in_shared[index + 5] * 0.0093f;


#else
		
		framebuffer_out[framebufferIndex] += framebuffer_in[framebufferIndex - 5] * 0.0093f;
		framebuffer_out[framebufferIndex] += framebuffer_in[framebufferIndex - 4] * 0.028002f;
		framebuffer_out[framebufferIndex] += framebuffer_in[framebufferIndex - 3] * 0.065984f;
		framebuffer_out[framebufferIndex] += framebuffer_in[framebufferIndex - 2] * 0.121703f;
		framebuffer_out[framebufferIndex] += framebuffer_in[framebufferIndex - 1] * 0.175713f;
		framebuffer_out[framebufferIndex] += framebuffer_in[framebufferIndex] * 0.198596f;
		framebuffer_out[framebufferIndex] += framebuffer_in[framebufferIndex + 1] * 0.175713f;
		framebuffer_out[framebufferIndex] += framebuffer_in[framebufferIndex + 2] * 0.121703f;
		framebuffer_out[framebufferIndex] += framebuffer_in[framebufferIndex + 3] * 0.065984f;
		framebuffer_out[framebufferIndex] += framebuffer_in[framebufferIndex + 4] * 0.028002f;
		framebuffer_out[framebufferIndex] += framebuffer_in[framebufferIndex + 5] * 0.0093f;
#endif
	}
}

__global__
void verticalGaussianBlur(int w, int h, glm::vec3 *framebuffer_in, glm::vec3 *framebuffer_out, int framebufferEdgeOffset) {

#ifdef GAUSSIANBLUR_SHAREDMEMORY
	//array size should be blocksize.x * (framebufferEdgeOffset + blocksize.y + framebufferEdgeOffset)
	//In our case -> 8 * (5 + 8 + 5) -> 144
	__shared__ glm::vec3 framebuffer_in_shared[144];

#endif

	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < w && y < h) {

		int framebufferIndex = (x + framebufferEdgeOffset) + ((y + framebufferEdgeOffset) * (w + 2 * framebufferEdgeOffset));
		framebuffer_out[framebufferIndex] = glm::vec3(0.f);

		int numOfelementsOneRow = w + 2 * framebufferEdgeOffset;

#ifdef GAUSSIANBLUR_SHAREDMEMORY
		// blocksize.x
		// 8
		int index = (threadIdx.y + 5) * 8 + threadIdx.x;


		if (threadIdx.y == 0) {
			// 40, 32, 24... -> 5 * blocksize.x          
			framebuffer_in_shared[index - 40] = framebuffer_in[framebufferIndex - 5 * numOfelementsOneRow];
			framebuffer_in_shared[index - 32] = framebuffer_in[framebufferIndex - 4 * numOfelementsOneRow];
			framebuffer_in_shared[index - 24] = framebuffer_in[framebufferIndex - 3 * numOfelementsOneRow];
			framebuffer_in_shared[index - 16] = framebuffer_in[framebufferIndex - 2 * numOfelementsOneRow];
			framebuffer_in_shared[index - 8] = framebuffer_in[framebufferIndex - 1 * numOfelementsOneRow];
		}

		if (threadIdx.y == blockDim.y - 1) {
			framebuffer_in_shared[index + 8] = framebuffer_in[framebufferIndex + 1 * numOfelementsOneRow];
			framebuffer_in_shared[index + 16] = framebuffer_in[framebufferIndex + 2 * numOfelementsOneRow];
			framebuffer_in_shared[index + 24] = framebuffer_in[framebufferIndex + 3 * numOfelementsOneRow];
			framebuffer_in_shared[index + 32] = framebuffer_in[framebufferIndex + 4 * numOfelementsOneRow];
			framebuffer_in_shared[index + 40] = framebuffer_in[framebufferIndex + 5 * numOfelementsOneRow];
		}

		framebuffer_in_shared[index] = framebuffer_in[framebufferIndex];

		__syncthreads();

		framebuffer_out[framebufferIndex] += framebuffer_in_shared[index - 40] * 0.0093f;
		framebuffer_out[framebufferIndex] += framebuffer_in_shared[index - 32] * 0.028002f;
		framebuffer_out[framebufferIndex] += framebuffer_in_shared[index - 24] * 0.065984f;
		framebuffer_out[framebufferIndex] += framebuffer_in_shared[index - 16] * 0.121703f;
		framebuffer_out[framebufferIndex] += framebuffer_in_shared[index - 8] * 0.175713f;
		framebuffer_out[framebufferIndex] += framebuffer_in_shared[index] * 0.198596f;
		framebuffer_out[framebufferIndex] += framebuffer_in_shared[index + 8] * 0.175713f;
		framebuffer_out[framebufferIndex] += framebuffer_in_shared[index + 16] * 0.121703f;
		framebuffer_out[framebufferIndex] += framebuffer_in_shared[index + 24] * 0.065984f;
		framebuffer_out[framebufferIndex] += framebuffer_in_shared[index + 32] * 0.028002f;
		framebuffer_out[framebufferIndex] += framebuffer_in_shared[index + 40] * 0.0093f;


#else

		framebuffer_out[framebufferIndex] += framebuffer_in[framebufferIndex - 5 * numOfelementsOneRow] * 0.0093f;
		framebuffer_out[framebufferIndex] += framebuffer_in[framebufferIndex - 4 * numOfelementsOneRow] * 0.028002f;
		framebuffer_out[framebufferIndex] += framebuffer_in[framebufferIndex - 3 * numOfelementsOneRow] * 0.065984f;
		framebuffer_out[framebufferIndex] += framebuffer_in[framebufferIndex - 2 * numOfelementsOneRow] * 0.121703f;
		framebuffer_out[framebufferIndex] += framebuffer_in[framebufferIndex - 1 * numOfelementsOneRow] * 0.175713f;
		framebuffer_out[framebufferIndex] += framebuffer_in[framebufferIndex] * 0.198596f;
		framebuffer_out[framebufferIndex] += framebuffer_in[framebufferIndex + 1 * numOfelementsOneRow] * 0.175713f;
		framebuffer_out[framebufferIndex] += framebuffer_in[framebufferIndex + 2 * numOfelementsOneRow] * 0.121703f;
		framebuffer_out[framebufferIndex] += framebuffer_in[framebufferIndex + 3 * numOfelementsOneRow] * 0.065984f;
		framebuffer_out[framebufferIndex] += framebuffer_in[framebufferIndex + 4 * numOfelementsOneRow] * 0.028002f;
		framebuffer_out[framebufferIndex] += framebuffer_in[framebufferIndex + 5 * numOfelementsOneRow] * 0.0093f;
#endif
	}
}

// downScaleRate should compatible with downScale_w & downScale_h
__global__
void sampleDownScaleSample(int downScale_w, int downScale_h, int downScaleRate, 
						   int w, int h, 
						   glm::vec3 *downScale_framebuffer, glm::vec3 *framebuffer, int framebufferEdgeOffset)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < downScale_w && y < downScale_h) {
		int index = (x + framebufferEdgeOffset) + ((y + framebufferEdgeOffset) * (downScale_w + 2 * framebufferEdgeOffset));

		glm::vec3& thisFrameBufferCol = downScale_framebuffer[index];
		thisFrameBufferCol = glm::vec3(0.0f);

		float totalSampleNumber = (float)downScaleRate * (float)downScaleRate;

		int ori_framebuffer_x, ori_framebuffer_y;
		int ori_framebuffer_index;

		for (int i = 0; i < downScaleRate; i++) {
			for (int j = 0; j < downScaleRate; j++) {
				ori_framebuffer_x = x * downScaleRate + i;
				ori_framebuffer_y = y * downScaleRate + j;

				ori_framebuffer_x = glm::clamp(ori_framebuffer_x, 0, w - 1);
				ori_framebuffer_y = glm::clamp(ori_framebuffer_y, 0, h - 1);

				ori_framebuffer_index = (ori_framebuffer_x) + (ori_framebuffer_y * w);

				thisFrameBufferCol += framebuffer[ori_framebuffer_index];
			}
		}
		//take the average value of samples
		thisFrameBufferCol *= (1.0f / totalSampleNumber);
	}
}


__global__
void brightFilter(int w, int h, glm::vec3 *framebuffer_in, glm::vec3 *framebuffer_out) {

	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < w && y < h) {
		int index = (x) + ((y) * (w));
		glm::vec3 thisFrameBuffer_in = framebuffer_in[index];

		float brightness = thisFrameBuffer_in.r * 0.2126f + thisFrameBuffer_in.g * 0.7152f + thisFrameBuffer_in.b * 0.0722f;
		framebuffer_out[index] = brightness * thisFrameBuffer_in;
	}
}

__global__
void combineFrameBuffer(int w, int h, glm::vec3 *mainScene_framebuffer, glm::vec3 *other_framebuffer, glm::vec3 *framebuffer_out, 
						int other_framebuffer_downScale_w, int other_framebuffer_downScaleRate,
					    int framebufferEdgeOffset) {

	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < w && y < h) {
		int mainSceneIdx = x + (y * w);
		glm::vec3 thisMainSceneFrameBufferCol = mainScene_framebuffer[mainSceneIdx];

		int other_framebufferIdx = (x / other_framebuffer_downScaleRate) + framebufferEdgeOffset + 
							     (((y / other_framebuffer_downScaleRate) + framebufferEdgeOffset) * 
								   (other_framebuffer_downScale_w + 2 * framebufferEdgeOffset));


		glm::vec3 otherFrameBufferColor = other_framebuffer[other_framebufferIdx];

		framebuffer_out[mainSceneIdx] = thisMainSceneFrameBufferCol + 1.0f * otherFrameBufferColor;
	}
}


/**
 * Called once at the beginning of the program to allocate memory.
 */

int GaussianBlurEdgeRoom = 5;

void rasterizeInit(int w, int h) {

#if defined(SSAAx2) || defined(MSAAx2)
	width = 2 * w;
	height = 2 * h;
#else
	width = w;
	height = h;
#endif

    
	cudaFree(dev_fragmentBuffer);
	cudaMalloc(&dev_fragmentBuffer, width * height * sizeof(Fragment));
	cudaMemset(dev_fragmentBuffer, 0, width * height * sizeof(Fragment));

    cudaFree(dev_framebuffer);
    cudaMalloc(&dev_framebuffer,   (width) * (height) * sizeof(glm::vec3));
    cudaMemset(dev_framebuffer, 0, (width) * (height) * sizeof(glm::vec3));

	cudaFree(dev_framebuffer1);
	cudaMalloc(&dev_framebuffer1, (width) * (height) * sizeof(glm::vec3));
	cudaMemset(dev_framebuffer1, 0, (width) * (height) * sizeof(glm::vec3));

	int downScaleRate = 10;
	cudaFree(dev_framebuffer_DownScaleBy10);
	cudaMalloc(&dev_framebuffer_DownScaleBy10, ((width / downScaleRate) + 2 * GaussianBlurEdgeRoom) * ((height / downScaleRate) + 2 * GaussianBlurEdgeRoom) * sizeof(glm::vec3));
	cudaMemset(dev_framebuffer_DownScaleBy10, 0, ((width / downScaleRate) + 2 * GaussianBlurEdgeRoom) * ((height / downScaleRate) + 2 * GaussianBlurEdgeRoom) * sizeof(glm::vec3));

	cudaFree(dev_framebuffer_DownScaleBy10_2);
	cudaMalloc(&dev_framebuffer_DownScaleBy10_2, ((width / downScaleRate) + 2 * GaussianBlurEdgeRoom) * ((height / downScaleRate) + 2 * GaussianBlurEdgeRoom) * sizeof(glm::vec3));
	cudaMemset(dev_framebuffer_DownScaleBy10_2, 0, ((width / downScaleRate) + 2 * GaussianBlurEdgeRoom) * ((height / downScaleRate) + 2 * GaussianBlurEdgeRoom) * sizeof(glm::vec3));

	cudaFree(dev_depth);
	cudaMalloc(&dev_depth, width * height * sizeof(int));

	checkCUDAError("rasterizeInit");
}

__global__
void initDepth(int w, int h, int * depth)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < w && y < h)
	{
		int index = x + (y * w);
		depth[index] = INT_MAX;
	}
}


/**
* kern function with support for stride to sometimes replace cudaMemcpy
* One thread is responsible for copying one component
*/
__global__ 
void _deviceBufferCopy(int N, BufferByte* dev_dst, const BufferByte* dev_src, int n, int byteStride, int byteOffset, int componentTypeByteSize) {
	
	// Attribute (vec3 position)
	// component (3 * float)
	// byte (4 * byte)

	// id of component
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (i < N) {
		int count = i / n;
		int offset = i - count * n;	// which component of the attribute

		for (int j = 0; j < componentTypeByteSize; j++) {
			
			dev_dst[count * componentTypeByteSize * n 
				+ offset * componentTypeByteSize 
				+ j]

				= 

			dev_src[byteOffset 
				+ count * (byteStride == 0 ? componentTypeByteSize * n : byteStride) 
				+ offset * componentTypeByteSize 
				+ j];
		}
	}
	

}

__global__
void _nodeMatrixTransform(
	int numVertices,
	VertexAttributePosition* position,
	VertexAttributeNormal* normal,
	glm::mat4 MV, glm::mat3 MV_normal) {

	// vertex id
	int vid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (vid < numVertices) {
		position[vid] = glm::vec3(MV * glm::vec4(position[vid], 1.0f));
		normal[vid] = glm::normalize(MV_normal * normal[vid]);
	}
}

glm::mat4 getMatrixFromNodeMatrixVector(const tinygltf::Node & n) {
	
	glm::mat4 curMatrix(1.0);

	const std::vector<double> &m = n.matrix;
	if (m.size() > 0) {
		// matrix, copy it

		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				curMatrix[i][j] = (float)m.at(4 * i + j);
			}
		}
	} else {
		// no matrix, use rotation, scale, translation

		if (n.translation.size() > 0) {
			curMatrix[3][0] = n.translation[0];
			curMatrix[3][1] = n.translation[1];
			curMatrix[3][2] = n.translation[2];
		}

		if (n.rotation.size() > 0) {
			glm::mat4 R;
			glm::quat q;
			q[0] = n.rotation[0];
			q[1] = n.rotation[1];
			q[2] = n.rotation[2];

			R = glm::mat4_cast(q);
			curMatrix = curMatrix * R;
		}

		if (n.scale.size() > 0) {
			curMatrix = curMatrix * glm::scale(glm::vec3(n.scale[0], n.scale[1], n.scale[2]));
		}
	}

	return curMatrix;
}

void traverseNode (
	std::map<std::string, glm::mat4> & n2m,
	const tinygltf::Scene & scene,
	const std::string & nodeString,
	const glm::mat4 & parentMatrix
	) 
{
	const tinygltf::Node & n = scene.nodes.at(nodeString);
	glm::mat4 M = parentMatrix * getMatrixFromNodeMatrixVector(n);
	n2m.insert(std::pair<std::string, glm::mat4>(nodeString, M));

	auto it = n.children.begin();
	auto itEnd = n.children.end();

	for (; it != itEnd; ++it) {
		traverseNode(n2m, scene, *it, M);
	}
}

void rasterizeSetBuffers(const tinygltf::Scene & scene) {

	totalNumPrimitives = 0;

	std::map<std::string, BufferByte*> bufferViewDevPointers;

	// 1. copy all `bufferViews` to device memory
	{
		std::map<std::string, tinygltf::BufferView>::const_iterator it(
			scene.bufferViews.begin());
		std::map<std::string, tinygltf::BufferView>::const_iterator itEnd(
			scene.bufferViews.end());

		for (; it != itEnd; it++) {
			const std::string key = it->first;
			const tinygltf::BufferView &bufferView = it->second;
			if (bufferView.target == 0) {
				continue; // Unsupported bufferView.
			}

			const tinygltf::Buffer &buffer = scene.buffers.at(bufferView.buffer);

			BufferByte* dev_bufferView;
			cudaMalloc(&dev_bufferView, bufferView.byteLength);
			cudaMemcpy(dev_bufferView, &buffer.data.front() + bufferView.byteOffset, bufferView.byteLength, cudaMemcpyHostToDevice);

			checkCUDAError("Set BufferView Device Mem");

			bufferViewDevPointers.insert(std::make_pair(key, dev_bufferView));

		}
	}



	// 2. for each mesh: 
	//		for each primitive: 
	//			build device buffer of indices, materail, and each attributes
	//			and store these pointers in a map
	{

		std::map<std::string, glm::mat4> nodeString2Matrix;
		auto rootNodeNamesList = scene.scenes.at(scene.defaultScene);

		{
			auto it = rootNodeNamesList.begin();
			auto itEnd = rootNodeNamesList.end();
			for (; it != itEnd; ++it) {
				traverseNode(nodeString2Matrix, scene, *it, glm::mat4(1.0f));
			}
		}


		// parse through node to access mesh

		auto itNode = nodeString2Matrix.begin();
		auto itEndNode = nodeString2Matrix.end();
		for (; itNode != itEndNode; ++itNode) {

			const tinygltf::Node & N = scene.nodes.at(itNode->first);
			const glm::mat4 & matrix = itNode->second;
			const glm::mat3 & matrixNormal = glm::transpose(glm::inverse(glm::mat3(matrix)));

			auto itMeshName = N.meshes.begin();
			auto itEndMeshName = N.meshes.end();

			for (; itMeshName != itEndMeshName; ++itMeshName) {

				const tinygltf::Mesh & mesh = scene.meshes.at(*itMeshName);

				auto res = mesh2PrimitivesMap.insert(std::pair<std::string, std::vector<PrimitiveDevBufPointers>>(mesh.name, std::vector<PrimitiveDevBufPointers>()));
				std::vector<PrimitiveDevBufPointers> & primitiveVector = (res.first)->second;

				// for each primitive
				for (size_t i = 0; i < mesh.primitives.size(); i++) {
					const tinygltf::Primitive &primitive = mesh.primitives[i];

					if (primitive.indices.empty())
						return;

					// TODO: add new attributes for your PrimitiveDevBufPointers when you add new attributes
					VertexIndex* dev_indices = NULL;
					VertexAttributePosition* dev_position = NULL;
					VertexAttributeNormal* dev_normal = NULL;
					VertexAttributeTexcoord* dev_texcoord0 = NULL;

					// ----------Indices-------------

					const tinygltf::Accessor &indexAccessor = scene.accessors.at(primitive.indices);
					const tinygltf::BufferView &bufferView = scene.bufferViews.at(indexAccessor.bufferView);
					BufferByte* dev_bufferView = bufferViewDevPointers.at(indexAccessor.bufferView);

					// assume type is SCALAR for indices
					int n = 1;
					int numIndices = indexAccessor.count;
					int componentTypeByteSize = sizeof(VertexIndex);
					int byteLength = numIndices * n * componentTypeByteSize;

					dim3 numThreadsPerBlock(128);
					dim3 numBlocks((numIndices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
					cudaMalloc(&dev_indices, byteLength);
					_deviceBufferCopy << <numBlocks, numThreadsPerBlock >> > (
						numIndices,
						(BufferByte*)dev_indices,
						dev_bufferView,
						n,
						indexAccessor.byteStride,
						indexAccessor.byteOffset,
						componentTypeByteSize);


					checkCUDAError("Set Index Buffer");


					// ---------Primitive Info-------

					// Warning: LINE_STRIP is not supported in tinygltfloader
					int numPrimitives;
					PrimitiveType primitiveType;
					switch (primitive.mode) {
					case TINYGLTF_MODE_TRIANGLES:
						primitiveType = PrimitiveType::Triangle;
						numPrimitives = numIndices / 3;
						break;
					case TINYGLTF_MODE_TRIANGLE_STRIP:
						primitiveType = PrimitiveType::Triangle;
						numPrimitives = numIndices - 2;
						break;
					case TINYGLTF_MODE_TRIANGLE_FAN:
						primitiveType = PrimitiveType::Triangle;
						numPrimitives = numIndices - 2;
						break;
					case TINYGLTF_MODE_LINE:
						primitiveType = PrimitiveType::Line;
						numPrimitives = numIndices / 2;
						break;
					case TINYGLTF_MODE_LINE_LOOP:
						primitiveType = PrimitiveType::Line;
						numPrimitives = numIndices + 1;
						break;
					case TINYGLTF_MODE_POINTS:
						primitiveType = PrimitiveType::Point;
						numPrimitives = numIndices;
						break;
					default:
						// output error
						break;
					};


					// ----------Attributes-------------

					auto it(primitive.attributes.begin());
					auto itEnd(primitive.attributes.end());

					int numVertices = 0;
					// for each attribute
					for (; it != itEnd; it++) {
						const tinygltf::Accessor &accessor = scene.accessors.at(it->second);
						const tinygltf::BufferView &bufferView = scene.bufferViews.at(accessor.bufferView);

						int n = 1;
						if (accessor.type == TINYGLTF_TYPE_SCALAR) {
							n = 1;
						}
						else if (accessor.type == TINYGLTF_TYPE_VEC2) {
							n = 2;
						}
						else if (accessor.type == TINYGLTF_TYPE_VEC3) {
							n = 3;
						}
						else if (accessor.type == TINYGLTF_TYPE_VEC4) {
							n = 4;
						}

						BufferByte * dev_bufferView = bufferViewDevPointers.at(accessor.bufferView);
						BufferByte ** dev_attribute = NULL;

						numVertices = accessor.count;
						int componentTypeByteSize;

						// Note: since the type of our attribute array (dev_position) is static (float32)
						// We assume the glTF model attribute type are 5126(FLOAT) here

						if (it->first.compare("POSITION") == 0) {
							componentTypeByteSize = sizeof(VertexAttributePosition) / n;
							dev_attribute = (BufferByte**)&dev_position;
						}
						else if (it->first.compare("NORMAL") == 0) {
							componentTypeByteSize = sizeof(VertexAttributeNormal) / n;
							dev_attribute = (BufferByte**)&dev_normal;
						}
						else if (it->first.compare("TEXCOORD_0") == 0) {
							componentTypeByteSize = sizeof(VertexAttributeTexcoord) / n;
							dev_attribute = (BufferByte**)&dev_texcoord0;
						}

						std::cout << accessor.bufferView << "  -  " << it->second << "  -  " << it->first << '\n';

						dim3 numThreadsPerBlock(128);
						dim3 numBlocks((n * numVertices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
						int byteLength = numVertices * n * componentTypeByteSize;
						cudaMalloc(dev_attribute, byteLength);

						_deviceBufferCopy << <numBlocks, numThreadsPerBlock >> > (
							n * numVertices,
							*dev_attribute,
							dev_bufferView,
							n,
							accessor.byteStride,
							accessor.byteOffset,
							componentTypeByteSize);

						std::string msg = "Set Attribute Buffer: " + it->first;
						checkCUDAError(msg.c_str());
					}

					// malloc for VertexOut
					VertexOut* dev_vertexOut;
					cudaMalloc(&dev_vertexOut, numVertices * sizeof(VertexOut));
					checkCUDAError("Malloc VertexOut Buffer");

					// ----------Materials-------------

					// You can only worry about this part once you started to 
					// implement textures for your rasterizer
					TextureData* dev_diffuseTex = NULL;
					int diffuseTexWidth = 0;
					int diffuseTexHeight = 0;
					if (!primitive.material.empty()) {
						const tinygltf::Material &mat = scene.materials.at(primitive.material);
						printf("material.name = %s\n", mat.name.c_str());

						if (mat.values.find("diffuse") != mat.values.end()) {
							std::string diffuseTexName = mat.values.at("diffuse").string_value;
							if (scene.textures.find(diffuseTexName) != scene.textures.end()) {
								const tinygltf::Texture &tex = scene.textures.at(diffuseTexName);
								if (scene.images.find(tex.source) != scene.images.end()) {
									const tinygltf::Image &image = scene.images.at(tex.source);

									size_t s = image.image.size() * sizeof(TextureData);
									cudaMalloc(&dev_diffuseTex, s);
									cudaMemcpy(dev_diffuseTex, &image.image.at(0), s, cudaMemcpyHostToDevice);
									
									diffuseTexWidth = image.width;
									diffuseTexHeight = image.height;

									checkCUDAError("Set Texture Image data");
								}
							}
						}

						// TODO: write your code for other materails
						// You may have to take a look at tinygltfloader
						// You can also use the above code loading diffuse material as a start point 
					}


					// ---------Node hierarchy transform--------
					cudaDeviceSynchronize();
					
					dim3 numBlocksNodeTransform((numVertices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
					_nodeMatrixTransform << <numBlocksNodeTransform, numThreadsPerBlock >> > (
						numVertices,
						dev_position,
						dev_normal,
						matrix,
						matrixNormal);

					checkCUDAError("Node hierarchy transformation");

					// at the end of the for loop of primitive
					// push dev pointers to map
					primitiveVector.push_back(PrimitiveDevBufPointers{
						primitive.mode,
						primitiveType,
						numPrimitives,
						numIndices,
						numVertices,

						dev_indices,
						dev_position,
						dev_normal,
						dev_texcoord0,

						dev_diffuseTex,
						diffuseTexWidth,
						diffuseTexHeight,

						dev_vertexOut	//VertexOut
					});

					totalNumPrimitives += numPrimitives;

				} // for each primitive

			} // for each mesh

		} // for each node

	}
	

	// 3. Malloc for dev_primitives
	{
		cudaMalloc(&dev_primitives, totalNumPrimitives * sizeof(Primitive));

#ifdef BACKFACE_CULLING_IN_PIPELINE
		cudaMalloc(&dev_primitives_after_backfaceCulling, totalNumPrimitives * sizeof(Primitive));
#endif
	}
	

	// Finally, cudaFree raw dev_bufferViews
	{

		std::map<std::string, BufferByte*>::const_iterator it(bufferViewDevPointers.begin());
		std::map<std::string, BufferByte*>::const_iterator itEnd(bufferViewDevPointers.end());
			
			//bufferViewDevPointers

		for (; it != itEnd; it++) {
			cudaFree(it->second);
		}

		checkCUDAError("Free BufferView Device Mem");
	}


}



__global__ 
void _vertexTransformAndAssembly(
	int numVertices, 
	PrimitiveDevBufPointers primitive, 
	glm::mat4 MVP, glm::mat4 MV, glm::mat3 MV_normal, 
	int width, int height,
	glm::mat4 selfRotateM) 
{

	// vertex id
	int vid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (vid < numVertices) {

		__shared__ glm::mat4 _selfRotateM;

		if (threadIdx.x == 0) {
			_selfRotateM = selfRotateM;
		}

		__syncthreads();

		// TODO: Apply vertex transformation here
		// Multiply the MVP matrix for each vertex position, this will transform everything into clipping space
		// Then divide the pos by its w element to transform into NDC space
		// Finally transform x and y to viewport space

		// TODO: Apply vertex assembly here
		// Assemble all attribute arraies into the primitive array

		VertexOut& this_dev_verticesOut = primitive.dev_verticesOut[vid];

		// Handle screen space postion
		glm::vec4 ClippingSpace_Pos = MVP * _selfRotateM * glm::vec4(primitive.dev_position[vid], 1.0f);
		glm::vec4 NDC_Pos = (1.0f / ClippingSpace_Pos.w) * ClippingSpace_Pos;
		glm::vec4 ScreenSpace_Pos = glm::vec4((NDC_Pos.x + 1.0f) * (float)width / 2.0f, 
											  (1.0f - NDC_Pos.y) * (float)height / 2.0f, 
											  NDC_Pos.z,
											  NDC_Pos.w);


		this_dev_verticesOut.pos = ScreenSpace_Pos;

		// Handle eye space postion
		this_dev_verticesOut.eyePos = glm::vec3(MV * _selfRotateM * glm::vec4(primitive.dev_position[vid], 1.0f));

		// Handle eye space normal
		this_dev_verticesOut.eyeNor = glm::normalize(MV_normal * glm::mat3(_selfRotateM) * primitive.dev_normal[vid]); // normalized

		// Handle uv
		if (primitive.dev_texcoord0 != NULL) {
			this_dev_verticesOut.texcoord0 = primitive.dev_texcoord0[vid];
		}
		else {
			this_dev_verticesOut.texcoord0 = glm::vec2(0.0f); //set a default value, in case of some uninitialized error
		}

		// Handle diffuse texture
		if (primitive.dev_diffuseTex != NULL) {
			//Assume all vertices use just one diffuse texture
			this_dev_verticesOut.dev_diffuseTex = primitive.dev_diffuseTex;
			this_dev_verticesOut.diffuseTexWidth = primitive.diffuseTexWidth;
			this_dev_verticesOut.diffuseTexHeight = primitive.diffuseTexHeight;
		}

#ifdef CORRECT_COLOR_LERP
		if (vid % 3 == 0) {
			this_dev_verticesOut.col = glm::vec3(0.95f, 0.25f, 0.25f);
		}
		else if (vid % 3 == 1) {
			this_dev_verticesOut.col = glm::vec3(0.25f, 0.95f, 0.25f);
		}
		else if (vid % 3 == 2) {
			this_dev_verticesOut.col = glm::vec3(0.25f, 0.25f, 0.95f);
		}
#endif 

	}
}

static int curPrimitiveBeginId = 0;

__global__ 
void _primitiveAssembly(int numIndices, int curPrimitiveBeginId, Primitive* dev_primitives, PrimitiveDevBufPointers primitive) {

	// index id
	int iid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (iid < numIndices) {

		// TODO: uncomment the following code for a start
		// This is primitive assembly for triangles

		int pid;	// id for cur primitives vector
		if (primitive.primitiveMode == TINYGLTF_MODE_TRIANGLES) {
			pid = iid / (int)primitive.primitiveType;
			dev_primitives[pid + curPrimitiveBeginId].v[iid % (int)primitive.primitiveType]
				= primitive.dev_verticesOut[primitive.dev_indices[iid]];
		}


		// TODO: other primitive types (point, line)
	}
	
}




__device__
void fillThisFragmentBuffer(Fragment& thisFragment,
						    glm::vec3 p, 
							glm::vec3 t1, glm::vec3 t2, glm::vec3 t3,
							VertexOut v1, VertexOut v2, VertexOut v3)
{
	//Assume all vertices in one primitive use one Diffuse texture
	int diffuseTexWidth = v1.diffuseTexWidth;
	int diffuseTexHeight = v1.diffuseTexHeight;
	TextureData* textureData = v1.dev_diffuseTex;

	// Handle Positions (assume gltf always has this property)
	glm::vec3 lerp_eyePos = vec3AttributePersCorrectionLerp(
		p,
		t1, t2, t3,
		v1.eyePos, v2.eyePos, v3.eyePos);

	thisFragment.eyePos = lerp_eyePos;


	// Handle Normals (assume gltf always has this property)
	glm::vec3 lerp_eyeNor = vec3AttributePersCorrectionLerp(
		p,
		t1, t2, t3,
		v1.eyeNor, v2.eyeNor, v3.eyeNor);

	lerp_eyeNor = glm::normalize(lerp_eyeNor); // normalized

	thisFragment.eyeNor = lerp_eyeNor;


	// Handle UV (assume gltf always has this property)
	glm::vec2 lerp_uv = vec2AttributePersCorrectionLerp(
		p,
		t1, t2, t3,
		v1.texcoord0, v2.texcoord0, v3.texcoord0);

	// Fetch color from texture
	if (textureData != NULL) {
		TextureData r, g, b;

#ifdef BILINEAR_TEXTURE_FILTER
		lerp_uv.x = lerp_uv.x * diffuseTexWidth  - 0.5f;
		lerp_uv.y = lerp_uv.y * diffuseTexHeight - 0.5f;

		float x = glm::floor(lerp_uv.x);
		float y = glm::floor(lerp_uv.y);
		float u_ratio = lerp_uv.x - x;
		float v_ratio = lerp_uv.y - y;
		float u_opposite = 1.0f - u_ratio;
		float v_opposite = 1.0f - v_ratio;


		int textIdx = x + diffuseTexWidth * y;

		int numOfTextureChannels = 3;

		r = (u_opposite * textureData[textIdx * numOfTextureChannels] + u_ratio * textureData[(textIdx + 1) * numOfTextureChannels]) * v_opposite
		  + (u_opposite * textureData[(textIdx + diffuseTexWidth) * numOfTextureChannels] + u_ratio * textureData[(textIdx + diffuseTexWidth + 1) * numOfTextureChannels]) * v_ratio;

		g = (u_opposite * textureData[textIdx * numOfTextureChannels + 1] + u_ratio * textureData[(textIdx + 1) * numOfTextureChannels + 1]) * v_opposite
			+ (u_opposite * textureData[(textIdx + diffuseTexWidth) * numOfTextureChannels + 1] + u_ratio * textureData[(textIdx + diffuseTexWidth + 1) * numOfTextureChannels + 1]) * v_ratio;
		
		b = (u_opposite * textureData[textIdx * numOfTextureChannels + 2] + u_ratio * textureData[(textIdx + 1) * numOfTextureChannels + 2]) * v_opposite
			+ (u_opposite * textureData[(textIdx + diffuseTexWidth) * numOfTextureChannels + 2] + u_ratio * textureData[(textIdx + diffuseTexWidth + 1) * numOfTextureChannels + 2]) * v_ratio;


#else
		glm::ivec2 textSpaceCoord = glm::ivec2(diffuseTexWidth * lerp_uv.x, diffuseTexHeight * (lerp_uv.y));

		int textIdx = textSpaceCoord.x + diffuseTexWidth * textSpaceCoord.y;

		// Assume texture data are row major
		// and there are 3 channels
		int numOfTextureChannels = 3;
		r = textureData[textIdx * numOfTextureChannels];
		g = textureData[textIdx * numOfTextureChannels + 1];
		b = textureData[textIdx * numOfTextureChannels + 2];

#endif // BILINEAR_TEXTURE_FILTER

		thisFragment.color = glm::vec3((float)r / 255.0f, (float)g / 255.0f, (float)b / 255.0f);
	}

	else {
		// Debug normal
		//thisFragment.color = glm::vec3(0.5f * (lerp_eyeNor.x + 1.0f),
		//						       0.5f * (lerp_eyeNor.y + 1.0f),
		//							   0.5f * (lerp_eyeNor.z + 1.0f));
#ifdef CORRECT_COLOR_LERP
		// Handle Colors
		glm::vec3 lerp_col = vec3AttributePersCorrectionLerp(
			p,
			t1, t2, t3,
			v1.col, v2.col, v3.col);

		thisFragment.color = lerp_col;
#else
		thisFragment.color = glm::vec3(0.95f, 0.95f, 0.95f);
#endif
	}
}


// Rasterizer - Fill method
// whole triangle mode
__device__
void rasterizer_fill_wholeTriangleMode(Fragment* fragmentBuffer, Primitive& thisPrimitive, int* depth,
									   glm::vec3 t1, glm::vec3 t2, glm::vec3 t3,
									   int w, int h) 
{	
	//Use AABB
	float minX = fminf(t1.x, fminf(t2.x, t3.x));
	float maxX = fmaxf(t1.x, fmaxf(t2.x, t3.x));
	float minY = fminf(t1.y, fminf(t2.y, t3.y));
	float maxY = fmaxf(t1.y, fmaxf(t2.y, t3.y));

	// make sure AABB is inside screen
	int startX = minX < 0 ? 0 : (int)glm::floor(minX);
	int endX   = maxX > w ? w : (int)glm::ceil(maxX);

	int startY = minY < 0 ? 0 : (int)glm::floor(minY);
	int endY   = maxY > h ? h : (int)glm::ceil(maxY);

#ifdef MSAAx2
	for (int i = startY; i <= endY; i += 2) {
		for (int j = startX; j <= endX; j += 2) {

			// if point is on(very close, depends on epsilon) the edge of triangle
			if (isPointOnTriangleEdge(glm::vec2(j + 0.5f, i + 0.5f), t1, t2, t3)) {
				// do Multi sample
				for (int p = 0; p < 2; p++) {
					for (int q = 0; q < 2; q++) {
						float lerp_depth = depthValuePersCorrectionLerp(glm::vec3(j + q, i + p, 0.f), t1, t2, t3);
						int lerp_depth_int = (int)(lerp_depth * 100000.0f);
						// Atomic depth buffer writing
						int fragmentIdx = (j + q) + ((i + p) * w);
						int old = depth[fragmentIdx];
						int assumed;

						do {
							assumed = old;
							old = atomicMin(&depth[fragmentIdx], lerp_depth_int);
						} while (assumed != old);

						//must use depth[index] to read again!
						if (lerp_depth_int <= depth[fragmentIdx]) {
							// pass depth test, this fragment is good, we will use it
							glm::vec3 p(j + q, i + p, lerp_depth);

							// fill this fragment Buffer
							fillThisFragmentBuffer(fragmentBuffer[fragmentIdx],
								p,
								t1, t2, t3,
								thisPrimitive.v[0], thisPrimitive.v[1], thisPrimitive.v[2]);
						}
					}
				}
			}

			// if point is not on the edge of triangle
			// but if it's inside the tirangle
			else if (isPosInTriange(glm::vec3(j + 0.5f, i + 0.5f, 0.f), t1, t2, t3)) {
				float lerp_depth = depthValuePersCorrectionLerp(glm::vec3(j + 0.5f, i + 0.5f, 0.f), t1, t2, t3);
				int lerp_depth_int = (int)(lerp_depth * 100000.0f);
				
				glm::vec3 p(j + 0.5f, i + 0.5f, lerp_depth);

				// ----------- fill sub-Sample 1 -----------------
				// Atomic depth buffer writing
				int fragmentIdx = j + (i * w);
				int old = depth[fragmentIdx];
				int assumed;
				do {
					assumed = old;
					old = atomicMin(&depth[fragmentIdx], lerp_depth_int);
				} while (assumed != old);

				if (lerp_depth_int <= depth[fragmentIdx]) {
					// fill this fragment Buffer
					fillThisFragmentBuffer(fragmentBuffer[fragmentIdx],
						p,
						t1, t2, t3,
						thisPrimitive.v[0], thisPrimitive.v[1], thisPrimitive.v[2]);
				}

				// ----------- fill sub-Sample 2 -----------------
				old = depth[fragmentIdx + 1];
				do {
					assumed = old;
					old = atomicMin(&depth[fragmentIdx + 1], lerp_depth_int);
				} while (assumed != old);
				if (lerp_depth_int <= depth[fragmentIdx + 1]) {
					// fill this fragment Buffer
					fragmentBuffer[fragmentIdx + 1] = fragmentBuffer[fragmentIdx];
				}

				// ----------- fill sub-Sample 3 -----------------
				old = depth[fragmentIdx + w];
				do {
					assumed = old;
					old = atomicMin(&depth[fragmentIdx + w], lerp_depth_int);
				} while (assumed != old);
				if (lerp_depth_int <= depth[fragmentIdx + w]) {
					// fill this fragment Buffer
					fragmentBuffer[fragmentIdx + w] = fragmentBuffer[fragmentIdx];
				}

				// ----------- fill sub-Sample 4 -----------------
				old = depth[fragmentIdx + w + 1];
				do {
					assumed = old;
					old = atomicMin(&depth[fragmentIdx + w + 1], lerp_depth_int);
				} while (assumed != old);
				if (lerp_depth_int <= depth[fragmentIdx + w + 1]) {
					// fill this fragment Buffer
					fragmentBuffer[fragmentIdx + w + 1] = fragmentBuffer[fragmentIdx];

				}

				
			}

		}
	}

#else

	for (int i = startY; i <= endY; i++) {
		for (int j = startX; j <= endX; j++) {
			// Test if this pos is in the triangle
			if (isPosInTriange(glm::vec3(j, i, 0.f), t1, t2, t3)) {

				//int fragmentIdx = j + (i * w);
				//int lerp_depth_int = conductDepthTest(j, i, t1, t2, t3, fragmentIdx, depth);

				//IMPORTANT! 
				//Should interpolate Z (depth) value first
				//z (depth) value is in camera(eye) space
				float lerp_depth = depthValuePersCorrectionLerp(glm::vec3(j, i, 0.f), t1, t2, t3);

				// OK... atomicMin only works for Int
				// but we want more accuracy for our depth value
				// so TRICK here!
				// multiply a really large number to get accuracy
				// just pay attention integar is between -2147483648 - 2147483647
				// 10000 may be a acceptable number
				int lerp_depth_int = (int)(lerp_depth * 100000.0f);

				// Atomic depth buffer writing
				int fragmentIdx = j + (i * w);
				int old = depth[fragmentIdx];
				int assumed;

				do {
					assumed = old;
					old = atomicMin(&depth[fragmentIdx], lerp_depth_int);
				} while (assumed != old);


				//must use depth[index] to read again!
				if (lerp_depth_int <= depth[fragmentIdx]) {
					// pass depth test, this fragment is good, we will use it
					glm::vec3 p((float)j, (float)i, lerp_depth);

					// fill this fragment Buffer
					fillThisFragmentBuffer(fragmentBuffer[fragmentIdx],
						p,
						t1, t2, t3,
						thisPrimitive.v[0], thisPrimitive.v[1], thisPrimitive.v[2]);

				}
			}
		}
	}

#endif // MSAAx2

}

// Rasterizer - Fill method
// wireFrame mode
__device__
void rasterizer_fill_wireFrameMode(Fragment* fragmentBuffer, int* depth,
								   glm::vec3 t1, glm::vec3 t2, glm::vec3 t3,
								   int w, int h) 
{
	//Use AABB
	float minX = fminf(t1.x, fminf(t2.x, t3.x));
	float maxX = fmaxf(t1.x, fmaxf(t2.x, t3.x));
	float minY = fminf(t1.y, fminf(t2.y, t3.y));
	float maxY = fmaxf(t1.y, fmaxf(t2.y, t3.y));

	// make sure AABB is inside screen
	int startX = minX < 0 ? 0 : (int)glm::floor(minX);
	int endX = maxX > w ? w : (int)glm::ceil(maxX);

	int startY = minY < 0 ? 0 : (int)glm::floor(minY);
	int endY = maxY > h ? h : (int)glm::ceil(maxY);

	glm::vec3 tris[3];
	tris[0] = t1;
	tris[1] = t2;
	tris[2] = t3;
	int fragmentIdx;

	float Epsilon = 0.08f; // this controls the accuracy(thickness) of each line segment
	glm::vec3 wireFrameCol = glm::vec3(0.35f, 0.85f, 0.35f);


	for (int i = startY; i <= endY; i++) {
		for (int j = startX; j <= endX; j++) {

			glm::vec3 barycentricCoord = calculateBarycentricCoordinate(tris, glm::vec2(j, i));

			if (glm::abs(barycentricCoord.x) < Epsilon) {
				if (barycentricCoord.y >= 0.0f && barycentricCoord.y <= 1.0f &&
					barycentricCoord.z >= 0.0f && barycentricCoord.z <= 1.0f) {
					fragmentIdx = j + (i * w);
					fragmentBuffer[fragmentIdx].color = wireFrameCol;
				}
			}
			else if (glm::abs(barycentricCoord.y) < Epsilon) {
				if (barycentricCoord.x >= 0.0f && barycentricCoord.x <= 1.0f &&
					barycentricCoord.z >= 0.0f && barycentricCoord.z <= 1.0f) {
					fragmentIdx = j + (i * w);
					fragmentBuffer[fragmentIdx].color = wireFrameCol;
				}
			}
			else if (glm::abs(barycentricCoord.z) < Epsilon) {
				if (barycentricCoord.y >= 0.0f && barycentricCoord.y <= 1.0f &&
					barycentricCoord.x >= 0.0f && barycentricCoord.x <= 1.0f) {
					fragmentIdx = j + (i * w);
					fragmentBuffer[fragmentIdx].color = wireFrameCol;
				}
			}

		}
	}
}

// Rasterizer - Fill method
// point mode
__device__
void rasterizer_fill_pointMode(Fragment* fragmentBuffer, int* depth,
							   glm::vec3 t1, glm::vec3 t2, glm::vec3 t3,
							   int w, int h)
{
	//Use AABB
	float minX = fminf(t1.x, fminf(t2.x, t3.x));
	float maxX = fmaxf(t1.x, fmaxf(t2.x, t3.x));
	float minY = fminf(t1.y, fminf(t2.y, t3.y));
	float maxY = fmaxf(t1.y, fmaxf(t2.y, t3.y));

	// make sure AABB is inside screen
	int startX = minX < 0 ? 0 : (int)glm::floor(minX);
	int endX = maxX > w ? w : (int)glm::ceil(maxX);

	int startY = minY < 0 ? 0 : (int)glm::floor(minY);
	int endY = maxY > h ? h : (int)glm::ceil(maxY);

	glm::vec3 tris[3];
	tris[0] = t1;
	tris[1] = t2;
	tris[2] = t3;
	int fragmentIdx;

	float Epsilon = 0.08f; // this controls the accuracy(thickness) of each line segment
	glm::vec3 pointCol = glm::vec3(0.85f, 0.85f, 0.85f);


	for (int i = startY; i <= endY; i++) {
		for (int j = startX; j <= endX; j++) {

			glm::vec3 barycentricCoord = calculateBarycentricCoordinate(tris, glm::vec2(j, i));

			if (glm::abs(barycentricCoord.x - 1.0f) < Epsilon) {
				if (glm::abs(barycentricCoord.y) < Epsilon &&
					glm::abs(barycentricCoord.z) < Epsilon) {
					fragmentIdx = j + (i * w);
					fragmentBuffer[fragmentIdx].color = pointCol;
				}
			}
			else if (glm::abs(barycentricCoord.y - 1.0f) < Epsilon) {
				if (glm::abs(barycentricCoord.x) < Epsilon &&
					glm::abs(barycentricCoord.z) < Epsilon) {
					fragmentIdx = j + (i * w);
					fragmentBuffer[fragmentIdx].color = pointCol;
				}
			}
			else if (glm::abs(barycentricCoord.z - 1.0f) < Epsilon) {
				if (glm::abs(barycentricCoord.x) < Epsilon &&
					glm::abs(barycentricCoord.y) < Epsilon) {
					fragmentIdx = j + (i * w);
					fragmentBuffer[fragmentIdx].color = pointCol;
				}
			}

		}
	}
}

// Rasterizer - Fill method
// Goal is to fill Fragment buffer
__global__
void rasterizer_fill(int numPrimitives, int curPrimitiveBeginId, Primitive* primitives, Fragment* fragmentBuffer, int* depth, int w, int h, int renderMode, glm::vec3 viewForwardVec)
{
	int primitiveIdx = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (primitiveIdx < numPrimitives) {

#ifdef BACKFACE_CULLING_IN_PIPELINE
		Primitive& thisPrimitive = primitives[primitiveIdx];
#else
		Primitive& thisPrimitive = primitives[primitiveIdx + curPrimitiveBeginId];
#endif 

#ifdef BACKFACE_CULLING_IN_RASTERIZER
		// Naive Back-face culling
		if (glm::dot(thisPrimitive.v[0].eyeNor, viewForwardVec) < 0 && 
			glm::dot(thisPrimitive.v[1].eyeNor, viewForwardVec) < 0 &&
			glm::dot(thisPrimitive.v[2].eyeNor, viewForwardVec) < 0) {
			return;
		}
#endif
		// need to use NDC depth value, so that all depth are realtive to near clip
		glm::vec3 t1(thisPrimitive.v[0].pos[0], thisPrimitive.v[0].pos[1], thisPrimitive.v[0].pos[2]);
		glm::vec3 t2(thisPrimitive.v[1].pos[0], thisPrimitive.v[1].pos[1], thisPrimitive.v[1].pos[2]);
		glm::vec3 t3(thisPrimitive.v[2].pos[0], thisPrimitive.v[2].pos[1], thisPrimitive.v[2].pos[2]);

		// Rasterize whole triangle
		if (renderMode == 1) {
			rasterizer_fill_wholeTriangleMode(fragmentBuffer, thisPrimitive, depth, 
											  t1, t2, t3,
											  w, h
											 );
		}

		// Rasterize wireframe
		if (renderMode == 2) {
			rasterizer_fill_wireFrameMode(fragmentBuffer, depth,
									      t1, t2, t3,
										  w, h);
		}

		// Rasterize point
		if (renderMode == 3) {
			rasterizer_fill_pointMode(fragmentBuffer, depth,
								      t1, t2, t3,
									  w, h);
		}

	}
}

#ifdef BACKFACE_CULLING_IN_PIPELINE

struct isBackFacing
{	
	glm::vec3 viewForwardVec;
	isBackFacing(glm::vec3 vec) : viewForwardVec(vec) {};

	__host__ __device__
	bool  operator()(const Primitive x)
	{
		return (glm::dot(x.v[0].eyeNor, viewForwardVec) < 0 &&
				glm::dot(x.v[1].eyeNor, viewForwardVec) < 0 &&
				glm::dot(x.v[2].eyeNor, viewForwardVec) < 0) ;
	}
};
#endif


/**
 * Perform rasterization.
 */
void rasterize(uchar4 *pbo, const glm::mat4 & MVP, const glm::mat4 & MV, const glm::mat3 MV_normal, 
	int renderMode, glm::mat4 selfRotateM,
	bool openPostProcess, 
	glm::vec3 viewForwardVec) {
    int sideLength2d = 8;
    dim3 blockSize2d(sideLength2d, sideLength2d);
    dim3 blockCount2d((width  - 1) / blockSize2d.x + 1,
					  (height - 1) / blockSize2d.y + 1);

	// Execute your rasterization pipeline here
	// (See README for rasterization pipeline outline.)

	// Vertex Process & primitive assembly
	{
		curPrimitiveBeginId = 0;
		dim3 numThreadsPerBlock(128);

		auto it = mesh2PrimitivesMap.begin();
		auto itEnd = mesh2PrimitivesMap.end();

		for (; it != itEnd; ++it) {
			auto p = (it->second).begin();	// each primitive
			auto pEnd = (it->second).end();
			for (; p != pEnd; ++p) {
				dim3 numBlocksForVertices((p->numVertices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
				dim3 numBlocksForIndices((p->numIndices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);

				_vertexTransformAndAssembly << < numBlocksForVertices, numThreadsPerBlock >> >(p->numVertices, *p, MVP, MV, MV_normal, width, height, selfRotateM);
				checkCUDAError("Vertex Processing");
				cudaDeviceSynchronize();
				_primitiveAssembly << < numBlocksForIndices, numThreadsPerBlock >> >
					(p->numIndices, 
					curPrimitiveBeginId, 
					dev_primitives, 
					*p);
				checkCUDAError("Primitive Assembly");

				curPrimitiveBeginId += p->numPrimitives;
			}
		}

		checkCUDAError("Vertex Processing and Primitive Assembly");
	}
	
	cudaMemset(dev_fragmentBuffer, 0, width * height * sizeof(Fragment));

	initDepth << <blockCount2d, blockSize2d >> >(width, height, dev_depth);

	// rasterize
	{	
		curPrimitiveBeginId = 0;
		dim3 numThreadsPerBlock(128);

		auto it = mesh2PrimitivesMap.begin();
		auto itEnd = mesh2PrimitivesMap.end();

		for (; it != itEnd; ++it) {
			auto p = (it->second).begin();	// each primitive
			auto pEnd = (it->second).end();
			for (; p != pEnd; ++p) {
				dim3 numBlocksForPrimitives((p->numPrimitives + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
				
#ifdef BACKFACE_CULLING_IN_PIPELINE
				// First copy Primitives to a new array
				cudaMemcpy(dev_primitives_after_backfaceCulling, dev_primitives + curPrimitiveBeginId, p->numPrimitives * sizeof(Primitive), cudaMemcpyDeviceToDevice);

				// Remove primitves facing backwards
				thrust::device_ptr<Primitive> dev_thrust_primitves(dev_primitives_after_backfaceCulling);
				int newPrimitiveSize = thrust::remove_if(dev_thrust_primitves, dev_thrust_primitves + p->numPrimitives, isBackFacing(viewForwardVec)) - dev_thrust_primitves;
				
				// Calculate new block size
				numBlocksForPrimitives = dim3((newPrimitiveSize + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);

				// rasterize based on new calculated primitives array
				rasterizer_fill << <numBlocksForPrimitives, numThreadsPerBlock >> >
					(newPrimitiveSize, curPrimitiveBeginId,
					 dev_primitives_after_backfaceCulling, dev_fragmentBuffer, dev_depth,
					 width, height, renderMode,
					 viewForwardVec);
				checkCUDAError("rasterizer_fill");
				cudaDeviceSynchronize();
				curPrimitiveBeginId += p->numPrimitives;

#else
				rasterizer_fill << <numBlocksForPrimitives, numThreadsPerBlock >> > 
					(p->numPrimitives, curPrimitiveBeginId, 
					 dev_primitives, dev_fragmentBuffer, dev_depth, 
					 width, height, renderMode,
					 viewForwardVec);
				checkCUDAError("rasterizer_fill");
				cudaDeviceSynchronize();
				curPrimitiveBeginId += p->numPrimitives;
#endif
			}
		}
	}

	//point light position for Lambert shading
	glm::vec3 lightPos(3.0f, 6.0f, -5.0f);

    // Copy depthbuffer colors into framebuffer
	render << <blockCount2d, blockSize2d >> >(width, height, lightPos, dev_fragmentBuffer, dev_framebuffer, renderMode, GaussianBlurEdgeRoom);
	checkCUDAError("fragment shader");

	if (openPostProcess) {
		// Post-processing Stage

		//---------------------- Bloom Effect Starts --------------------------------
		// Bright Filter
		brightFilter << <blockCount2d, blockSize2d >> > (width, height, dev_framebuffer, dev_framebuffer1);


		// Down Scale
		int downScaleRate = 10;
		dim3 blockCount2d_DownScaleBy10((width / downScaleRate - 1) / blockSize2d.x + 1,
			(height / downScaleRate - 1) / blockSize2d.y + 1);
		sampleDownScaleSample << <blockCount2d_DownScaleBy10, blockSize2d >> > (width / downScaleRate, height / downScaleRate, downScaleRate,
			width, height,
			dev_framebuffer_DownScaleBy10, dev_framebuffer1, GaussianBlurEdgeRoom);


		// GaussianBlur 11 samples horizontally and vertically in our case
		// Make Sure blockSize2d not change, we need to decide shared memory size based on that
		horizontalGaussianBlur << <blockCount2d_DownScaleBy10, blockSize2d >> > (width / downScaleRate, height / downScaleRate, dev_framebuffer_DownScaleBy10, dev_framebuffer_DownScaleBy10_2, GaussianBlurEdgeRoom);
		verticalGaussianBlur << <blockCount2d_DownScaleBy10, blockSize2d >> > (width / downScaleRate, height / downScaleRate, dev_framebuffer_DownScaleBy10_2, dev_framebuffer_DownScaleBy10, GaussianBlurEdgeRoom);


		// Combine
		combineFrameBuffer << <blockCount2d, blockSize2d >> > (width, height,
			dev_framebuffer, dev_framebuffer_DownScaleBy10, dev_framebuffer1,
			width / downScaleRate, downScaleRate, GaussianBlurEdgeRoom);
		checkCUDAError("post processing");
		//---------------------- Bloom Effect Ends --------------------------------


#if defined(SSAAx2) || defined(MSAAx2)
		dim3 blockCount2d_AAx2_DownScaleBy2(((width / 2) - 1)  / blockSize2d.x + 1,
										      ((height / 2) - 1) / blockSize2d.y + 1);
		sendImageToPBO_AAxN << <blockCount2d_AAx2_DownScaleBy2, blockSize2d >> > (pbo, width / 2, height / 2, dev_framebuffer1, 2);

#else
		// Copy framebuffer into OpenGL buffer for OpenGL previewing
		sendImageToPBO << <blockCount2d, blockSize2d >> > (pbo, width, height, dev_framebuffer1, GaussianBlurEdgeRoom, width / 1, 1);

		// Downscale Debug
		//sendImageToPBO << <blockCount2d, blockSize2d >> >(pbo, width, height, dev_framebuffer_DownScaleBy10, GaussianBlurEdgeRoom, width / downScaleRate, downScaleRate);
#endif
		checkCUDAError("copy render result to pbo");
	}

	//Ignore post processing stage
	else {
#if defined(SSAAx2) || defined(MSAAx2)
		dim3 blockCount2d_AAx2_DownScaleBy2(((width / 2) - 1)  / blockSize2d.x + 1,
											  ((height / 2) - 1) / blockSize2d.y + 1);
		sendImageToPBO_AAxN << <blockCount2d_AAx2_DownScaleBy2, blockSize2d >> > (pbo, width / 2, height / 2, dev_framebuffer, 2);

#else
		sendImageToPBO << <blockCount2d, blockSize2d >> > (pbo, width, height, dev_framebuffer, GaussianBlurEdgeRoom, width / 1, 1);
#endif
	}
}

/**
 * Called once at the end of the program to free CUDA memory.
 */
void rasterizeFree() {

    // deconstruct primitives attribute/indices device buffer

	auto it(mesh2PrimitivesMap.begin());
	auto itEnd(mesh2PrimitivesMap.end());
	for (; it != itEnd; ++it) {
		for (auto p = it->second.begin(); p != it->second.end(); ++p) {
			cudaFree(p->dev_indices);
			cudaFree(p->dev_position);
			cudaFree(p->dev_normal);
			cudaFree(p->dev_texcoord0);
			cudaFree(p->dev_diffuseTex);

			cudaFree(p->dev_verticesOut);

			
			//TODO: release other attributes and materials
		}
	}

	////////////

    cudaFree(dev_primitives);
    dev_primitives = NULL;

#ifdef BACKFACE_CULLING_IN_PIPELINE
	cudaFree(dev_primitives_after_backfaceCulling);
	dev_primitives_after_backfaceCulling = NULL;
#endif

	cudaFree(dev_fragmentBuffer);
	dev_fragmentBuffer = NULL;

    cudaFree(dev_framebuffer);
    dev_framebuffer = NULL;

	cudaFree(dev_framebuffer1);
	dev_framebuffer1 = NULL;

	cudaFree(dev_framebuffer_DownScaleBy10);
	dev_framebuffer_DownScaleBy10 = NULL;

	cudaFree(dev_framebuffer_DownScaleBy10_2);
	dev_framebuffer_DownScaleBy10_2 = NULL;

	cudaFree(dev_depth);
	dev_depth = NULL;

    checkCUDAError("rasterize Free");
}
