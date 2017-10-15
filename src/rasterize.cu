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
		// glm::vec3 col;
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
static Fragment *dev_fragmentBuffer = NULL;
static glm::vec3 *dev_framebuffer = NULL;

static int * dev_depth = NULL;	// you might need this buffer when doing depth test

/**
 * Kernel that writes the image to the OpenGL PBO directly.
 */
__global__ 
void sendImageToPBO(uchar4 *pbo, int w, int h, glm::vec3 *image) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * w);

    if (x < w && y < h) {
        glm::vec3 color;
        color.x = glm::clamp(image[index].x, 0.0f, 1.0f) * 255.0;
        color.y = glm::clamp(image[index].y, 0.0f, 1.0f) * 255.0;
        color.z = glm::clamp(image[index].z, 0.0f, 1.0f) * 255.0;
        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

/** 
* Writes fragment colors to the framebuffer
*/
__global__
void render(int w, int h, glm::vec3 lightPos, Fragment *fragmentBuffer, glm::vec3 *framebuffer, int renderMode) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * w);

    if (x < w && y < h) {
        //framebuffer[index] = fragmentBuffer[index].color;

		// TODO: add your fragment shader code here

		// whole Triangle render mode
		if (renderMode == 1) {
			// Lambert shading
			Fragment thisFragment = fragmentBuffer[index];

			glm::vec3 lightVec = lightPos - thisFragment.eyePos;
			lightVec = glm::normalize(lightVec);

			float light_cosTheta = glm::min(glm::max(glm::dot(thisFragment.eyeNor, lightVec), 0.0f), 1.0f);

			float light_intensity = light_cosTheta + 0.15f; // add ambient term so that we can still see points that are not lit by point light 

			framebuffer[index] = light_intensity * thisFragment.color;
		}

		// wireframe or point mode
		if (renderMode == 2 || renderMode == 3) {
			framebuffer[index] = fragmentBuffer[index].color;
		}


    }
}

/**
 * Called once at the beginning of the program to allocate memory.
 */
void rasterizeInit(int w, int h) {
    width = w;
    height = h;
	cudaFree(dev_fragmentBuffer);
	cudaMalloc(&dev_fragmentBuffer, width * height * sizeof(Fragment));
	cudaMemset(dev_fragmentBuffer, 0, width * height * sizeof(Fragment));
    cudaFree(dev_framebuffer);
    cudaMalloc(&dev_framebuffer,   width * height * sizeof(glm::vec3));
    cudaMemset(dev_framebuffer, 0, width * height * sizeof(glm::vec3));
    
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
		this_dev_verticesOut.eyePos = glm::vec3(MV * glm::vec4(primitive.dev_position[vid], 1.0f));

		// Handle eye space normal
		this_dev_verticesOut.eyeNor = glm::normalize(MV_normal * primitive.dev_normal[vid]); // normalized

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
		glm::ivec2 textSpaceCoord = glm::ivec2(diffuseTexWidth * lerp_uv.x, diffuseTexHeight * (lerp_uv.y));

		int textIdx = textSpaceCoord.x + diffuseTexWidth * textSpaceCoord.y;

		// Assume texture data are row major
		// and there are 3 channels
		int numOfTextureChannels = 3;
		TextureData r = textureData[textIdx * numOfTextureChannels];
		TextureData g = textureData[textIdx * numOfTextureChannels + 1];
		TextureData b = textureData[textIdx * numOfTextureChannels + 2];

		thisFragment.color = glm::vec3((float)r / 255.0f, (float)g / 255.0f, (float)b / 255.0f);
	}

	// Debug normal
	else {
		
		thisFragment.color = glm::vec3(0.5f * (lerp_eyeNor.x + 1.0f),
								       0.5f * (lerp_eyeNor.y + 1.0f),
									   0.5f * (lerp_eyeNor.z + 1.0f));
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
	int endX = maxX > w ? w : (int)glm::ceil(maxX);

	int startY = minY < 0 ? 0 : (int)glm::floor(minY);
	int endY = maxY > h ? h : (int)glm::ceil(maxY);

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
				int lerp_depth_int = (int)(lerp_depth * 10000.0f);

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
void rasterizer_fill(int numPrimitives, int curPrimitiveBeginId, Primitive* primitives, Fragment* fragmentBuffer, int* depth, int w, int h, int renderMode) {
	int primitiveIdx = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (primitiveIdx < numPrimitives) {
		Primitive& thisPrimitive = primitives[primitiveIdx + curPrimitiveBeginId];

		//glm::vec3 t1(thisPrimitive.v[0].pos[0], thisPrimitive.v[0].pos[1], thisPrimitive.v[0].eyePos[2]);
		glm::vec3 t1(thisPrimitive.v[0].pos[0], thisPrimitive.v[0].pos[1], thisPrimitive.v[0].pos[2]);

		//glm::vec3 t2(thisPrimitive.v[1].pos[0], thisPrimitive.v[1].pos[1], thisPrimitive.v[1].eyePos[2]);
		glm::vec3 t2(thisPrimitive.v[1].pos[0], thisPrimitive.v[1].pos[1], thisPrimitive.v[1].pos[2]);

		//glm::vec3 t3(thisPrimitive.v[2].pos[0], thisPrimitive.v[2].pos[1], thisPrimitive.v[2].eyePos[2]);
		glm::vec3 t3(thisPrimitive.v[2].pos[0], thisPrimitive.v[2].pos[1], thisPrimitive.v[2].pos[2]);

		// Rasterize whole triangle
		if (renderMode == 1) {
			rasterizer_fill_wholeTriangleMode(fragmentBuffer, thisPrimitive, depth, 
											  t1, t2, t3,
											  w, h);
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


/**
 * Perform rasterization.
 */
void rasterize(uchar4 *pbo, const glm::mat4 & MVP, const glm::mat4 & MV, const glm::mat3 MV_normal, int renderMode, glm::mat4 selfRotateM) {
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
				
				rasterizer_fill << <numBlocksForPrimitives, numThreadsPerBlock >> > (p->numPrimitives, curPrimitiveBeginId, dev_primitives, dev_fragmentBuffer, dev_depth, width, height, renderMode);
				checkCUDAError("rasterizer_fill");
				cudaDeviceSynchronize();
				curPrimitiveBeginId += p->numPrimitives;
			}
		}
	}

	//point light position for Lambert shading
	glm::vec3 lightPos(3.0f, 6.0f, -5.0f);

    // Copy depthbuffer colors into framebuffer
	render << <blockCount2d, blockSize2d >> >(width, height, lightPos, dev_fragmentBuffer, dev_framebuffer, renderMode);
	checkCUDAError("fragment shader");
    // Copy framebuffer into OpenGL buffer for OpenGL previewing
    sendImageToPBO<<<blockCount2d, blockSize2d>>>(pbo, width, height, dev_framebuffer);
    checkCUDAError("copy render result to pbo");
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

	cudaFree(dev_fragmentBuffer);
	dev_fragmentBuffer = NULL;

    cudaFree(dev_framebuffer);
    dev_framebuffer = NULL;

	cudaFree(dev_depth);
	dev_depth = NULL;

    checkCUDAError("rasterize Free");
}
