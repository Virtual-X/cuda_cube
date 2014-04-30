#include "Constants.h"
#include "TimerC.h"
#include "utils.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/transform.h>

#include <list>

enum {
	MaxSolutions = 10000,
	MaxInput = 64 * 1024,
	MaxOutput = 2 * 1024 * 1024
};

template<typename T>
__host__
static T* raw(thrust::device_vector<T>& vector)
{
	return thrust::raw_pointer_cast(vector.data());
}

__host__ __device__
int GetCandidatesOffsetIndex(int position, int code, int piece) {
	return piece + PiecesCount * (code + CodesCount * position);
}

__device__
int GetCode(uint64_t m, int p) {
	return ((m >> (p + 1)) & 0x1f) | ((m >> (p + 15 - 5)) & 0xe0);
}

__device__
int GetNextPos(uint64_t grid, int position, const int* d_nextValidPosition) {
	int pos = d_nextValidPosition[(grid >> position) & 0xff];
	if (pos >= 0)
		return position + pos;
	int byte = position / 8;
	for (int i = byte + 1; i < 8; i++) {
		pos = d_nextValidPosition[(grid >> (i * 8)) & 0xff];
		if (pos >= 0)
			return (i * 8) + pos;
	}
	return -1;
}

__host__
int InitNextValidPosition(int i) {
	int validPos = -1;
	for (int k = 0; k < 8; k++) {
		if ((i & (1 << k)) == 0) {
			validPos = k;
			break;
		}
	}
	return validPos;
}

struct Globals {
	const int2* d_candidatesOffsets; // [PositionsCount][CodesCount][PiecesCount], d_candidatesOffsets[GetCandidatesOffsetIndex(position, code, piece)]
	CandidatesMask d_candidatesMask;
	const int* d_nextValidPosition; // [256];

	SituationT* d_solutions;
	int* d_solutionsCount;

	int* d_solversCount; // [PiecesCount]
};

struct Locals {
	uint64_t* d_grid;
	int* d_position;
	int* d_permutationOrder;
	int* d_situation;
};

int GetActualPiece(const thrust::host_vector<int>& solversCountH)
{
    for (int i = solversCountH.size(); i > 0; i--) {
    	if (solversCountH[i - 1] >= MaxInput)
    		return i - 1;
    }
    for (int i = 0; i < PositionsCount; i++) {
    	if (solversCountH[i] > 0)
    		return i;
    }
    return -1;
}

__global__
void CudaInit(Globals globals, Locals locals)
{
	__shared__ int solversCount;

	const int ind = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (ind >= PiecesCount)
		return;

	if (ind == 0)
		solversCount = 0;

	__syncthreads();

	int2 beginEnd = globals.d_candidatesOffsets[ind];
    for (int index = beginEnd.x; index < beginEnd.y; index++) {
    	uint64_t mask = globals.d_candidatesMask[index]; // to be coalesced
		const int o = atomicAdd(&solversCount, 1);
		locals.d_position[o] = 1;
		locals.d_grid[o] = mask;

		int* current = locals.d_situation + ind;
    	current[0] = index;
        for (int k = 1; k < PiecesCount; k++) {
        	current[k * MaxOutput] = k;
		}
        if (ind > 0) {
        	current[ind * MaxOutput] = 0;
        }
    }

	__syncthreads();

	if (ind == 0)
		globals.d_solversCount[0] = solversCount;
}


template<bool IsLastStep>
__global__
void CudaStep(const Globals globals, const Locals locals, const int actualPiece, const int off, const int count)
{
	const int ind = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (ind >= count)
	  return;

	const int id = actualPiece * MaxOutput + off + ind;
	const uint64_t grid = locals.d_grid[id];
	const int position = locals.d_position[id];

	const int pos = GetNextPos(grid, position, globals.d_nextValidPosition);
    const int code = GetCode(grid, pos);

    const int2* d_candidatesOffsets_ = globals.d_candidatesOffsets + GetCandidatesOffsetIndex(pos, code, 0);
	for (int i = actualPiece; i < PiecesCount; i++) {
		const int id2 = (actualPiece * PiecesCount + i) * MaxOutput + off + ind;
		const int piece = locals.d_situation[id2];
		int2 beginEnd = d_candidatesOffsets_[piece];
        for (int index = beginEnd.x; index < beginEnd.y; index++) {
        	uint64_t mask = globals.d_candidatesMask[index]; // to be coalesced
            if (!(mask & grid)) {
            	if (IsLastStep) {
            		const int n = atomicAdd(globals.d_solutionsCount, 1);
            		const int* current = locals.d_situation + (actualPiece * PiecesCount * MaxOutput + off + ind);
            		int* solution = globals.d_solutions[n].pieces;
                    for (int k = 0; k < actualPiece; k++) {
            			solution[k] = current[k * MaxOutput];
            		}
                    solution[actualPiece] = i;
            	}
            	else {
            		const int n = atomicAdd(globals.d_solversCount + (actualPiece + 1), 1);
            		if (n >= MaxOutput)
            			return;

            		int o = (actualPiece + 1) * MaxOutput + n;
            		locals.d_position[o] = pos + 1;
            		locals.d_grid[o] = mask | grid;

            		const int* current = locals.d_situation + (actualPiece * PiecesCount * MaxOutput + off + ind);
            		int* other = locals.d_situation + ((actualPiece + 1) * PiecesCount * MaxOutput + n);
                    for (int k = 0; k < actualPiece; k++) {
                    	other[k * MaxOutput] = current[k * MaxOutput];
            		}
                    other[actualPiece * MaxOutput] = index; // last element of situation
                    for (int k = actualPiece + 1; k < i; k++) {
                    	other[k * MaxOutput] = current[k * MaxOutput];
            		}
                    if (i > actualPiece) {
                    	other[i * MaxOutput] = current[actualPiece * MaxOutput]; // permutation order
                    }
                    for (int k = i + 1; k < PiecesCount; k++) {
                    	other[k * MaxOutput] = current[k * MaxOutput];
            		}
            	}
            }
        }
	}

	if (ind == 0)
		atomicSub(globals.d_solversCount + actualPiece, count);
}

void CudaSolve(CandidatesOffsets candidatesOffsets, CandidatesMask candidatesMask, std::list<SituationT>& solutions)
{
	TimerC timerInit;

	Globals globals;

	thrust::host_vector<int2> candidatesOffsetsH(PositionsCount * CodesCount * PiecesCount);
    for (int i = 0; i < PositionsCount; i++) {
        for (int j = 0; j < CodesCount; j++) {
            for (int k = 0; k < PiecesCount; k++) {
            	const int (&co)[PiecesCount + 1] = candidatesOffsets[i][j];
            	candidatesOffsetsH[GetCandidatesOffsetIndex(i, j, k)] = make_int2(co[k], co[k + 1]);
            }
        }
    }
	thrust::device_vector<int2> candidatesOffsetsV(candidatesOffsetsH);
	globals.d_candidatesOffsets = raw(candidatesOffsetsV);

	thrust::device_vector<uint64_t> candidatesMaskV(candidatesMask, candidatesMask + candidatesOffsets[PositionsCount - 1][CodesCount - 1][PiecesCount]);
	globals.d_candidatesMask = raw(candidatesMaskV);

	thrust::device_vector<int> nextValidPositionV(256);
    for (int i = 0; i < 256; i++) {
    	nextValidPositionV[i] = InitNextValidPosition(i);
    }
	globals.d_nextValidPosition = raw(nextValidPositionV);

	thrust::device_vector<int> solutionsCountV(1);
	globals.d_solutionsCount = raw(solutionsCountV);

	thrust::device_vector<SituationT> solutionsV(MaxSolutions);
	globals.d_solutions = raw(solutionsV);

	thrust::device_vector<int> solversCountV(PiecesCount);
	globals.d_solversCount = raw(solversCountV);

	Locals locals;

	thrust::device_vector<uint64_t> gridV(PiecesCount * MaxOutput); // actualPiece * MaxOutput + threadIdx
	locals.d_grid = raw(gridV);

	thrust::device_vector<int> positionV(PiecesCount * MaxOutput); // actualPiece * MaxOutput + threadIdx
	locals.d_position = raw(positionV);

	// permutationOrder up to actualPiece, situation afterwards
	thrust::device_vector<int> situationV(PiecesCount * PiecesCount * MaxOutput); // (actualPiece * PiecesCount + piece) * MaxOutput + threadIdx
	locals.d_situation = raw(situationV);

	int actualPiece = 0;

	CudaInit<<< 1, 256 >>>(globals, locals);

	thrust::host_vector<int> solversCountH(solversCountV);

	timerInit.Record("Init");
	int n = 0;
	do {
		TimerC timerLoop;

		int count = solversCountV[actualPiece];

		std::cout << n++ << ": s[" << actualPiece << "] = " << count;

		const int off = count > MaxInput ? count - MaxInput : 0;
		count = min(count, MaxInput);
	    const int blockSize = 512;
	    const int gridSize = (count + blockSize - 1) / blockSize;
	    if (actualPiece == PiecesCount - 1) {
	    	CudaStep<true>
	    	<<< gridSize, blockSize >>>
	    	(globals, locals, actualPiece, off, count);
	    }
	    else {
	    	CudaStep<false>
	    	<<< gridSize, blockSize >>>
	    	(globals, locals, actualPiece, off, count);
	    }
	    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());


		solversCountH[actualPiece] = solversCountV[actualPiece];
		solversCountH[actualPiece + 1] = solversCountV[actualPiece + 1];
		actualPiece = GetActualPiece(solversCountH);

		timerLoop.Record("");
		if (solutionsCountV[0] > 0)
			break;
	} while (actualPiece >= 0);

	solutions.insert(solutions.end(), solutionsV.begin(), solutionsV.begin() + solutionsCountV[0]);

	timerInit.Record("Total");
	std::cout << "Solutions found: " << solutionsCountV[0] << std::endl;
}
