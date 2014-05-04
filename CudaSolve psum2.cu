#include "Constants.h"
#include "TimerC.h"
#include "utils.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#if defined _DEBUG
const bool DoDebug = true;
#else
const bool DoDebug = false;
#endif

enum {
	MaxInput = 64 * 1024,
	MaxOutput = 2 * 1024 * 1024
};

enum ErrorCode {
	ErrorCodeFinished = -1,
	ErrorCodeNone = 0,
	ErrorCodeInvalidPos = 1,
	ErrorCodeInvalidCode = 1 << 1,
	ErrorCodeInvalidIndex = 1 << 2,
	ErrorCodeInvalidId = 1 << 3,
	ErrorCodeInvalidId2 = 1 << 4,
	ErrorCodeInvalidPiece = 1 << 5,
	ErrorCodeInvalidCandidatesOffsetIndex = 1 << 6,
	ErrorCodeInvalidLastStepFlag = 1 << 7,
	ErrorCodeInvalidSolutionsCount = 1 << 8,
	ErrorCodeInvalidSolversCount = 1 << 9,
	ErrorCodeInvalidO_LastStep = 1 << 10,
	ErrorCodeInvalidO_NotLastStep = 1 << 11,
	ErrorCodeInvalidP_NotLastStep = 1 << 12,
	ErrorCodeInvalidQ_NotLastStep = 1 << 13,
	ErrorCodeInvalidX_NotLastStep = 1 << 14,
	ErrorCodeInvalidActualPiece = 1 << 15,
	ErrorCodeInvalidLastStepFlagA = 1 << 16,
	ErrorCodeInvalidLastStepFlagB = 1 << 17,
	ErrorCodeInvalidLastStepFlagC = 1 << 18,
	ErrorCodeInvalidCount = 1 << 19,
	ErrorCodeInvalidPsum1 = 1 << 20,
	ErrorCodeInvalidPsum2 = 1 << 21
};

typedef volatile int vint;
typedef volatile uint64_t vuint64_t;

template<typename T>
__host__
static T* raw(thrust::device_vector<T>& vector) {
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

__constant__
int d_nextValidPosition[256]; // better global? or just while?

__device__
int GetNextPos(uint64_t grid, int position) {
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

	int* d_error;
};

struct Locals {
	uint64_t* d_grid;
	int* d_position;
	int* d_situation;
};

__host__ __device__
int GetActualPiece(const int* solversCount) {
	for (int i = PiecesCount - 1; i >= 0; i--) {
		if (solversCount[i] >= MaxInput)
			return i;
	}
	for (int i = 0; i < PiecesCount; i++) {
		if (solversCount[i] > 0)
			return i;
	}
	return -1;
}

__device__
bool CheckIndex(const int index, const int count, const Globals& globals, ErrorCode error) {
	if (index >= count) {
		globals.d_error[0] = error;
		globals.d_error[1] = index;
	}
	return index >= count;
}

enum {
	BlockSize = 64
};

// could do better: http://www.cudahandbook.com/uploads/Chapter_13._Scan.pdf
__device__
int shared_psum(const int idx, vint* _data, const int count) {
	__syncthreads();
	const int last = _data[count - 1];
	int d = idx > 0 ? _data[idx - 1] : 0;
	__syncthreads();
	_data[idx] = d;
	__syncthreads();
	for (int offset = 1; offset < count; offset *= 2) {
		if (idx >= offset)
			d = _data[idx - offset];
		__syncthreads();
		if (idx >= offset)
			_data[idx] += d;
		__syncthreads();
	}

	__syncthreads();
	return last + _data[count - 1];
}

__global__
void SumScan(int* data, const int count) {
	__shared__ volatile int _data[BlockSize];
	const int idx = threadIdx.x;
	if (idx >= count)
		return;
	_data[idx] = data[idx];
	__syncthreads();
	int tot = shared_psum(idx, _data, count);
	data[idx] = _data[idx];
	__syncthreads();
	if (idx == 0)
		data[count] = tot;
}

__device__ __host__
int get_index(int value, const vint* _psum, const int count, const int totSum) {
	int i = count * value / totSum;
	if (_psum[i] > value) {
		i--;
		while (_psum[i] > value)
			i--;
		return i;
	}
	while (i < count && _psum[i] <= value)
		i++;

	return i - 1;
}

template<bool Debug>
__device__
void InitStep(const Globals globals, const Locals locals, const int actualPiece, const int ind,
		vint (&_id2)[BlockSize], vint (&_pos)[BlockSize], vuint64_t (&_grid)[BlockSize],
		const int* (&situation), const int2* (&candidatesOffsets)) {
	const int id = actualPiece * MaxOutput + ind;
	if (Debug && CheckIndex(id, PiecesCount * MaxOutput, globals, ErrorCodeInvalidId))
		return;

	const int id2 = actualPiece * PiecesCount * MaxOutput + ind;
	if (Debug && CheckIndex(id2 + (PiecesCount - 1) * MaxOutput, PiecesCount * PiecesCount * MaxOutput,
			globals, ErrorCodeInvalidId2))
		return;

	const uint64_t grid = locals.d_grid[id];
	const int position = locals.d_position[id];

	const int pos = GetNextPos(grid, position);
	if (Debug && CheckIndex(pos, 64, globals, ErrorCodeInvalidPos))
		return;

	const int code = GetCode(grid, pos);
	if (Debug && CheckIndex(code, 256, globals, ErrorCodeInvalidCode))
		return;

	const int candidatesOffsetIndex = GetCandidatesOffsetIndex(pos, code, 0);
	if (Debug && CheckIndex(candidatesOffsetIndex + PiecesCount - 1, PositionsCount * CodesCount * PiecesCount,
			globals, ErrorCodeInvalidCandidatesOffsetIndex))
		return;

	const int idx = threadIdx.x;
	_id2[idx] = id2;
	_pos[idx] = pos + 1;
	_grid[idx] = grid;

	situation += id2;
	candidatesOffsets += candidatesOffsetIndex;
}

template<bool Debug>
__device__
void GetBeginAndCount(const Globals globals, const int* situation, const int2* candidatesOffsets, const int candidate,
		volatile int *_begin, volatile int* _count) {
	const int piece = situation[candidate * MaxOutput];
	if (Debug && CheckIndex(piece, PiecesCount, globals, ErrorCodeInvalidPiece))
		return;

	const int2 beginEnd = candidatesOffsets[piece]; // to be replaced by begin and count

	const int idx = threadIdx.x;
	_begin[idx] = beginEnd.x;
	_count[idx] = beginEnd.y - beginEnd.x;
}

template<bool Debug>
__device__
void AddSolution(const Globals globals, const int actualPiece,
		const int* current, const int other_ind, const int maskIndex) {
	if (Debug && CheckIndex(actualPiece, PiecesCount, globals, ErrorCodeInvalidLastStepFlagA))
		return;
	if (Debug && CheckIndex(PiecesCount, actualPiece + 2, globals, ErrorCodeInvalidLastStepFlagB))
		return;

	if (Debug && CheckIndex(other_ind, MaxSolutions, globals, ErrorCodeInvalidSolutionsCount))
		return;

	int* solution = globals.d_solutions[other_ind].pieces;
	for (int k = 0; k < actualPiece; k++) {
		solution[k] = current[k * MaxOutput];
	}
	solution[actualPiece] = maskIndex; // last element of situation
}

template<bool Debug>
__device__
bool InitNextStep(const Globals globals, const Locals locals, const int actualPiece,
		const int* current, const int other_ind, const int maskIndex,
		const int candidate, const int pos, const uint64_t grid) {

	if (Debug && CheckIndex(actualPiece, PiecesCount - 1, globals, ErrorCodeInvalidLastStepFlagC))
		return false;

	if (Debug && CheckIndex(other_ind, MaxOutput, globals, ErrorCodeInvalidSolversCount))
		return false;

	const int other_id = (actualPiece + 1) * MaxOutput + other_ind;
	if (Debug && CheckIndex(other_id, PiecesCount * MaxOutput,
			globals, ErrorCodeInvalidO_NotLastStep))
		return false;

	locals.d_position[other_id] = pos;
	locals.d_grid[other_id] = grid;

	if (Debug && CheckIndex(actualPiece + 1, PiecesCount,
			globals, ErrorCodeInvalidActualPiece))
		return false;

	const int other_id2 = (actualPiece + 1) * PiecesCount * MaxOutput + other_ind;
	if (Debug && CheckIndex(other_id2 + (PiecesCount - 1) * MaxOutput, PiecesCount * PiecesCount * MaxOutput,
			globals, ErrorCodeInvalidQ_NotLastStep))
		return false;

	int* other = locals.d_situation + other_id2;
	for (int k = 0; k < PiecesCount; k++) {
		other[k * MaxOutput] = current[k * MaxOutput];
	}

	other[candidate * MaxOutput] = current[actualPiece * MaxOutput]; // permutation order
	other[actualPiece * MaxOutput] = maskIndex; // last element of situation

	if (Debug) {
		for (int k = actualPiece + 1; k < PiecesCount; k++) {
			if (CheckIndex(other[k * MaxOutput], PiecesCount, globals, ErrorCodeInvalidX_NotLastStep))
				return false;
		}
	}

	return true;
}

template<bool IsLastStep, bool Debug>
__global__
void CudaStep(const Globals globals, const Locals locals, const int actualPiece, const int off, const int count) {
	__shared__ int _id2[BlockSize];
	__shared__ int _pos[BlockSize];
	__shared__ uint64_t _grid[BlockSize];

	__shared__ vint _begin[BlockSize];
	__shared__ vint _count[BlockSize];
	__shared__ vint _temp[BlockSize + 1];

//	__shared__ vint __id2[BlockSize];
//	__shared__ vint __pos[BlockSize];
//	__shared__ vuint64_t __grid[BlockSize];
//	__shared__ vint __index[BlockSize];

	if (Debug && CheckIndex(0, count, globals, ErrorCodeInvalidCount))
		return;

	const int idx = threadIdx.x;
	const int ind = (blockIdx.x * blockDim.x) + idx;

	const int blockCount = min(count - blockIdx.x * blockDim.x, BlockSize);

//	if (blockIdx.x + 1 != 1) // gridDim.x)
//		return;

	const int* situation = locals.d_situation;
	const int2* candidatesOffsets = globals.d_candidatesOffsets;
	if (idx < blockCount)
		InitStep<Debug>(globals, locals, actualPiece, off + ind, _id2, _pos, _grid, situation, candidatesOffsets);

	for (int candidate = actualPiece; candidate < PiecesCount; candidate++) {
		if (idx < blockCount)
			GetBeginAndCount<Debug>(globals, situation, candidatesOffsets, candidate, _begin, _count);

		const int totCount = shared_psum(idx, _count, blockCount);

		if (totCount == 0)
			continue;

		// here we are changing the use of idx, work for others
		for (int j = 0; j < totCount; j += BlockSize) {
			const int k = j + idx;

			int px;
			int index;
			uint64_t mask;
			uint64_t grid;
			bool valid = k < totCount;

			if (valid) {
				px = get_index(k, _count, blockCount, totCount); // the same for many
				index = _begin[px] + (k - _count[px]); // somewhat increasing
				mask = globals.d_candidatesMask[index]; // somewhat coalesced
				grid = _grid[px];
				valid = (mask & grid) == 0;
			}

			_temp[idx] = valid ? 1 : 0;


			if (Debug && CheckIndex(0, min(totCount - j, BlockSize), globals, ErrorCodeInvalidPsum2))
				return;

			const int nvalid = shared_psum(idx, _temp, min(totCount - j, BlockSize));

			if (idx == 0) {
				int* targetCount = IsLastStep ? globals.d_solutionsCount : globals.d_solversCount + (actualPiece + 1);
				_temp[BlockSize] = atomicAdd(targetCount, nvalid);
			}
			__syncthreads(); // for _temp[BlockSize]

//			if (valid) {
//				// taking instead of setting might free the threads faster
//				const int compact = _temp[idx];
//				if (!IsLastStep) {
//					__grid[compact] = mask | grid;
//					__pos[compact] = _pos[px];
//				}
//				__id2[compact] = _id2[px];
//				__index[compact] = index;
//			}
//			__syncthreads(); // for compacting

			if (valid) {
				const int* current = locals.d_situation + _id2[px]; // somewhat increasing

				const int other_ind = _temp[BlockSize] + _temp[idx]; // coalesced!

				if (Debug && CheckIndex(other_ind, MaxOutput, globals, ErrorCodeInvalidId2))
					return;

				const int maskIndex = index;

				if (IsLastStep) {
					AddSolution<Debug>(globals, actualPiece, current, other_ind, maskIndex);
				}
				else {
					if (!InitNextStep<Debug>(globals, locals, actualPiece, current, other_ind, maskIndex, candidate, _pos[px], mask | grid))
						return;
				}
			}
		}
	}
}

template<bool Debug>
__global__
void CudaLoop(const Globals globals, const Locals locals, const int steps) {
	for (int i = 0; i < steps; i++) {
		int actualPiece = GetActualPiece(globals.d_solversCount);
		if (actualPiece < 0) {
			globals.d_error[0] = ErrorCodeFinished;
			return;
		}

		int count = globals.d_solversCount[actualPiece];
		const int off = count > MaxInput ? count - MaxInput : 0;
		count = min(count, MaxInput);

		globals.d_solversCount[actualPiece] -= count;

		const int blockSize = min(BlockSize, count);
		const int gridSize = (count + blockSize - 1) / blockSize;

		if (actualPiece == PiecesCount - 1) {
			CudaStep<true, DoDebug> <<<gridSize, blockSize>>>(globals, locals, actualPiece, off, count);
		}
		else {
			CudaStep<false, DoDebug> <<<gridSize, blockSize>>>(globals, locals, actualPiece, off, count);
		}
		cudaDeviceSynchronize();
	}
}

template<bool Debug>
int HostLoop(const Globals globals, const Locals locals, const int steps,
		thrust::device_vector<int>& solversCountV, thrust::host_vector<int>& solversCountH) {
	for (int i = 0; i < steps; i++) {
		int actualPiece = GetActualPiece(&solversCountH[0]);
		if (actualPiece < 0) {
			return -1;
		}

		int count = solversCountH[actualPiece];
		const int off = count > MaxInput ? count - MaxInput : 0;
		count = min(count, MaxInput);

		solversCountH[actualPiece] -= count;
		solversCountV[actualPiece] = solversCountH[actualPiece];

		const int blockSize = BlockSize;
		const int gridSize = (count + blockSize - 1) / blockSize;

		//std::cout << "Launch " << actualPiece << " (" << count << ")" << std::endl;

		if (actualPiece == PiecesCount - 1) {
			CudaStep<true, DoDebug> <<<gridSize, blockSize>>>(globals, locals, actualPiece, off, count);
		}
		else {
			CudaStep<false, DoDebug> <<<gridSize, blockSize>>>(globals, locals, actualPiece, off, count);
			solversCountH[actualPiece + 1] = solversCountV[actualPiece + 1];
		}
		cudaDeviceSynchronize();
		checkCudaErrors(cudaGetLastError());
	}
	return 0;
}

int CudaSolve(CandidatesOffsets candidatesOffsets, CandidatesMask candidatesMask, SituationT (&solutions)[MaxSolutions]) {

//	int psum[] = { 0, 1, 3, 3, 4, 7, 7, 15, 25, 25 };
//	for (int j = 25; j < 40; j++) {
//		for (int i = 0; i < j; i++) {
//			std::cout << get_index(i, psum, 10, j);
//		}
//		std::cout << std::endl;
//	}
	const int nn = BlockSize - 10;
	int data[nn];
	int psum[nn];
	int sum = 0;
	for (int i = 0; i < nn; i++) {
		data[i] = (int) (sin(i * 3) * 5 + 5);
//		std::cout << sum << " ";
		psum[i] = sum;
		sum += data[i];
	}
	thrust::device_vector<int> xx(data, data + nn);
	SumScan<<< 1, BlockSize >>>(raw(xx), nn - 1);
	cudaDeviceSynchronize();
	int res[nn];
	std::copy(xx.begin(), xx.end(), res);

//	std::cout << std::endl;
//	for (int i = nn - 10; i < nn; i++) {
//		std::cout << psum[i] << " ";
//	}
//	std::cout << std::endl;
//	for (int i = nn - 10; i < nn; i++) {
//		std::cout << res[i] << " ";
//	}
	for (int i = 0; i < nn; i++) {
//		std::cout << psum[i] << " ";
//		std::cout << res[i] << " ";
		if (res[i] != psum[i]) {
			std::cout << std::endl;
			std::cout << "Diff at " << i << ": " << res[i] << " vs " << psum[i] << std::endl;
			break;
		}
	}

//	return 0;

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

	const int masksCount = candidatesOffsets[PositionsCount - 1][CodesCount - 1][PiecesCount];
	assert(candidatesOffsetsH[0].x == 0);
	assert(candidatesOffsetsH[0].y > 0);
	for (int i = 1; i < PositionsCount; i++) {
		assert(candidatesOffsetsH[i].x >= candidatesOffsetsH[i - 1].y);
		assert(candidatesOffsetsH[i].y >= candidatesOffsetsH[i].x);
		assert(candidatesOffsetsH[i].y < masksCount);
	}

	thrust::device_vector<int2> candidatesOffsetsV(candidatesOffsetsH);
	globals.d_candidatesOffsets = raw(candidatesOffsetsV);

	thrust::device_vector<uint64_t> candidatesMaskV(candidatesMask, candidatesMask + masksCount);
	globals.d_candidatesMask = raw(candidatesMaskV);

	thrust::host_vector<int> nextValidPositionH(256);
	for (int i = 0; i < 256; i++) {
		nextValidPositionH[i] = InitNextValidPosition(i);
	}
	thrust::device_vector<int> nextValidPositionV(nextValidPositionH);
	globals.d_nextValidPosition = raw(nextValidPositionV);
	cudaMemcpyToSymbol(d_nextValidPosition, &nextValidPositionH[0], sizeof(int) * 256);

	thrust::device_vector<int> solutionsCountV(1);
	globals.d_solutionsCount = raw(solutionsCountV);

	thrust::device_vector<SituationT> solutionsV(MaxSolutions);
	globals.d_solutions = raw(solutionsV);

	thrust::device_vector<int> solversCountV(PiecesCount);
	globals.d_solversCount = raw(solversCountV);

	thrust::device_vector<int> errorV(2);
	globals.d_error = raw(errorV);

	Locals locals;

	thrust::device_vector<uint64_t> gridV(PiecesCount * MaxOutput); // actualPiece * MaxOutput + threadIdx
	locals.d_grid = raw(gridV);

	thrust::device_vector<int> positionV(PiecesCount * MaxOutput); // actualPiece * MaxOutput + threadIdx
	locals.d_position = raw(positionV);

// permutationOrder up to actualPiece, situation afterwards
	thrust::device_vector<int> situationV(PiecesCount * PiecesCount * MaxOutput); // (actualPiece * PiecesCount + piece) * MaxOutput + threadIdx
	locals.d_situation = raw(situationV);

	solversCountV[0] = 1;
	for (int i = 1; i < PositionsCount; i++)
		situationV[i * MaxOutput] = i;

	thrust::host_vector<int> solversCountH(solversCountV);

	timerInit.Record("Init");
	TimerC timerLoop;

	int steps = 0;
	const int cudaSteps = 500;
	int error = 0;
	while (error == 0) {
		if (HostLoop<DoDebug>(globals, locals, cudaSteps, solversCountV, solversCountH) < 0)
			break;
//		CudaLoop<DoDebug> <<<1, 1>>>(globals, locals, cudaSteps);
//		cudaDeviceSynchronize();
//		checkCudaErrors(cudaGetLastError());

		int sc = solutionsCountV[0];
		std::cout << steps << " [" << (DoDebug ? "D" : "R") << "]: solutions = " << sc;
		timerLoop.Record("");
		if (sc > 300)
			break;

		error = errorV[0];
		steps += cudaSteps;
	};

	if (error > 0) {
		int errorNum = 0;
		while (error > 1) {
			error /= 2;
			errorNum++;
		}
		std::cout << "\n\nERROR! " << errorNum << " (" << errorV[1] << ")\n" << std::endl;
	}

	const int solutionsCount = solutionsCountV[0];
	thrust::copy(solutionsV.begin(), solutionsV.begin() + solutionsCount, solutions);

	timerInit.Record("\nTotal");
	std::cout << "Solutions found: " << solutionsCountV[0] << std::endl;
	return solutionsCount;
}
