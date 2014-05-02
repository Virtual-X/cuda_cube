#include "Constants.h"
#include "TimerC.h"
#include "utils.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

enum {
	MaxInput = 64 * 1024,
	MaxOutput = 2 * 1024 * 1024
};

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

enum ErrorCode {
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
	ErrorCodeInvalidLastStepFlagC = 1 << 18
};

struct Globals {
	const int2* d_candidatesOffsets; // [PositionsCount][CodesCount][PiecesCount], d_candidatesOffsets[GetCandidatesOffsetIndex(position, code, piece)]
	CandidatesMask d_candidatesMask;
	const int* d_nextValidPosition; // [256];

     SituationT* d_solutions; // should be int[PiecesCount][MaxSolutions] to allow coalesced writing, although very very seldom
	int* d_solutionsCount;

	int* d_solversCount; // [PiecesCount]

     int* d_error; // [2]
};

struct Locals {
	uint64_t* d_grid;
	int* d_position;
	int* d_permutationOrder;
	int* d_situation;
};

int GetActualPiece(const thrust::host_vector<int>& solversCountH) {
	for (int i = solversCountH.size(); i > 0; i--) {
		if (solversCountH[i - 1] >= MaxInput)
			return i - 1;
	}
	for (int i = 0; i < (int)solversCountH.size(); i++) {
		if (solversCountH[i] > 0)
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
  BlockSize = 512
};

int shared_psum(int (&_n)[BlockSize]) {
  __syncthreads();

  // do

  __syncthreads();

  return last + n[las];
}

template<bool IsLastStep, bool Debug>
__device__
void InitStep(const Globals globals, const Locals locals, const int actualPiece, const int ind,
  int (&_id2)[BlockSize], int (&_pos)[BlockSize], int (&_grid)[BlockSize],
  const int* (&situation), const int2* (&candidatesOffsets)) {
{
  const int id = actualPiece * MaxOutput + ind;
  if (Debug && CheckIndex(id, PiecesCount * MaxOutput, globals, ErrorCodeInvalidId))
    return;

  const int id2 = actualPiece * PiecesCount * MaxOutput + ind;
  if (Debug && CheckIndex(id2 + (PiecesCount - 1) * MaxOutput, PiecesCount * PiecesCount * MaxOutput,
    globals, ErrorCodeInvalidId2))
    return;

  const uint64_t grid = locals.d_grid[id];
  const int position = locals.d_position[id];

  const int pos = GetNextPos(grid, position, globals.d_nextValidPosition);
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

template<bool IsLastStep, bool Debug>
__device__
void GetBeginAndCount(const Globals globals, const int* situation, const int i,
  int (&_begin)[BlockSize], int (&_count)[BlockSize]) {
{
  const int piece = situation[i * MaxOutput];
  if (Debug && CheckIndex(piece, PiecesCount, globals, ErrorCodeInvalidPiece))
    return;

  int2 beginEnd = candidatesOffsets[piece];

  const int idx = threadIdx.x;
  _begin[idx] = beginEnd.x;
  _count[idx] = beginEnd.y - beginEnd.x;
}

template<bool IsLastStep, bool Debug>
__global__
void CudaStep(const Globals globals, const Locals locals, const int actualPiece, const int off, const int count) {
  __shared__ int _id2[BlockSize];
  __shared__ int _pos[BlockSize];
  __shared__ uint64_t _grid[BlockSize];

  __shared__ int _begin[BlockSize];
  __shared__ int _count[BlockSize];
  __shared__ int _temp[BlockSize + 1];

  __shared__ int __id2[BlockSize];
  __shared__ int __pos[BlockSize];
  __shared__ uint64_t __grid[BlockSize];

  const int idx = threadIdx.x;
  const int ind = (blockIdx.x * blockDim.x) + idx;

  const int* situation = locals.d_situation;
  const int2* candidatesOffsets = globals.d_candidatesOffsets;
  if (ind < count)
    InitStep<Debug>(globals, locals, actualPiece, off + ind, _id2, _pos, _grid, situation, candidatesOffsets);
  
  for (int i = actualPiece; i < PiecesCount; i++) {
    if (ind < count)
      GetBeginAndCount(globals, situation, i, _begin, _count);

    const int totCount = shared_psum(_count, count);

    // here we are changing the use of idx, work for others
    for (int j = 0; j < totCount; j += BlockSize) {
      const int k = j + idx;
      if (k >= totCount)
        break; // nothing better to do?

      const int px = get_index(k, _count); // the same for many
      const int index = _begin[px] + (k - _count[px]); // somewhat increasing
      uint64_t mask = globals.d_candidatesMask[index]; // somewhat coalesced
      const int grid = _grid[px];

      const bool valid = (mask & grid) == 0;
      _temp[idx] = valid ? 1 : 0;
      const int nvalid = shared_psum(_temp, min(totCount - j, BlockSize));

      if (idx == 0) {
        int* targetCount = IsLastStep ? globals.d_solutionsCount : globals.d_solversCount + (actualPiece + 1);
        _temp[BlockSize] = atomicAdd(targetCount, nvalid);
      }

      if (valid) {
        const int dest = _temp[idx];
        if (!IsLastStep) {
          __grid[dest] = mask | grid;
          __pos[dest] = _pos[px];
        }
        __id2[dest] = _id2[px];
      }

      if (idx >= nvalid)
        continue;

      __syncthreads(); // needs _temp[BlockSize], do it only if not continue

      const int* current = locals.d_situation + __id2[idx]; // somewhat increasing
      const int n = _temp[BlockSize] + idx; // coalesced!
                 
      if (IsLastStep) {
		    if (Debug && CheckIndex(actualPiece, PiecesCount, globals, ErrorCodeInvalidLastStepFlagA))
			   return;
		    if (Debug && CheckIndex(PiecesCount, actualPiece + 2, globals, ErrorCodeInvalidLastStepFlagB))
			   return;

		    if (Debug && CheckIndex(n, MaxSolutions, globals, ErrorCodeInvalidSolutionsCount))
			   return;

		    int* solution = globals.d_solutions[n].pieces;
		    for (int k = 0; k < actualPiece; k++) {
			   solution[k] = current[k * MaxOutput];
		    }
		    solution[actualPiece] = index; // last element of situation
	   } else {
		    if (Debug && CheckIndex(actualPiece, PiecesCount - 1, globals, ErrorCodeInvalidLastStepFlagC))
			   return;

		    if (Debug && CheckIndex(n, MaxOutput, globals, ErrorCodeInvalidSolversCount))
			   return;

		    const int o = (actualPiece + 1) * MaxOutput + n; // other id
		    if (Debug && CheckIndex(o, PiecesCount * MaxOutput,
				    globals, ErrorCodeInvalidO_NotLastStep))
			   return;

              locals.d_position[o] = __pos[idx];
		    locals.d_grid[o] = __grid[idx];

		    if (Debug && CheckIndex(actualPiece + 1, PiecesCount,
				    globals, ErrorCodeInvalidActualPiece))
			   return;

		    const int q = (actualPiece + 1) * PiecesCount * MaxOutput + n; // other id2
		    if (Debug && CheckIndex(q + (PiecesCount - 1) * MaxOutput, PiecesCount * PiecesCount * MaxOutput,
				    globals, ErrorCodeInvalidQ_NotLastStep))
			   return;

		    int* other = locals.d_situation + q;
		    for (int k = 0; k < PiecesCount; k++) {
			   other[k * MaxOutput] = current[k * MaxOutput];
		    }
		    other[i * MaxOutput] = current[actualPiece * MaxOutput]; // permutation order
		    other[actualPiece * MaxOutput] = index; // last element of situation

		    if (Debug) {
			   for (int k = actualPiece + 1; k < PiecesCount; k++) {
				    if (CheckIndex(other[k * MaxOutput], PiecesCount, globals, ErrorCodeInvalidX_NotLastStep))
					   return;
			   }
		    }
	   }
    }
}

int CudaSolve(CandidatesOffsets candidatesOffsets,	CandidatesMask candidatesMask, SituationT (&solutions)[MaxSolutions]) {
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

	thrust::device_vector<uint64_t> candidatesMaskV(candidatesMask,	candidatesMask + masksCount);
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

	int actualPiece = 0;
	solversCountV[0] = 1;
	for (int i = 1; i < PositionsCount; i++) {
		situationV[i * MaxOutput] = i;
	}

	thrust::host_vector<int> solversCountH(solversCountV);

	timerInit.Record("Init");
	TimerC timerLoop;

#if defined _DEBUG
	const bool DoDebug = true;
#else
	const bool DoDebug = false;
#endif

	int n = 0;
	do {
		int count = solversCountV[actualPiece];

		int sc = solutionsCountV[0];
		if (sc > 10000)
			break;

		bool disp = (n % 500) == 0;

		if (disp) {
//			std::cout << n << ": s[" << actualPiece << "] = " << count;
			std::cout << n << " [" << (DoDebug ? "D" : "R") << "]: solutions = " << sc;
		}

		const int off = count > MaxInput ? count - MaxInput : 0;
		count = min(count, MaxInput);
		const int blockSize = 512;
		const int gridSize = (count + blockSize - 1) / blockSize;
		if (actualPiece == PiecesCount - 1) {
			CudaStep<true, DoDebug> <<<gridSize, blockSize>>>(globals, locals, actualPiece, off, count);
		} else {
			CudaStep<false, DoDebug> <<<gridSize, blockSize>>>(globals, locals, actualPiece, off, count);
		}
		cudaDeviceSynchronize();
		checkCudaErrors(cudaGetLastError());

		int error = errorV[0];
		if (error) {
			int errorNum = 0;
			while (error > 1) {
				error /= 2;
				errorNum++;
			}
			std::cout << "\n\nERROR! " << errorNum << " (" << errorV[1] << ")\n" << std::endl;
			break;
		}

		solversCountH[actualPiece] = solversCountV[actualPiece];
		if (actualPiece < PiecesCount - 1) {
			solversCountH[actualPiece + 1] = solversCountV[actualPiece + 1];
		}

		actualPiece = GetActualPiece(solversCountH);

		if (disp)
			timerLoop.Record("");

		++n;
	} while (actualPiece >= 0);

	const int solutionsCount = solutionsCountV[0];
	thrust::copy(solutionsV.begin(), solutionsV.begin() + solutionsCount, solutions);

	timerInit.Record("\nTotal");
	std::cout << "Solutions found: " << solutionsCountV[0] << std::endl;
	return solutionsCount;
}
