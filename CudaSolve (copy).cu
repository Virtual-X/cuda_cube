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
	ErrorCodeInvalidLastStepFlagC = 1 << 18
};

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

template<bool IsLastStep, bool Debug>
__global__
void CudaStep(const Globals globals, const Locals locals, const int actualPiece, const int off, const int count) {
	const int ind = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (ind >= count)
		return;

	const int id = actualPiece * MaxOutput + off + ind;
	if (Debug && CheckIndex(id, PiecesCount * MaxOutput, globals, ErrorCodeInvalidId))
		return;

	const int id2 = actualPiece * PiecesCount * MaxOutput + off + ind;
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

	const int* current = locals.d_situation + id2;

	const int2* d_candidatesOffsets_ = globals.d_candidatesOffsets + candidatesOffsetIndex;
	for (int i = actualPiece; i < PiecesCount; i++) {
		const int piece = locals.d_situation[id2 + i * MaxOutput];
		if (Debug && CheckIndex(piece, PiecesCount, globals, ErrorCodeInvalidPiece))
			return;

		int n_ = 0;
		int index_[24];
		uint64_t mask_[24];

		int2 beginEnd = d_candidatesOffsets_[piece]; // not coalesced (at all)
		for (int index = beginEnd.x; index < beginEnd.y; index++) {
			uint64_t mask = globals.d_candidatesMask[index]; // to be coalesced
			if (mask & grid)
				continue;

			mask_[n_] = mask;
			index_[n_] = index;
			n_++;
		}
		if (!n_)
			continue;
		for (int j = 0; j < n_; j++) {
			const int index = index_[j];
			const uint64_t mask = mask_[j];

			if (IsLastStep) {
				if (Debug && CheckIndex(actualPiece, PiecesCount, globals, ErrorCodeInvalidLastStepFlagA))
					return;
				if (Debug && CheckIndex(PiecesCount, actualPiece + 2, globals, ErrorCodeInvalidLastStepFlagB))
					return;

				const int n = atomicAdd(globals.d_solutionsCount, 1);
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

				const int n = atomicAdd(globals.d_solversCount + (actualPiece + 1), 1);
				if (Debug && CheckIndex(n, MaxOutput, globals, ErrorCodeInvalidSolversCount))
					return;

				const int o = (actualPiece + 1) * MaxOutput + n;
				if (Debug && CheckIndex(o, PiecesCount * MaxOutput,
						globals, ErrorCodeInvalidO_NotLastStep))
					return;

				locals.d_position[o] = pos + 1;
				locals.d_grid[o] = mask | grid;

				if (Debug && CheckIndex(actualPiece + 1, PiecesCount,
						globals, ErrorCodeInvalidActualPiece))
					return;

				const int q = (actualPiece + 1) * PiecesCount * MaxOutput + n;
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

		const int blockSize = 512;
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

		const int blockSize = 512;
		const int gridSize = (count + blockSize - 1) / blockSize;

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

	solversCountV[0] = 1;
	for (int i = 1; i < PositionsCount; i++) {
		situationV[i * MaxOutput] = i;
	}

	thrust::host_vector<int> solversCountH(solversCountV);

	timerInit.Record("Init");
	TimerC timerLoop;

	int steps = 0;
	const int cudaSteps = 2000;
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
