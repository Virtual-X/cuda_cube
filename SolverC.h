/*
 * SolverC.h
 *
 *  Created on: Apr 28, 2014
 *      Author: igor
 */

#ifndef SOLVERC_H_
#define SOLVERC_H_

#include "Solution.h"
#include "CandidatesCalculator.h"
#include "CandidatesT.h"

void CudaSolve(CandidatesOffsets candidatesOffsets, CandidatesMask candidatesMask, std::list<SituationT>& solutions);

class SolverC {
public:

    std::list<Solution> solutions;

    SolverC(const std::shared_ptr<const Board>& board) {
        auto c = CandidatesCalculator(board).GetCandidates();

        auto gridSize(board->GetSize());
        Coord pos(Ints(gridSize.size()));

        std::vector<std::vector<FixedPieces>> candidatesPerPiece;
        Enumerate(pos, gridSize, [&] {
        	candidatesPerPiece.push_back((*c)[pos]);
            return true;
        });

        candidatesT = std::make_shared<CandidatesT>(candidatesPerPiece);
	}

    void Solve() {
        std::list<SituationT> solutions;
        CudaSolve(candidatesT->candidatesOffsets, candidatesT->candidatesMask, solutions);
        candidatesT->Convert(solutions, this->solutions);
    }

private:
    std::shared_ptr<const CandidatesT> candidatesT;
};

#endif /* SOLVERC_H_ */
