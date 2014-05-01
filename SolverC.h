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

#include <iostream>

int CudaSolve(CandidatesOffsets candidatesOffsets, CandidatesMask candidatesMask, SituationT (&solutions)[MaxSolutions]);

class SolverC {
public:

    std::list<Solution> solutions;

    SolverC(const std::shared_ptr<const Board>& board) {
        auto c = CandidatesCalculator(board).GetCandidates();

        auto gridSize(board->GetSize());
        Coord pos(Ints(gridSize.size()));

        std::shared_ptr<CandidatesT::Type> candidatesPerPiece = std::make_shared<CandidatesT::Type>();
        Enumerate(pos, gridSize, [&] {
        	candidatesPerPiece->push_back((*c)[pos]);
            return true;
        });

        candidatesT = std::make_shared<CandidatesT>(candidatesPerPiece);
	}

    void Solve() {
        SituationT ss[MaxSolutions];
        const int n = CudaSolve(candidatesT->candidatesOffsets, candidatesT->candidatesMask, ss);
    	for (int i = 0; i < n; i++)
    		solutions.push_back(candidatesT->Convert(ss[i]));
    }

private:
    std::shared_ptr<const CandidatesT> candidatesT;
};

#endif /* SOLVERC_H_ */
