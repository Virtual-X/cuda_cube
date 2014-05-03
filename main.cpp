
#include "Timer.h"
#include "BoardLoader.cpp"

#include "SolverC.h"

#include <iostream>

static const std::string boardName = "Tetris4.txt";

void SolveTetrisCube() {
    std::shared_ptr<Board> board;
    try {
        board = BoardLoader::LoadFile(boardName);
    }
    catch (std::string& error) {
//        std::cout << "Error loading Tetris4.txt: " << error;
//        std::cout << "\nLoading default problem set..." << std::endl;
        board = BoardLoader::LoadDefault();
    }

    Timer t;
    t.Start();
    SolverC solver(board);
    t.Lap("init");

    solver.Solve();
    std::cout << std::endl << "Total solutions found: " << (int)solver.solutions.size() << std::endl;
    t.Lap("completion");
}

int main() {
    SolveTetrisCube();
}
