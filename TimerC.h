/*
 * TimerC.h
 *
 *  Created on: Apr 30, 2014
 *      Author: igor
 */

#ifndef TIMERC_H_
#define TIMERC_H_

#include <iostream>

class TimerC {
public:
	TimerC() {
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);
	}

	~TimerC() {
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

	void Record(std::string text) {
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);

		float elapsedTime;
		cudaEventElapsedTime(&elapsedTime, start, stop);

		std::cout << text << ": " << elapsedTime / 1000 << " s" << std::endl;
	}
private:
	cudaEvent_t start, stop;
};



#endif /* TIMERC_H_ */
