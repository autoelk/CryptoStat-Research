//******************************************************************************
//
// File:    TestState.cu
// Unit:    Statistical Test State CUDA functions
//
// This CUDA source file is copyright (C) 2018 by Alan Kaminsky. All rights
// reserved. For further information, contact the author, Alan Kaminsky, at
// ark@cs.rit.edu.
//
// This CUDA source file is part of the CryptoStat Library ("CryptoStat").
// CryptoStat is free software; you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free Software
// Foundation; either version 3 of the License, or (at your option) any later
// version.
//
// CryptoStat is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
// A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// A copy of the GNU General Public License is provided in the file gpl.txt. You
// may also obtain a copy of the GNU General Public License on the World Wide
// Web at http://www.gnu.org/licenses/gpl.html.
//
//******************************************************************************

#ifndef __TEST_STATE_CU__
#define __TEST_STATE_CU__

#include "../util/BitSet.cu"

/**
 * Structures and functions for performing statistical tests, including the run
 * test and the birthday test.
 *
 * @author  Alan Kaminsky
 * @version 21-Feb-2018
 */

// Table of run test bin probabilities, indexed by bit group size (1..8) and
// comparison pattern (0..7).
__constant__ double runTestBinPr [9] [8] =
	{
	// 0-bit groups (not used)
		{
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
		},
	// 1-bit groups
		{
		0.0/16.0,
		0.0/16.0,
		1.0/16.0,
		3.0/16.0,
		0.0/16.0,
		4.0/16.0,
		3.0/16.0,
		5.0/16.0,
		},
	// 2-bit groups
		{
		1.0/256.0,
		15.0/256.0,
		35.0/256.0,
		45.0/256.0,
		15.0/256.0,
		65.0/256.0,
		45.0/256.0,
		35.0/256.0,
		},
	// 3-bit groups
		{
		70.0/4096.0,
		378.0/4096.0,
		714.0/4096.0,
		630.0/4096.0,
		378.0/4096.0,
		966.0/4096.0,
		630.0/4096.0,
		330.0/4096.0,
		},
	// 4-bit groups
		{
		1820.0/65536.0,
		7140.0/65536.0,
		12580.0/65536.0,
		9180.0/65536.0,
		7140.0/65536.0,
		14620.0/65536.0,
		9180.0/65536.0,
		3876.0/65536.0,
		},
	// 5-bit groups
		{
		35960.0/1048576.0,
		122760.0/1048576.0,
		210056.0/1048576.0,
		139128.0/1048576.0,
		122760.0/1048576.0,
		226424.0/1048576.0,
		139128.0/1048576.0,
		52360.0/1048576.0,
		},
	// 6-bit groups
		{
		635376.0/16777216.0,
		2031120.0/16777216.0,
		3428880.0/16777216.0,
		2162160.0/16777216.0,
		2031120.0/16777216.0,
		3559920.0/16777216.0,
		2162160.0/16777216.0,
		766480.0/16777216.0,
		},
	// 7-bit groups
		{
		10668000.0/268435456.0,
		33028128.0/268435456.0,
		55396384.0/268435456.0,
		34076640.0/268435456.0,
		33028128.0/268435456.0,
		56444896.0/268435456.0,
		34076640.0/268435456.0,
		11716640.0/268435456.0,
		},
	// 8-bit groups
		{
		174792640.0/4294967296.0,
		532668480.0/4294967296.0,
		890576960.0/4294967296.0,
		541056960.0/4294967296.0,
		532668480.0/4294967296.0,
		898965440.0/4294967296.0,
		541056960.0/4294967296.0,
		183181376.0/4294967296.0,
		}
	};

// Returns log(x!).
__device__ double logFactorial
	(int x)
	{
	return lgamma (x + 1.0);
	}

// Returns the log Bayes factor for a binomial model.
__device__ double binomialLBF
	(int n,     // Number of trials
	 int k,     // Number of successes
	 double p)  // Success probability
	{
	return logFactorial(n + 1) - logFactorial(k) - logFactorial(n - k) +
		k*log(p) + (n - k)*log1p(-p);
	}

// Returns the log Bayes factor for the birthday test.
__device__ double birthdayLBF
	(int n,           // Number of blocks
	 int k,           // Number of no-collision blocks
	 double log_p,    // No-collision probability
	 double log_1mp)  // Collision probability
	{
	return logFactorial(n + 1) - logFactorial(k) - logFactorial(n - k) +
		k*log_p + (n - k)*log_1mp;
	}

// Returns the log Bayes factor for a discrete distribution.
__device__ double discreteLBF
	(int nbins,       // Number of bins
	 int total,       // Total of bin counts
	 int* actual,     // Actual bin counts
	 double* expect)  // Expected bin probabilities
	{
	int i, cumulActual, k;
	double cumulExpect, dif, maxdif, p;

	// Find the bin whose cumulative distribution is farthest from the expected
	// cumulative distibution.
	cumulActual = 0;
	cumulExpect = 0.0;
	maxdif = -1.0;
	k = 0;
	p = 0.0;
	for (i = 0; i < nbins; ++ i)
		{
		cumulActual += actual[i];
		cumulExpect += expect[i];
		dif = abs((double)cumulActual/(double)total - cumulExpect);
		if (dif > maxdif)
			{
			maxdif = dif;
			k = cumulActual;
			p = cumulExpect;
			}
		}

	// Compute log Bayes factor.
	return (p == 0.0) ? 0.0 : binomialLBF (total, k, p);
	}

// Test state structure.
typedef struct
	{
	int diff;         // Used by difference sequence and avalanche sequence
	                  // Used by run test:
	int prev;         //   Bit group value in previous sample
	int runlen;       //   Run length
	int runbin;       //   Bin index
	int run_k[8];     //   Bin counts
	int run_n;        //   Total count
	double* run_exp;  //   Expected bin probabilities*
	                  // Used by birthday test:
	int blkmax;       //   Maximum block length*
	int blklen;       //   Block length
	int coll;         //   True if collision in block, false if no collision
	bitSet_t seen;    //   For detecting collisions
	int noColl_k;     //   Bin count
	int noColl_n;     //   Total of bin counts
	double log_p;     //   log pr(no collision)*
	double log_1mp;   //   log pr(collision)*
	                  // *Nonvarying; initialized based on bit group size
	}
	testState_t;

// Test result structure.
typedef struct
	{
	double runTestLBF;       // Run test log Bayes factor
	double birthdayTestLBF;  // Birthday test log Bayes factor
	double aggLBF;           // Aggregate log Bayes factor
	}
	testResult_t;

// Initialize the given test state structure for the given bit group size.
// The nonvarying fields are initialized.
__device__ void tsInit
	(testState_t* ts,
	 int G)           // Bit group size, 1 .. 8
	{
	int m = 1 << G;
	ts->run_exp = runTestBinPr[G];
	ts->blkmax = m;
	ts->log_p = logFactorial(m) - m*log((double)m);
	ts->log_1mp = log(1.0 - exp(ts->log_p));
	}

// Clear the given test state structure. The varying fields are cleared.
__device__ void tsClear
	(testState_t* ts)
	{
	ts->diff =
	ts->prev =
	ts->runlen =
	ts->runbin =
	ts->run_k[0] =
	ts->run_k[1] =
	ts->run_k[2] =
	ts->run_k[3] =
	ts->run_k[4] =
	ts->run_k[5] =
	ts->run_k[6] =
	ts->run_k[7] =
	ts->run_n =
	ts->blklen =
	ts->coll =
	ts->noColl_k =
	ts->noColl_n = 0;

	bsClear (&ts->seen);
	}

// Accumulate the given bit group sample into the given test state structure.
__device__ void tsAccumulate
	(testState_t* ts,
	 int sample)
	{
	// Accumulate for run test.
	if (ts->runlen == 0)
		{
		++ ts->runlen;
		}
	else if (ts->runlen == 1)
		{
		if (ts->prev >= sample)
			ts->runbin |= 4;
		++ ts->runlen;
		}
	else if (ts->runlen == 2)
		{
		if (ts->prev >= sample)
			ts->runbin |= 2;
		++ ts->runlen;
		}
	else if (ts->runlen == 3)
		{
		if (ts->prev >= sample)
			ts->runbin |= 1;
		++ ts->run_k[ts->runbin];
		++ ts->run_n;
		ts->runlen = 0;
		ts->runbin = 0;
		}
	ts->prev = sample;

	// Accumulate for birthday test.
	if (bsContains (&ts->seen, sample))
		ts->coll = 1;
	else
		bsAdd (&ts->seen, sample);
	++ ts->blklen;
	if (ts->blklen == ts->blkmax)
		{
		if (! ts->coll) ++ ts->noColl_k;
		++ ts->noColl_n;
		bsClear (&ts->seen);
		ts->blklen = 0;
		ts->coll = 0;
		}
	}

// Compute the log Bayes factors for the statistical test data in the given test
// state structure, and store the LBFs in the given test result structure.
__device__ void tsComputeResults
	(testState_t* ts,
	 testResult_t* tr)
	{
	// Compute LBF for run test.
	tr->runTestLBF =
		discreteLBF (8, ts->run_n, ts->run_k, ts->run_exp);

	// Compute LBF for birthday test.
	tr->birthdayTestLBF =
		birthdayLBF (ts->noColl_n, ts->noColl_k, ts->log_p, ts->log_1mp);

	// Compute aggregate LBF.
	tr->aggLBF = tr->runTestLBF + tr->birthdayTestLBF;
	}

#endif
