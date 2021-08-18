//******************************************************************************
//
// File:    Direct.cu
// Unit:    Statistical Test CUDA functions
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

#include <stdint.h>
#include "TestState.cu"

/**
 * The CUDA implementation of a statistical test object that tests the direct
 * sequence of C outputs.
 *
 * @author  Alan Kaminsky
 * @version 19-Feb-2018
 */

// Accumulate the given sample of a bit group into the given test state.
__device__ void accumulate
	(testState_t* ts,  // Test state object
	 int index,        // Index of sample in sequence under test
	 int sample)       // Sample of bit group
	{
	tsAccumulate (ts, sample);
	}

#include "TestKernel.cu"
