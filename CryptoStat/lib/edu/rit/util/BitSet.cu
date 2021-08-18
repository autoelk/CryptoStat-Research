//******************************************************************************
//
// File:    BitSet.cu
// Unit:    Bit Set CUDA functions
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

#ifndef __BIT_SET_CU__
#define __BIT_SET_CU__

/**
 * Structures and functions for a bit set that can hold elements from 0 to 255.
 *
 * @author  Alan Kaminsky
 * @version 14-Feb-2018
 */

// Bit set type.
typedef struct
	{
	uint32_t bit [8];  // Array of 8*32 = 256 bits
	}
	bitSet_t;

// Clear the given bit set.
__device__ void bsClear
	(bitSet_t* set)
	{
	int i;
	for (i = 0; i < 8; ++ i)
		set->bit[i] = 0;
	}

// Test whether the given bit set contains the given value. Returns nonzero if
// so, zero if not.
__device__ uint32_t bsContains
	(bitSet_t* set,
	 int elem)
	{
	return set->bit[elem >> 5] & (1 << (elem & 31));
	}

// Add the given element to the given bit set.
__device__ void bsAdd
	(bitSet_t* set,
	 int elem)
	{
	set->bit[elem >> 5] |= (1 << (elem & 31));
	}

#endif
