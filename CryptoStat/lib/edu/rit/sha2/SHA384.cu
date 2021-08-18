//******************************************************************************
//
// File:    SHA384.cu
// Unit:    SHA-384 CUDA functions
//
// This CUDA source file is copyright (C) 2017 by Alan Kaminsky. All rights
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
#include "../util/BigInt.cu"

/**
 * The CUDA implementation of the SHA-384 hash function. The <I>A</I> input is
 * the first part of the message (512 bits). The <I>B</I> input is the second
 * part of the message (320 bits). The <I>C</I> output is the digest (384 bits).
 *
 * @author  Alan Kaminsky
 * @version 15-Sep-2017
 */

// Initialization vector.
#define IV_0 0xcbbb9d5dc1059ed8
#define IV_1 0x629a292a367cd507
#define IV_2 0x9159015a3070dd17
#define IV_3 0x152fecd8f70e5939
#define IV_4 0x67332667ffc00b31
#define IV_5 0x8eb44a8768581511
#define IV_6 0xdb0c2e0d64f98fa7
#define IV_7 0x47b5481dbefa4fa4

/**
 * Record digest for the current round. Working variables ha through hh are
 * stored in the proper words of the digest (dig).
 */
__device__ void recordDigest
	(uint32_t* dig,
	 uint64_t ha,
	 uint64_t hb,
	 uint64_t hc,
	 uint64_t hd,
	 uint64_t he,
	 uint64_t hf,
	 uint64_t hg,
	 uint64_t hh)
	{
	uint64_t tmp;
	tmp = ha + IV_0;
	dig[11] = (uint32_t)(tmp >> 32);
	dig[10] = (uint32_t)(tmp);
	tmp = hb + IV_1;
	dig[ 9] = (uint32_t)(tmp >> 32);
	dig[ 8] = (uint32_t)(tmp);
	tmp = hc + IV_2;
	dig[ 7] = (uint32_t)(tmp >> 32);
	dig[ 6] = (uint32_t)(tmp);
	tmp = hd + IV_3;
	dig[ 5] = (uint32_t)(tmp >> 32);
	dig[ 4] = (uint32_t)(tmp);
	tmp = he + IV_4;
	dig[ 3] = (uint32_t)(tmp >> 32);
	dig[ 2] = (uint32_t)(tmp);
	tmp = hf + IV_5;
	dig[ 1] = (uint32_t)(tmp >> 32);
	dig[ 0] = (uint32_t)(tmp);
	}

#include "SHA512Base.cu"
