//******************************************************************************
//
// File:    SHA224.cu
// Unit:    SHA-224 CUDA functions
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

/**
 * The CUDA implementation of the SHA-224 hash function. The <I>A</I> input is
 * the first part of the message (256 bits). The <I>B</I> input is the second
 * part of the message (160 bits). The <I>C</I> output is the digest (224 bits).
 *
 * @author  Alan Kaminsky
 * @version 15-Sep-2017
 */

// Initialization vector.
#define IV_0 0xc1059ed8
#define IV_1 0x367cd507
#define IV_2 0x3070dd17
#define IV_3 0xf70e5939
#define IV_4 0xffc00b31
#define IV_5 0x68581511
#define IV_6 0x64f98fa7
#define IV_7 0xbefa4fa4

/**
 * Record digest for the current round. Working variables ha through hh are
 * stored in the proper words of the digest (dig).
 */
__device__ void recordDigest
	(uint32_t* dig,
	 uint32_t ha,
	 uint32_t hb,
	 uint32_t hc,
	 uint32_t hd,
	 uint32_t he,
	 uint32_t hf,
	 uint32_t hg,
	 uint32_t hh)
	{
	dig[6] = ha + IV_0;
	dig[5] = hb + IV_1;
	dig[4] = hc + IV_2;
	dig[3] = hd + IV_3;
	dig[2] = he + IV_4;
	dig[1] = hf + IV_5;
	dig[0] = hg + IV_6;
	}

#include "SHA256Base.cu"
