//******************************************************************************
//
// File:    SHA512_224.cu
// Unit:    SHA-512/224 CUDA functions
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
 * The CUDA implementation of the SHA-512/224 hash function. The <I>A</I> input
 * is the first part of the message (512 bits). The <I>B</I> input is the second
 * part of the message (320 bits). The <I>C</I> output is the digest (224 bits).
 *
 * @author  Alan Kaminsky
 * @version 15-Sep-2017
 */

// Initialization vector.
#define IV_0 0x8C3D37C819544DA2L
#define IV_1 0x73E1996689DCD4D6L
#define IV_2 0x1DFAB7AE32FF9C82L
#define IV_3 0x679DD514582F9FCFL
#define IV_4 0x0F6D2B697BD44DA8L
#define IV_5 0x77E36F7304C48942L
#define IV_6 0x3F9D85A86A1D36C8L
#define IV_7 0x1112E6AD91D692A1L

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
	dig[6] = (uint32_t)(tmp >> 32);
	dig[5] = (uint32_t)(tmp);
	tmp = hb + IV_1;
	dig[4] = (uint32_t)(tmp >> 32);
	dig[3] = (uint32_t)(tmp);
	tmp = hc + IV_2;
	dig[2] = (uint32_t)(tmp >> 32);
	dig[1] = (uint32_t)(tmp);
	tmp = hd + IV_3;
	dig[0] = (uint32_t)(tmp >> 32);
	}

#include "SHA512Base.cu"
