//******************************************************************************
//
// File:    SHA1.cu
// Unit:    SHA1 CUDA functions
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
 * The CUDA implementation of the SHA-1 hash function. The <I>A</I> input is the
 * first part of the message (256 bits). The <I>B</I> input is the second part
 * of the message (160 bits). The <I>C</I> output is the digest (160 bits).
 *
 * @author  Alan Kaminsky
 * @version 15-Sep-2017
 */

// Initialization vector.
#define IV_0 0x67452301
#define IV_1 0xefcdab89
#define IV_2 0x98badcfe
#define IV_3 0x10325476
#define IV_4 0xc3d2e1f0

/**
 * Returns leftRotate (x, 1).
 */
__device__ uint32_t ROTL_1
	(uint32_t x)
	{
	return (x << 1) | (x >> 31);
	}

/**
 * Returns leftRotate (x, 5).
 */
__device__ uint32_t ROTL_5
	(uint32_t x)
	{
	return (x << 5) | (x >> 27);
	}

/**
 * Returns leftRotate (x, 30).
 */
__device__ uint32_t ROTL_30
	(uint32_t x)
	{
	return (x << 30) | (x >> 2);
	}

/**
 * Evaluate the cryptographic function for the given combination of inputs. On
 * return, the function's output for input <TT>A</TT>, input <TT>B</TT>, and
 * each round <TT>r</TT> has been stored in <TT>C[a][b][r]</TT>.
 *
 * @param  NA     Number of <I>A</I> inputs.
 * @param  Asize  <I>A</I> input bigint size in words.
 * @param  A      <I>A</I> input value.
 * @param  NB     Number of <I>B</I> inputs.
 * @param  Bsize  <I>B</I> input bigint size in words.
 * @param  B      <I>B</I> input value.
 * @param  R      Number of rounds.
 * @param  Csize  <I>C</I> output bigint size in words.
 * @param  C      3-D array of <I>C</I> output values.
 * @param  a      <I>A</I> input index.
 * @param  b      <I>B</I> input index.
 */
__device__ void evaluate
	(int NA,
	 int Asize,
	 uint32_t* A,
	 int NB,
	 int Bsize,
	 uint32_t* B,
	 int R,
	 int Csize,
	 uint32_t* C,
	 int a,
	 int b)
	{
	uint32_t W [16];
	uint32_t ha, hb, hc, hd, he, f_t, K_t, tmp;
	int s, t;
	uint32_t* C_a_b_t;

	// Store message block in message schedule W.
	biUnpackBigEndian (Asize, A, W, 0);
	biUnpackBigEndian (Bsize, B, W, 8);
	W[13] = 0x80000000;
	W[14] = 0;
	W[15] = 416;

	// Initialize working variables.
	ha = IV_0;
	hb = IV_1;
	hc = IV_2;
	hd = IV_3;
	he = IV_4;

	// Do 80 rounds.
	for (t = 0; t <= 79; ++ t)
		{
		s = t & 15;

		// Compute round function.
		if (t >= 16)
			W[s] = ROTL_1 (W[(s+13)&15] ^ W[(s+8)&15] ^ W[(s+2)&15] ^ W[s]);
		if (t <= 19)
			{
			f_t = (hb & hc) ^ (~hb & hd);
			K_t = 0x5a827999;
			}
		else if (t <= 39)
			{
			f_t = hb ^ hc ^ hd;
			K_t = 0x6ed9eba1;
			}
		else if (t <= 59)
			{
			f_t = (hb & hc) ^ (hb & hd) ^ (hc & hd);
			K_t = 0x8f1bbcdc;
			}
		else
			{
			f_t = hb ^ hc ^ hd;
			K_t = 0xca62c1d6;
			}
		tmp = ROTL_5 (ha) + f_t + he + K_t + W[s];
		he = hd;
		hd = hc;
		hc = ROTL_30 (hb);
		hb = ha;
		ha = tmp;

		// Record digest for this round.
		C_a_b_t = biElem3D (Csize, C, NB, R, a, b, t);
		C_a_b_t[4] = ha + IV_0;
		C_a_b_t[3] = hb + IV_1;
		C_a_b_t[2] = hc + IV_2;
		C_a_b_t[1] = hd + IV_3;
		C_a_b_t[0] = he + IV_4;
		}
	}

#include "../crst/FunctionKernel.cu"
