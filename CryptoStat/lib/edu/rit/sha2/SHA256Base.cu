//******************************************************************************
//
// File:    SHA256Base.cu
// Unit:    SHA-224 and SHA-256 CUDA functions
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

#ifndef __SHA256BASE_CU__
#define __SHA256BASE_CU__

#include <stdint.h>
#include "../util/BigInt.cu"

/**
 * The CUDA implementation of the common code for the SHA-224 and SHA-256 hash
 * functions. The <I>A</I> input is the first part of the message (256 bits).
 * The <I>B</I> input is the second part of the message (160 bits). The <I>C</I>
 * output is the digest (224 or 256 bits).
 *
 * @author  Alan Kaminsky
 * @version 15-Sep-2017
 */

// Round constants.
__constant__ uint32_t K [64] =
	{
	0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 
	0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 
	0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 
	0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 
	0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 
	0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 
	0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 
	0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 
	0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 
	0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 
	0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 
	0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 
	0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 
	0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 
	0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 
	0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2, 
	};

/**
 * Right-rotate functions.
 */
__device__ uint32_t ROTR_2 (uint32_t x)
	{
	return (x >> 2) | (x << 30);
	}

__device__ uint32_t ROTR_6 (uint32_t x)
	{
	return (x >> 6) | (x << 26);
	}

__device__ uint32_t ROTR_7
	(uint32_t x)
	{
	return (x >> 7) | (x << 25);
	}

__device__ uint32_t ROTR_11 (uint32_t x)
	{
	return (x >> 11) | (x << 21);
	}

__device__ uint32_t ROTR_13 (uint32_t x)
	{
	return (x >> 13) | (x << 19);
	}

__device__ uint32_t ROTR_17 (uint32_t x)
	{
	return (x >> 17) | (x << 15);
	}

__device__ uint32_t ROTR_18 (uint32_t x)
	{
	return (x >> 18) | (x << 14);
	}

__device__ uint32_t ROTR_19 (uint32_t x)
	{
	return (x >> 19) | (x << 13);
	}

__device__ uint32_t ROTR_22 (uint32_t x)
	{
	return (x >> 22) | (x << 10);
	}

__device__ uint32_t ROTR_25 (uint32_t x)
	{
	return (x >> 25) | (x << 7);
	}

/**
 * The little functions.
 */
__device__ uint32_t Ch (uint32_t x, uint32_t y, uint32_t z)
	{
	return (x & y) ^ (~x & z);
	}

__device__ uint32_t Maj (uint32_t x, uint32_t y, uint32_t z)
	{
	return (x & y) ^ (x & z) ^ (y & z);
	}

__device__ uint32_t Sigma_0 (uint32_t x)
	{
	return ROTR_2(x) ^ ROTR_13(x) ^ ROTR_22 (x);
	}

__device__ uint32_t Sigma_1 (uint32_t x)
	{
	return ROTR_6(x) ^ ROTR_11(x) ^ ROTR_25 (x);
	}

__device__ uint32_t sigma_0 (uint32_t x)
	{
	return ROTR_7(x) ^ ROTR_18(x) ^ (x >> 3);
	}

__device__ uint32_t sigma_1 (uint32_t x)
	{
	return ROTR_17(x) ^ ROTR_19(x) ^ (x >> 10);
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
	uint32_t ha, hb, hc, hd, he, hf, hg, hh, tmp1, tmp2;
	int s, t;

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
	hf = IV_5;
	hg = IV_6;
	hh = IV_7;

	// Do 64 rounds.
	for (t = 0; t <= 63; ++ t)
		{
		s = t & 15;

		// Compute round function.
		if (t >= 16)
			W[s] = sigma_1 (W[(s+14)&15]) + W[(s+9)&15] +
				sigma_0 (W[(s+1)&15]) + W[s];
		tmp1 = hh + Sigma_1(he) + Ch(he,hf,hg) + K[t] + W[s];
		tmp2 = Sigma_0(ha) + Maj(ha,hb,hc);
		hh = hg;
		hg = hf;
		hf = he;
		he = hd + tmp1;
		hd = hc;
		hc = hb;
		hb = ha;
		ha = tmp1 + tmp2;

		// Record digest for this round.
		recordDigest (biElem3D (Csize, C, NB, R, a, b, t),
			ha, hb, hc, hd, he, hf, hg, hh);
		}
	}

#include "../crst/FunctionKernel.cu"

#endif
