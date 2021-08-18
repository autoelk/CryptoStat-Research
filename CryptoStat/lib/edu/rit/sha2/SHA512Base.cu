//******************************************************************************
//
// File:    SHA512Base.cu
// Unit:    SHA-384 and SHA-512 CUDA functions
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

#ifndef __SHA512BASE_CU__
#define __SHA512BASE_CU__

#include <stdint.h>
#include "../util/BigInt.cu"

/**
 * The CUDA implementation of the common code for the SHA-384 and SHA-512 hash
 * functions. The <I>A</I> input is the first part of the message (512 bits).
 * The <I>B</I> input is the second part of the message (320 bits). The <I>C</I>
 * output is the digest (224, 256, 384, or 512 bits).
 *
 * @author  Alan Kaminsky
 * @version 15-Sep-2017
 */

// Round constants.
__constant__ uint64_t K [80] =
	{
	0x428a2f98d728ae22L, 0x7137449123ef65cdL,
	0xb5c0fbcfec4d3b2fL, 0xe9b5dba58189dbbcL,
	0x3956c25bf348b538L, 0x59f111f1b605d019L,
	0x923f82a4af194f9bL, 0xab1c5ed5da6d8118L,
	0xd807aa98a3030242L, 0x12835b0145706fbeL,
	0x243185be4ee4b28cL, 0x550c7dc3d5ffb4e2L,
	0x72be5d74f27b896fL, 0x80deb1fe3b1696b1L,
	0x9bdc06a725c71235L, 0xc19bf174cf692694L,
	0xe49b69c19ef14ad2L, 0xefbe4786384f25e3L,
	0x0fc19dc68b8cd5b5L, 0x240ca1cc77ac9c65L,
	0x2de92c6f592b0275L, 0x4a7484aa6ea6e483L,
	0x5cb0a9dcbd41fbd4L, 0x76f988da831153b5L,
	0x983e5152ee66dfabL, 0xa831c66d2db43210L,
	0xb00327c898fb213fL, 0xbf597fc7beef0ee4L,
	0xc6e00bf33da88fc2L, 0xd5a79147930aa725L,
	0x06ca6351e003826fL, 0x142929670a0e6e70L,
	0x27b70a8546d22ffcL, 0x2e1b21385c26c926L,
	0x4d2c6dfc5ac42aedL, 0x53380d139d95b3dfL,
	0x650a73548baf63deL, 0x766a0abb3c77b2a8L,
	0x81c2c92e47edaee6L, 0x92722c851482353bL,
	0xa2bfe8a14cf10364L, 0xa81a664bbc423001L,
	0xc24b8b70d0f89791L, 0xc76c51a30654be30L,
	0xd192e819d6ef5218L, 0xd69906245565a910L,
	0xf40e35855771202aL, 0x106aa07032bbd1b8L,
	0x19a4c116b8d2d0c8L, 0x1e376c085141ab53L,
	0x2748774cdf8eeb99L, 0x34b0bcb5e19b48a8L,
	0x391c0cb3c5c95a63L, 0x4ed8aa4ae3418acbL,
	0x5b9cca4f7763e373L, 0x682e6ff3d6b2b8a3L,
	0x748f82ee5defb2fcL, 0x78a5636f43172f60L,
	0x84c87814a1f0ab72L, 0x8cc702081a6439ecL,
	0x90befffa23631e28L, 0xa4506cebde82bde9L,
	0xbef9a3f7b2c67915L, 0xc67178f2e372532bL,
	0xca273eceea26619cL, 0xd186b8c721c0c207L,
	0xeada7dd6cde0eb1eL, 0xf57d4f7fee6ed178L,
	0x06f067aa72176fbaL, 0x0a637dc5a2c898a6L,
	0x113f9804bef90daeL, 0x1b710b35131c471bL,
	0x28db77f523047d84L, 0x32caab7b40c72493L,
	0x3c9ebe0a15c9bebcL, 0x431d67c49c100d4cL,
	0x4cc5d4becb3e42b6L, 0x597f299cfc657e2aL,
	0x5fcb6fab3ad6faecL, 0x6c44198c4a475817L,
	};

/**
 * Right-rotate functions.
 */
__device__ uint64_t ROTR_1 (uint64_t x)
	{
	return (x >> 1) | (x << 63);
	}

__device__ uint64_t ROTR_8 (uint64_t x)
	{
	return (x >> 8) | (x << 56);
	}

__device__ uint64_t ROTR_14 (uint64_t x)
	{
	return (x >> 14) | (x << 50);
	}

__device__ uint64_t ROTR_18 (uint64_t x)
	{
	return (x >> 18) | (x << 46);
	}

__device__ uint64_t ROTR_19 (uint64_t x)
	{
	return (x >> 19) | (x << 45);
	}

__device__ uint64_t ROTR_28 (uint64_t x)
	{
	return (x >> 28) | (x << 36);
	}

__device__ uint64_t ROTR_34 (uint64_t x)
	{
	return (x >> 34) | (x << 30);
	}

__device__ uint64_t ROTR_39 (uint64_t x)
	{
	return (x >> 39) | (x << 25);
	}

__device__ uint64_t ROTR_41 (uint64_t x)
	{
	return (x >> 41) | (x << 23);
	}

__device__ uint64_t ROTR_61 (uint64_t x)
	{
	return (x >> 61) | (x << 3);
	}

/**
 * The little functions.
 */
__device__ uint64_t Ch (uint64_t x, uint64_t y, uint64_t z)
	{
	return (x & y) ^ (~x & z);
	}

__device__ uint64_t Maj (uint64_t x, uint64_t y, uint64_t z)
	{
	return (x & y) ^ (x & z) ^ (y & z);
	}

__device__ uint64_t Sigma_0 (uint64_t x)
	{
	return ROTR_28(x) ^ ROTR_34(x) ^ ROTR_39 (x);
	}

__device__ uint64_t Sigma_1 (uint64_t x)
	{
	return ROTR_14(x) ^ ROTR_18(x) ^ ROTR_41 (x);
	}

__device__ uint64_t sigma_0 (uint64_t x)
	{
	return ROTR_1(x) ^ ROTR_8(x) ^ (x >> 7);
	}

__device__ uint64_t sigma_1 (uint64_t x)
	{
	return ROTR_19(x) ^ ROTR_61(x) ^ (x >> 6);
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
	uint64_t W [16];
	uint64_t ha, hb, hc, hd, he, hf, hg, hh, tmp1, tmp2;
	int s, t;

	// Store message block in message schedule W.
	biUnpackLongBigEndian (Asize, A, W, 0);
	biUnpackLongBigEndian (Bsize, B, W, 8);
	W[13] = 0x8000000000000000L;
	W[14] = 0L;
	W[15] = 832L;

	// Initialize working variables.
	ha = IV_0;
	hb = IV_1;
	hc = IV_2;
	hd = IV_3;
	he = IV_4;
	hf = IV_5;
	hg = IV_6;
	hh = IV_7;

	// Do 80 rounds.
	for (t = 0; t <= 79; ++ t)
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
