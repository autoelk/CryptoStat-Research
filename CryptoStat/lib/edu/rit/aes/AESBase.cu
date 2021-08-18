//******************************************************************************
//
// File:    AESBase.cu
// Unit:    AES CUDA functions
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
 * The CUDA implementation of the common functionality for all key sizes of the
 * AES block cipher. The <I>A</I> input is the plaintext (128 bits). The
 * <I>B</I> input is the key (128, 192, or 256 bits). The <I>C</I> output is the
 * ciphertext (128 bits).
 *
 * @author  Alan Kaminsky
 * @version 09-Oct-2017
 */

/**
 * The AES S-box.
 */
__constant__ uint8_t sbox [256] =
	{
	/* 00 */ 0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5,
	/* 08 */ 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
	/* 10 */ 0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0,
	/* 18 */ 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
	/* 20 */ 0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc,
	/* 28 */ 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
	/* 30 */ 0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a,
	/* 38 */ 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
	/* 40 */ 0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0,
	/* 48 */ 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
	/* 50 */ 0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b,
	/* 58 */ 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
	/* 60 */ 0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85,
	/* 68 */ 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
	/* 70 */ 0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5,
	/* 78 */ 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
	/* 80 */ 0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17,
	/* 88 */ 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
	/* 90 */ 0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88,
	/* 98 */ 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
	/* a0 */ 0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c,
	/* a8 */ 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
	/* b0 */ 0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9,
	/* b8 */ 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
	/* c0 */ 0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6,
	/* c8 */ 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
	/* d0 */ 0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e,
	/* d8 */ 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
	/* e0 */ 0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94,
	/* e8 */ 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
	/* f0 */ 0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68,
	/* f8 */ 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16,
	};

/**
 * Table for computing the xtime function.
 */
__constant__ uint8_t xtime [256] =
	{
	/* 00 */ 0x00, 0x02, 0x04, 0x06, 0x08, 0x0A, 0x0C, 0x0E,
	/* 08 */ 0x10, 0x12, 0x14, 0x16, 0x18, 0x1A, 0x1C, 0x1E,
	/* 10 */ 0x20, 0x22, 0x24, 0x26, 0x28, 0x2A, 0x2C, 0x2E,
	/* 18 */ 0x30, 0x32, 0x34, 0x36, 0x38, 0x3A, 0x3C, 0x3E,
	/* 20 */ 0x40, 0x42, 0x44, 0x46, 0x48, 0x4A, 0x4C, 0x4E,
	/* 28 */ 0x50, 0x52, 0x54, 0x56, 0x58, 0x5A, 0x5C, 0x5E,
	/* 30 */ 0x60, 0x62, 0x64, 0x66, 0x68, 0x6A, 0x6C, 0x6E,
	/* 38 */ 0x70, 0x72, 0x74, 0x76, 0x78, 0x7A, 0x7C, 0x7E,
	/* 40 */ 0x80, 0x82, 0x84, 0x86, 0x88, 0x8A, 0x8C, 0x8E,
	/* 48 */ 0x90, 0x92, 0x94, 0x96, 0x98, 0x9A, 0x9C, 0x9E,
	/* 50 */ 0xA0, 0xA2, 0xA4, 0xA6, 0xA8, 0xAA, 0xAC, 0xAE,
	/* 58 */ 0xB0, 0xB2, 0xB4, 0xB6, 0xB8, 0xBA, 0xBC, 0xBE,
	/* 60 */ 0xC0, 0xC2, 0xC4, 0xC6, 0xC8, 0xCA, 0xCC, 0xCE,
	/* 68 */ 0xD0, 0xD2, 0xD4, 0xD6, 0xD8, 0xDA, 0xDC, 0xDE,
	/* 70 */ 0xE0, 0xE2, 0xE4, 0xE6, 0xE8, 0xEA, 0xEC, 0xEE,
	/* 78 */ 0xF0, 0xF2, 0xF4, 0xF6, 0xF8, 0xFA, 0xFC, 0xFE,
	/* 80 */ 0x1B, 0x19, 0x1F, 0x1D, 0x13, 0x11, 0x17, 0x15,
	/* 88 */ 0x0B, 0x09, 0x0F, 0x0D, 0x03, 0x01, 0x07, 0x05,
	/* 90 */ 0x3B, 0x39, 0x3F, 0x3D, 0x33, 0x31, 0x37, 0x35,
	/* 98 */ 0x2B, 0x29, 0x2F, 0x2D, 0x23, 0x21, 0x27, 0x25,
	/* a0 */ 0x5B, 0x59, 0x5F, 0x5D, 0x53, 0x51, 0x57, 0x55,
	/* a8 */ 0x4B, 0x49, 0x4F, 0x4D, 0x43, 0x41, 0x47, 0x45,
	/* b0 */ 0x7B, 0x79, 0x7F, 0x7D, 0x73, 0x71, 0x77, 0x75,
	/* b8 */ 0x6B, 0x69, 0x6F, 0x6D, 0x63, 0x61, 0x67, 0x65,
	/* c0 */ 0x9B, 0x99, 0x9F, 0x9D, 0x93, 0x91, 0x97, 0x95,
	/* c8 */ 0x8B, 0x89, 0x8F, 0x8D, 0x83, 0x81, 0x87, 0x85,
	/* d0 */ 0xBB, 0xB9, 0xBF, 0xBD, 0xB3, 0xB1, 0xB7, 0xB5,
	/* d8 */ 0xAB, 0xA9, 0xAF, 0xAD, 0xA3, 0xA1, 0xA7, 0xA5,
	/* e0 */ 0xDB, 0xD9, 0xDF, 0xDD, 0xD3, 0xD1, 0xD7, 0xD5,
	/* e8 */ 0xCB, 0xC9, 0xCF, 0xCD, 0xC3, 0xC1, 0xC7, 0xC5,
	/* f0 */ 0xFB, 0xF9, 0xFF, 0xFD, 0xF3, 0xF1, 0xF7, 0xF5,
	/* f8 */ 0xEB, 0xE9, 0xEF, 0xED, 0xE3, 0xE1, 0xE7, 0xE5,
	};

/**
 * Structure for one AES word.
 */
typedef struct
	{
	uint8_t r0, r1, r2, r3;
	}
	word_t;

// Load word w from the given integer.
__device__ void wordLoad
	(word_t *w,
	 uint32_t value)
	{
	w->r0 = (uint8_t)(value >> 24);
	w->r1 = (uint8_t)(value >> 16);
	w->r2 = (uint8_t)(value >>  8);
	w->r3 = (uint8_t)(value);
	}

// Unload word w into an integer and return it.
__device__ uint32_t wordUnload
	(word_t *w)
	{
	return
		(((uint32_t)(w->r0)) << 24) |
		(((uint32_t)(w->r1)) << 16) |
		(((uint32_t)(w->r2)) <<  8) |
		(((uint32_t)(w->r3))      );
	}

// Exclusive-or the given round constant into word w.
__device__ void wordXorRcon
	(word_t *w,
	 uint8_t rcon)
	{
	w->r0 ^= rcon;
	}

// Exclusive-or word x into word w.
__device__ void wordXor
	(word_t *w,
	 word_t *x)
	{
	w->r0 ^= x->r0;
	w->r1 ^= x->r1;
	w->r2 ^= x->r2;
	w->r3 ^= x->r3;
	}

// Perform the SubWord oepration on word w.
__device__ void wordSubWord
	(word_t *w)
	{
	w->r0 = sbox[w->r0];
	w->r1 = sbox[w->r1];
	w->r2 = sbox[w->r2];
	w->r3 = sbox[w->r3];
	}

// Perform the RotWord operation on word w.
__device__ void wordRotWord
	(word_t *w)
	{
	uint8_t t = w->r0;
	w->r0 = w->r1;
	w->r1 = w->r2;
	w->r2 = w->r3;
	w->r3 = t;
	}

/**
 * Structure for the AES key schedule.
 * For AES-128: #define NK 4
 * For AES-192: #define NK 6
 * For AES-256: #define NK 8
 */
typedef struct
	{
	word_t w[NK];
	uint8_t rcon;
	int n;
	int i;
	}
	keysched_t;

// Load key schedule k from the given bigint.
__device__ void keyLoad
	(keysched_t *k,
	 uint32_t *bigint)
	{
	int j;
	for (j = 0; j < NK; ++ j)
		wordLoad (&k->w[j], bigint[NK-1-j]);
	k->rcon = 1;
	k->n = NK;
	k->i = 0;
	}

// Returns (x + 1) mod NK.
__device__ int keyIncr
	(int x)
	{
	++ x;
	return (x == NK) ? 0 : x;
	}

// Returns (x - 1) mod NK.
__device__ int keyDecr
	(int x)
	{
	-- x;
	return (x == -1) ? NK - 1 : x;
	}

// Compute the next word in key schedule k.
__device__ void keyUpdate
	(keysched_t *k)
	{
	word_t temp = k->w[keyDecr(k->i)];
	if (k->i == 0)
		{
		wordRotWord (&temp);
		wordSubWord (&temp);
		wordXorRcon (&temp, k->rcon);
		k->rcon = xtime[k->rcon];
		}
	else if (NK > 6 && k->i == 4)
		{
		wordSubWord (&temp);
		}
	wordXor (&k->w[k->i], &temp);
	}

// Get the next subkey word from key schedule k.
__device__ word_t* keyNext
	(keysched_t *k)
	{
	word_t* rv;
	if (k->n == 0)
		keyUpdate (k);
	else
		-- k->n;
	rv = &k->w[k->i];
	k->i = keyIncr (k->i);
	return rv;
	}

/**
 * Structure for the AES state.
 */
typedef struct
	{
	word_t c0, c1, c2, c3;
	}
	state_t;

// Load state s from the given bigint.
__device__ void stateLoad
	(state_t *s,
	 uint32_t *bigint)
	{
	wordLoad (&s->c0, bigint[3]);
	wordLoad (&s->c1, bigint[2]);
	wordLoad (&s->c2, bigint[1]);
	wordLoad (&s->c3, bigint[0]);
	}

// Unload state s into the given bigint.
__device__ void stateUnload
	(state_t *s,
	 uint32_t *bigint)
	{
	bigint[3] = wordUnload (&s->c0);
	bigint[2] = wordUnload (&s->c1);
	bigint[1] = wordUnload (&s->c2);
	bigint[0] = wordUnload (&s->c3);
	}

// Perform the AddRoundKey operation on state s.
__device__ void stateAddRoundKey
	(state_t *s,
	 keysched_t *keySched)
	{
	wordXor (&s->c0, keyNext (keySched));
	wordXor (&s->c1, keyNext (keySched));
	wordXor (&s->c2, keyNext (keySched));
	wordXor (&s->c3, keyNext (keySched));
	}

// Perform the SubBytes operation on state s.
__device__ void stateSubBytes
	(state_t *s)
	{
	wordSubWord (&s->c0);
	wordSubWord (&s->c1);
	wordSubWord (&s->c2);
	wordSubWord (&s->c3);
	}

// Perform the ShiftRows operation on state s.
__device__ void stateShiftRows
	(state_t *s)
	{
	uint32_t t;

	// Row 1.
	t = s->c0.r1;
	s->c0.r1 = s->c1.r1;
	s->c1.r1 = s->c2.r1;
	s->c2.r1 = s->c3.r1;
	s->c3.r1 = t;

	// Row 2.
	t = s->c0.r2;
	s->c0.r2 = s->c2.r2;
	s->c2.r2 = t;
	t = s->c1.r2;
	s->c1.r2 = s->c3.r2;
	s->c3.r2 = t;

	// Row 3.
	t = s->c3.r3;
	s->c3.r3 = s->c2.r3;
	s->c2.r3 = s->c1.r3;
	s->c1.r3 = s->c0.r3;
	s->c0.r3 = t;
	}

// Perform the MixColumns operation on state s.
__device__ void stateMixColumns
	(state_t *s)
	{
	uint32_t t, sum;

	// Column 0.
	t = s->c0.r0;
	sum = s->c0.r0 ^ s->c0.r1 ^ s->c0.r2 ^ s->c0.r3;
	s->c0.r0 ^= xtime [s->c0.r0 ^ s->c0.r1] ^ sum;
	s->c0.r1 ^= xtime [s->c0.r1 ^ s->c0.r2] ^ sum;
	s->c0.r2 ^= xtime [s->c0.r2 ^ s->c0.r3] ^ sum;
	s->c0.r3 ^= xtime [s->c0.r3 ^ t       ] ^ sum;

	// Column 1.
	t = s->c1.r0;
	sum = s->c1.r0 ^ s->c1.r1 ^ s->c1.r2 ^ s->c1.r3;
	s->c1.r0 ^= xtime [s->c1.r0 ^ s->c1.r1] ^ sum;
	s->c1.r1 ^= xtime [s->c1.r1 ^ s->c1.r2] ^ sum;
	s->c1.r2 ^= xtime [s->c1.r2 ^ s->c1.r3] ^ sum;
	s->c1.r3 ^= xtime [s->c1.r3 ^ t       ] ^ sum;

	// Column 2.
	t = s->c2.r0;
	sum = s->c2.r0 ^ s->c2.r1 ^ s->c2.r2 ^ s->c2.r3;
	s->c2.r0 ^= xtime [s->c2.r0 ^ s->c2.r1] ^ sum;
	s->c2.r1 ^= xtime [s->c2.r1 ^ s->c2.r2] ^ sum;
	s->c2.r2 ^= xtime [s->c2.r2 ^ s->c2.r3] ^ sum;
	s->c2.r3 ^= xtime [s->c2.r3 ^ t       ] ^ sum;

	// Column 3.
	t = s->c3.r0;
	sum = s->c3.r0 ^ s->c3.r1 ^ s->c3.r2 ^ s->c3.r3;
	s->c3.r0 ^= xtime [s->c3.r0 ^ s->c3.r1] ^ sum;
	s->c3.r1 ^= xtime [s->c3.r1 ^ s->c3.r2] ^ sum;
	s->c3.r2 ^= xtime [s->c3.r2 ^ s->c3.r3] ^ sum;
	s->c3.r3 ^= xtime [s->c3.r3 ^ t       ] ^ sum;
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
	int r;
	state_t state;
	keysched_t subkey;

	// A input = plaintext.
	stateLoad (&state, A);

	// B input = key.
	keyLoad (&subkey, B);

	// Do all but the last round.
	stateAddRoundKey (&state, &subkey);
	for (r = 0; r <= R - 2; ++ r)
		{
		stateSubBytes (&state);
		stateShiftRows (&state);
		stateMixColumns (&state);
		stateAddRoundKey (&state, &subkey);
		stateUnload (&state, biElem3D (Csize, C, NB, R, a, b, r));
		}

	// Do last round.
	stateSubBytes (&state);
	stateShiftRows (&state);
	stateAddRoundKey (&state, &subkey);
	stateUnload (&state, biElem3D (Csize, C, NB, R, a, b, R - 1));
	}

#include "../crst/FunctionKernel.cu"
