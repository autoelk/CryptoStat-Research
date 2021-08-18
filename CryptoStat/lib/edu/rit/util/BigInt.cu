//******************************************************************************
//
// File:    BigInt.cu
// Unit:    BigInt CUDA functions
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

#ifndef __BIGINT_CU__
#define __BIGINT_CU__

#include <stdint.h>

/**
 * A big integer (bigint) is an arbitrary-size unsigned integer. A bigint is
 * implemented as an array of unsigned ints (type uint32_t) holding the value in
 * little-endian order. In the functions below, the first argument is always the
 * size (in unsigned ints) of the bigint or bigints being manipulated. Each
 * bigint itself is passed as a pointer to the first unsigned int of the value.
 *
 * @author  Alan Kaminsky
 * @version 24-Aug-2017
 */

/**
 * Returns a pointer to the element at the given index in the given array of
 * bigints.
 */
__device__ uint32_t* biElem
	(int size,         // Bigint size
	 uint32_t* array,  // Pointer to array of bigints
	 int index)        // Index of array element
	{
	return &array[size*index];
	}

/**
 * Returns a pointer to the element at the given indexes in the given 2-D array
 * of bigints.
 */
__device__ uint32_t* biElem2D
	(int size,        // Bigint size
	 uint32_t* array, // Pointer to 2-D array of bigints
	 int len2,        // Length of second dimension
	 int index1,      // First index of array element
	 int index2)      // Second index of array element
	{
	return &array[size*(index1*len2 + index2)];
	}

/**
 * Returns a pointer to the element at the given indexes in the given 3-D array
 * of bigints.
 */
__device__ uint32_t* biElem3D
	(int size,        // Bigint size
	 uint32_t* array, // Pointer to 3-D array of bigints
	 int len2,        // Length of second dimension
	 int len3,        // Length of third dimension
	 int index1,      // First index of array element
	 int index2,      // Second index of array element
	 int index3)      // Third index of array element
	{
	return &array[size*((index1*len2 + index2)*len3 + index3)];
	}

/**
 * Set bigint a to zero.
 */
__device__ void biClear
	(int size,     // Bigint size
	 uint32_t* a)  // Pointer to bigint
	{
	for (int i = 0; i < size; ++ i)
		a[i] = 0;
	}

/**
 * Set bigint a to bigint b.
 */
__device__ void biAssign
	(int size,     // Bigint size
	 uint32_t* a,  // Pointer to bigint a
	 uint32_t* b)  // Pointer to bigint b
	{
	for (int i = 0; i < size; ++ i)
		a[i] = b[i];
	}

/**
 * Set bigint a to the exclusive-or of itself and bigint b.
 */
__device__ void biXor
	(int size,     // Bigint size
	 uint32_t* a,  // Pointer to bigint a
	 uint32_t* b)  // Pointer to bigint b
	{
	for (int i = 0; i < size; ++ i)
		a[i] ^= b[i];
	}

/**
 * Get the bit at the given position in bigint a.
 */
__device__ int biGetBit
	(uint32_t* a,  // Pointer to bigint a
	 int index)    // Bit index in bigint a
	{
	return (a[index >> 5] >> (index & 31)) & 1;
	}

/**
 * Put the given value into the bit at the given position in bigint a.
 */
__device__ void biPutBit
	(uint32_t* a,  // Pointer to bigint a
	 int index,    // Bit index in bigint a
	 int bit)      // Bit value to put (0 or 1)
	{
	if (bit == 0)
		a[index >> 5] &= ~(1 << (index & 31));
	else
		a[index >> 5] |=  (1 << (index & 31));
	}

/**
 * Set the bit at the given index of bigint a to the exclusive-or of itself and
 * the given bit.
 */
__device__ void biXorBit
	(uint32_t* a,  // Pointer to bigint a
	 int index,    // Bit index in bigint a
	 int bit)      // Bit value to exclusive-or (0 or 1)
	{
	a[index >> 5] ^= bit << (index & 31);
	}

/**
 * Scatter bits of long integer val into the given positions of bigint a.
 */
__device__ void biScatterLongBits
	(uint32_t* a,   // Pointer to bigint a
	 int n,         // Number of bits to scatter
	 int* pos,      // Array of bit positions in bigint a
	 uint64_t val)  // Long integer to scatter
	{
	int i;
	for (i = 0; i < n; ++ i)
		{
		biPutBit (a, pos[i], (int)(val & 1));
		val >>= 1;
		}
	}

/**
 * Scatter bits of bigint val into the given positions of bigint a.
 */
__device__ void biScatterBigIntBits
	(uint32_t* a,    // Pointer to bigint a
	 int n,          // Number of bits to scatter
	 int* pos,       // Array of bit positions in bigint a
	 uint32_t* val)  // Long integer to scatter
	{
	int i;
	for (i = 0; i < n; ++ i)
		biPutBit (a, pos[i], biGetBit (val, i));
	}

/**
 * Returns a group of noncontiguous bits from bigint a. Assumes 0 <= n <= 32;
 * for 0 <= i <= n-1, 0 <= pos[i] < size*32.
 */
__device__ int biGatherBits
	(uint32_t* a,  // Pointer to bigint
	 int n,        // Number of bits to gather
	 int* pos)     // Array of bit positions to gather
	{
	int bitGroup = 0;
	int d = 0;
	for (int i = 0; i < n; ++ i)
		{
		bitGroup |= ((a[pos[i] >> 5] >> (pos[i] & 31)) & 1) << d;
		++ d;
		}
	return bitGroup;
	}

/**
 * Unpack bigint a into a portion of the given integer array in big-endian
 * order.
 */
__device__ void biUnpackBigEndian
	(int size,        // Bigint size
	 uint32_t* a,     // Pointer to bigint a
	 uint32_t* array, // Pointer to integer array
	 int off)         // First array index to unpack
	{
	int i, j;
	for (i = size - 1, j = off; i >= 0; -- i, ++ j)
		array[j] = a[i];
	}

/**
 * Unpack bigint a into a portion of the given long integer array in big-endian
 * order.
 */
__device__ void biUnpackLongBigEndian
	(int size,        // Bigint size
	 uint32_t* a,     // Pointer to bigint a
	 uint64_t* array, // Pointer to long integer array
	 int off)         // First array index to unpack
	{
	int i;
	int N = size/2;
	for (i = 0; i < N; ++ i)
		array[off+i] = ((uint64_t)(a[size-2*i-2])) |
			((uint64_t)(a[size-2*i-1]) << 32);
	}

/**
 * Print bigint a on the standard output.
 */
__device__ void biPrint
	(int size,     // Bigint size
	 uint32_t* a)  // Pointer to bigint a
	{
	for (int i = size - 1; i >= 0; -- i)
		printf ("%08x", a[i]);
	}

#endif
