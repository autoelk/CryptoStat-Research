//******************************************************************************
//
// File:    FunctionKernel.cu
// Unit:    FunctionKernel CUDA functions
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
 * The CUDA kernel that evaluates a cryptographic function.
 *
 * @author  Alan Kaminsky
 * @version 15-Sep-2017
 */

///**
// * Evaluate the cryptographic function for the given combination of inputs. On
// * return, the function's output for input <TT>A</TT>, input <TT>B</TT>, and
// * each round <TT>r</TT> has been stored in <TT>C[a][b][r]</TT>.
// *
// * @param  NA     Number of <I>A</I> inputs.
// * @param  Asize  <I>A</I> input bigint size in words.
// * @param  A      <I>A</I> input value.
// * @param  NB     Number of <I>B</I> inputs.
// * @param  Bsize  <I>B</I> input bigint size in words.
// * @param  B      <I>B</I> input value.
// * @param  R      Number of rounds.
// * @param  Csize  <I>C</I> output bigint size in words.
// * @param  C      3-D array of <I>C</I> output values.
// * @param  a      <I>A</I> input index.
// * @param  b      <I>B</I> input index.
// */
//__device__ void evaluate
//	(int NA,
//	 int Asize,
//	 uint32_t* A,
//	 int NB,
//	 int Bsize,
//	 uint32_t* B,
//	 int R,
//	 int Csize,
//	 uint32_t* C,
//	 int a,
//	 int b)

/**
 * Evaluate the cryptographic function for all combinations of inputs. The
 * <TT>evaluateFunction()</TT> kernel function is called with a 1-D grid of 1-D
 * blocks.
 * <P>
 * The kernel evaluates the function for each combination of an <I>A</I> input
 * and a <I>B</I> input in parallel, partitioning the evaluations among all the
 * threads in all the blocks. The kernel stores the function's output for each
 * <I>A</I> input, each <I>B</I> input, and each round in the <I>C</I> output
 * array.
 *
 * @param  NA     Number of <I>A</I> inputs.
 * @param  Asize  <I>A</I> input bigint size in words.
 * @param  A      Array of <I>A</I> input values.
 * @param  NB     Number of <I>B</I> inputs.
 * @param  Bsize  <I>B</I> input bigint size in words.
 * @param  B      Array of <I>B</I> input values.
 * @param  R      Number of rounds.
 * @param  Csize  <I>C</I> output bigint size in words.
 * @param  C      3-D array of <I>C</I> output values.
 */
extern "C" __global__ void evaluateFunction
	(int NA,
	 int Asize,
	 uint32_t* A,
	 int NB,
	 int Bsize,
	 uint32_t* B,
	 int R,
	 int Csize,
	 uint32_t* C)
	{
	int NA_NB, nthr, rank, i, a, b;
	uint32_t* A_a;
	uint32_t* B_b;

	NA_NB = NA*NB;

	// Determine number of threads and this thread's rank.
	nthr = gridDim.x*blockDim.x;
	rank = blockIdx.x*blockDim.x + threadIdx.x;

	// Evaluate the function.
	for (i = rank; i < NA_NB; i += nthr)
		{
		a = i/NB;
		b = i%NB;
		A_a = biElem (Asize, A, a);
		B_b = biElem (Bsize, B, b);
		evaluate (NA, Asize, A_a, NB, Bsize, B_b, R, Csize, C, a, b);
		}
	}
