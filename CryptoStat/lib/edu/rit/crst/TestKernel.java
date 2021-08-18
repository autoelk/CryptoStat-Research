//******************************************************************************
//
// File:    TestKernel.java
// Package: edu.rit.crst
// Unit:    Interface edu.rit.crst.TestKernel
//
// This Java source file is copyright (C) 2018 by Alan Kaminsky. All rights
// reserved. For further information, contact the author, Alan Kaminsky, at
// ark@cs.rit.edu.
//
// This Java source file is part of the CryptoStat Library ("CryptoStat").
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

package edu.rit.crst;

import edu.rit.gpu.GpuIntArray;
import edu.rit.gpu.GpuIntMatrix;
import edu.rit.gpu.GpuStructMatrix;
import edu.rit.gpu.Kernel;

/**
 * Interface TestKernel specifies the interface for a GPU kernel function that
 * computes the logarithms of the Bayes factors for all rounds and bit groups
 * for a {@linkplain Test} object.
 *
 * @author  Alan Kaminsky
 * @version 19-Feb-2018
 */
public interface TestKernel
	extends Kernel
	{

	/**
	 * Compute the log Bayes factors. The <TT>computeTest()</TT> kernel function
	 * is called with a 1-D grid of 1-D blocks. The rounds and bit groups are
	 * partitioned among all the threads in all the blocks. Each thread computes
	 * the log Bayes factors for certain rounds/bit groups, storing the results
	 * in the <TT>testResult</TT> matrix.
	 *
	 * @param  NA
	 *     Number of <I>A</I> inputs.
	 * @param  NB
	 *     Number of <I>B</I> inputs.
	 * @param  R
	 *     Number of rounds.
	 * @param  Csize
	 *     <I>C</I> output bigint size in words.
	 * @param  C
	 *     3-D array of <I>C</I> output values.
	 * @param  NBG
	 *     Number of bit groups.
	 * @param  testResult
	 *     2-D array of test results; <TT>testResult[r][g]</TT> is the log Bayes
	 *     factor for round <TT>r</TT> bit group <TT>g</TT>,
	 *     0&nbsp;&le;&nbsp;<TT>r</TT>&nbsp;&le;&nbsp;<TT>R</TT>&minus;1,
	 *     0&nbsp;&le;&nbsp;<TT>g</TT>&nbsp;&le;&nbsp;<TT>NBG</TT>&minus;1.
	 * @param  G
	 *     Bit size of bit groups.
	 * @param  pos
	 *     2-D array of bit positions in the bit groups; <TT>pos[g][p]</TT> is
	 *     the original bit position for bit group <TT>g</TT> bit group position
	 *     <TT>p</TT>,
	 *     0&nbsp;&le;&nbsp;<TT>g</TT>&nbsp;&le;&nbsp;<TT>NBG</TT>&minus;1,
	 *     0&nbsp;&le;&nbsp;<TT>p</TT>&nbsp;&le;&nbsp;<TT>G</TT>&minus;1.
	 * @param  a
	 *     Input <I>A</I> index,
	 *     0&nbsp;&le;&nbsp;<TT>a</TT>&le;&nbsp;<TT>NA</TT>&minus;1, or &minus;1
	 *     to iterate over all <I>A</I> inputs.
	 * @param  b
	 *     Input <I>B</I> index,
	 *     0&nbsp;&le;&nbsp;<TT>b</TT>&le;&nbsp;<TT>NB</TT>&minus;1, or &minus;1
	 *     to iterate over all <I>B</I> inputs.
	 */
	public void computeTest
		(int NA,
		 int NB,
		 int R,
		 int Csize,
		 GpuIntArray C,
		 int NBG,
		 GpuStructMatrix<TestResult> testResult,
		 int G,
		 GpuIntMatrix pos,
		 int a,
		 int b);

	}
