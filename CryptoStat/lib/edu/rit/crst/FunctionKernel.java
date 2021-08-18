//******************************************************************************
//
// File:    FunctionKernel.java
// Package: edu.rit.crst
// Unit:    Interface edu.rit.crst.FunctionKernel
//
// This Java source file is copyright (C) 2017 by Alan Kaminsky. All rights
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
import edu.rit.gpu.Kernel;

/**
 * Interface FunctionKernel specifies the interface for a GPU kernel function
 * that evaluates a cryptographic {@linkplain Function} on a series of inputs.
 *
 * @author  Alan Kaminsky
 * @version 25-Apr-2017
 */
public interface FunctionKernel
	extends Kernel
	{

	/**
	 * Evaluate the cryptographic function for all combinations of inputs. The
	 * <TT>evaluateFunction()</TT> kernel function is called with a 1-D grid of
	 * 1-D blocks.
	 *
	 * @param  NA           Number of <I>A</I> inputs.
	 * @param  Asize        <I>A</I> input bigint size in words.
	 * @param  A            Array of <I>A</I> input values.
	 * @param  NB           Number of <I>B</I> inputs.
	 * @param  Bsize        <I>B</I> input bigint size in words.
	 * @param  B            Array of <I>B</I> input values.
	 * @param  R            Number of rounds.
	 * @param  Csize        <I>C</I> output bigint size in words.
	 * @param  C            3-D array of <I>C</I> output values.
	 */
	public void evaluateFunction
		(int NA,
		 int Asize,
		 GpuIntArray A,
		 int NB,
		 int Bsize,
		 GpuIntArray B,
		 int R,
		 int Csize,
		 GpuIntArray C);

	}
