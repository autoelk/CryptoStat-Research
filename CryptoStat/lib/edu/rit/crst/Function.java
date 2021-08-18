//******************************************************************************
//
// File:    Function.java
// Package: edu.rit.crst
// Unit:    Class edu.rit.crst.Function
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

import edu.rit.gpu.Gpu;
import edu.rit.util.BigInt;
import edu.rit.util.GpuBigIntArray;
import java.io.IOException;

/**
 * Class Function is the abstract base class for a cryptographic function. A
 * cryptographic function has two inputs <I>A</I> and <I>B</I> and one output
 * <I>C</I>, each of which is a {@linkplain BigInt}.
 *
 * @author  Alan Kaminsky
 * @version 14-Sep-2017
 */
public abstract class Function
	extends Computation
	{

// Exported constructors.

	/**
	 * Construct a new cryptographic function object.
	 */
	public Function()
		{
		super();
		}

// Exported operations.

	/**
	 * Get a constructor expression for this cryptographic function. The
	 * constructor expression can be passed to the {@link
	 * edu.rit.util.Instance#newInstance(String)
	 * edu.rit.util.Instance.newInstance()} method to construct an object that
	 * is the same as this cryptographic function.
	 *
	 * @return  Constructor expression.
	 */
	public abstract String constructor();

	/**
	 * Get a description of this cryptographic function.
	 *
	 * @return  Description.
	 */
	public abstract String description();

	/**
	 * Get a description of input <I>A</I> for this cryptographic function.
	 *
	 * @return  Input <I>A</I> description.
	 */
	public abstract String A_description();

	/**
	 * Get the bit size of input <I>A</I> for this cryptographic function.
	 *
	 * @return  Input <I>A</I> bit size.
	 */
	public abstract int A_bitSize();

	/**
	 * Get a description of input <I>B</I> for this cryptographic function.
	 *
	 * @return  Input <I>B</I> description.
	 */
	public abstract String B_description();

	/**
	 * Get the bit size of input <I>B</I> for this cryptographic function.
	 *
	 * @return  Input <I>B</I> bit size.
	 */
	public abstract int B_bitSize();

	/**
	 * Get a description of output <I>C</I> for this cryptographic function.
	 *
	 * @return  Output <I>C</I> description.
	 */
	public abstract String C_description();

	/**
	 * Get the bit size of output <I>C</I> for this cryptographic function.
	 *
	 * @return  Output <I>C</I> bit size.
	 */
	public abstract int C_bitSize();

	/**
	 * Get the number of rounds for this cryptographic function.
	 *
	 * @return  Number of rounds.
	 */
	public abstract int rounds();

	/**
	 * Evaluate this cryptographic function. The function is evaluated on inputs
	 * <I>A</I> and <I>B</I>, and the output of each round is stored in the
	 * <I>C</I> array.
	 * <P>
	 * <I>Note:</I> The <TT>evaluate()</TT> method is performed on the CPU in a
	 * single thread. It is intended as a cross-check for the GPU kernel.
	 *
	 * @param  A  Input <I>A</I>.
	 * @param  B  Input <I>B</I>.
	 * @param  C  Array of outputs <I>C</I>, indexed by round.
	 */
	public abstract void evaluate
		(BigInt A,
		 BigInt B,
		 BigInt[] C);

	/**
	 * Get the GPU kernel for this cryptographic function.
	 *
	 * @param  gpu  GPU accelerator.
	 *
	 * @return  Kernel.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 */
	public FunctionKernel functionKernel
		(Gpu gpu)
		throws IOException
		{
		return kernel (gpu, FunctionKernel.class);
		}

	}
